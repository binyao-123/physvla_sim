#!/usr/bin/env python3
"""Predict full-dataset joint dynamics and write them to per-asset info1.json.

Reproduce (cd physvla_sim/Augment_code; export DASHSCOPE_API_KEY='sk-...'):

  # Preview a small subset without calling Qwen or writing files.
  .venv/bin/python dyn/fill_info1_dynamics.py --limit 5 --dry-run

  # Predict every dataset asset and write damping/stiffness/effort_limit to info1.json.
  .venv/bin/python dyn/fill_info1_dynamics.py --sleep 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUGMENT_ROOT = Path(__file__).resolve().parents[1]
if str(AUGMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUGMENT_ROOT))

from merge_info1 import (
    DEFAULT_DYNAMICS,
    DEFAULT_VLM_BASE,
    build_info1,
    iter_assets,
    load_json,
    write_info1,
)
from prompt.load_prompt import (
    DEFAULT_DATASET_ROOT,
    find_category_scale_image,
    list_movable_joints,
)
from predict_joint_dynamics import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    append_error,
    call_vlm,
    load_prompt,
    parse_vlm_json,
    render_prompt,
    validate_prediction,
)

DEFAULT_BENCH = AUGMENT_ROOT / "fitr_bench.json"
DEFAULT_VLM_RESULTS = AUGMENT_ROOT / "experiments" / "vlm_results.jsonl"
DEFAULT_ERROR_LOG = AUGMENT_ROOT / "experiments" / "full_dynamics_errors.jsonl"
DEFAULT_PROGRESS_LOG = AUGMENT_ROOT / "experiments" / "full_dynamics_progress.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_references(bench_path: Path) -> dict[str, dict[str, Any]]:
    bench = load_json(bench_path)
    refs: dict[str, dict[str, Any]] = {}
    for category, category_data in bench["categories"].items():
        assets = category_data.get("assets") or []
        if not assets:
            raise ValueError(f"No reference asset for category {category}.")
        reference = assets[0]
        try:
            reference["scale"]["L_real_cm_gt"] = float(
                reference["scale"]["L_real_cm_gt"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Reference scale missing for {category}.") from exc
        refs[category] = reference
    return refs


def load_latest_vlm_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("status") != "ok" or not record.get("result"):
                continue
            key = (record["category"], str(record["asset_id"]))
            previous = latest.get(key)
            if previous is None or record.get("timestamp", "") >= previous.get("timestamp", ""):
                latest[key] = record
    return latest


def raw_longest_edge_cm(record: dict[str, Any] | None) -> float | None:
    if not record or not record.get("raw_response"):
        return None
    try:
        return float(parse_vlm_json(str(record["raw_response"]))["longest_edge_cm"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def resolve_target_size_cm(
    *,
    category: str,
    asset_id: str,
    record: dict[str, Any] | None,
    reference: dict[str, Any],
    reference_record: dict[str, Any] | None,
) -> float:
    result = record.get("result") if record else None
    value = result.get("longest_edge_cm") if result else None
    if value is None:
        raise ValueError("No successful VLM longest_edge_cm result.")
    size_cm = float(value)
    reference_cm = float(reference["scale"]["L_real_cm_gt"])
    ratio = size_cm / reference_cm
    if 0.25 <= ratio <= 4.0:
        return size_cm

    target_raw_cm = raw_longest_edge_cm(record)
    reference_raw_cm = raw_longest_edge_cm(reference_record)
    if target_raw_cm and reference_raw_cm and reference_raw_cm > 0:
        return reference_cm * target_raw_cm / reference_raw_cm
    raise ValueError(
        f"Implausible scale result {size_cm:.3f} cm for {category}/{asset_id}; "
        "no raw-VLM scale fallback is available."
    )


def load_successful_keys(path: Path, *, model: str) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("status") in {"ok", "skipped"} and record.get("model") == model:
                keys.add((record["category"], str(record["asset_id"])))
    return keys


def load_error_keys(path: Path, *, model: str) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("model") == model:
                keys.add((record["category"], str(record["asset_id"])))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict full-dataset joint dynamics and write info1.json."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--vlm-base", type=Path, default=DEFAULT_VLM_BASE)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--vlm-results", type=Path, default=DEFAULT_VLM_RESULTS)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based offset in numerical category/asset order.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Additional VLM attempts after a per-asset request or JSON failure.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--progress-log", type=Path, default=DEFAULT_PROGRESS_LOG)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Only process assets listed in --error-log for this model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip assets recorded as successful for this model in --progress-log.",
    )
    args = parser.parse_args()

    if args.asset_id and (not args.category or len(args.category) != 1):
        parser.error("--asset-id requires exactly one --category.")
    if args.start_index < 0 or args.retries < 0:
        parser.error("--start-index and --retries must be nonnegative.")

    vlm_base = load_json(args.vlm_base)
    dynamics = load_json(args.dynamics)
    references = load_references(args.bench_path)
    scale_records = load_latest_vlm_records(args.vlm_results)
    system, user_template = load_prompt(args.prompt)

    if args.asset_id:
        category = args.category[0]
        category_dir = vlm_base["categories"][category]["category_dir"]
        work = [
            (
                category,
                args.asset_id,
                args.dataset_root / category_dir / args.asset_id,
            )
        ]
    else:
        work = list(iter_assets(vlm_base, args.dataset_root, categories=args.category))
    work = work[args.start_index:]
    if args.retry_errors:
        failed = load_error_keys(args.error_log, model=args.model)
        work = [item for item in work if (item[0], item[1]) in failed]
    if args.resume:
        done = load_successful_keys(args.progress_log, model=args.model)
        work = [item for item in work if (item[0], item[1]) not in done]
    if args.limit is not None:
        work = work[: args.limit]

    if not args.dry_run and not os.getenv("DASHSCOPE_API_KEY"):
        print("DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 1
    api_key = os.getenv("DASHSCOPE_API_KEY", "")

    print(f"planned={len(work)} model={args.model} info1=direct-write")
    written = errors = 0
    for index, (category, asset_id, asset_dir) in enumerate(work, start=1):
        raw: str | None = None
        try:
            if not asset_dir.is_dir():
                raise FileNotFoundError(f"Asset directory not found: {asset_dir}")
            reference = references[category]
            scale_record = scale_records.get((category, asset_id))
            target_size_cm = resolve_target_size_cm(
                category=category,
                asset_id=asset_id,
                record=scale_record,
                reference=reference,
                reference_record=scale_records.get(
                    (category, str(reference["asset_id"]))
                ),
            )
            scale_result = dict(scale_record["result"])
            scale_result["longest_edge_cm"] = target_size_cm

            category_dir = vlm_base["categories"][category]["category_dir"]
            reference_dir = args.dataset_root / category_dir / str(reference["asset_id"])
            joints = list_movable_joints(asset_dir)
            if not joints:
                if not args.dry_run:
                    append_error(
                        args.progress_log,
                        {
                            "status": "skipped",
                            "category": category,
                            "asset_id": asset_id,
                            "model": args.model,
                            "timestamp": utc_now(),
                            "reason": "No movable joints found.",
                        },
                    )
                print(f"[{index}/{len(work)}] skipped {category}/{asset_id}: no movable joints")
                continue

            if args.dry_run:
                print(
                    f"[dry] {category}/{asset_id} reference={reference['asset_id']} "
                    f"size_cm={target_size_cm:.2f} joints={len(joints)}"
                )
                continue

            for attempt in range(args.retries + 1):
                try:
                    raw = call_vlm(
                        system=system,
                        user=render_prompt(
                            user_template,
                            category,
                            asset_id,
                            joints,
                            reference_size_cm=float(reference["scale"]["L_real_cm_gt"]),
                            target_size_cm=target_size_cm,
                        ),
                        image_paths=[
                            find_category_scale_image(reference_dir, category),
                            find_category_scale_image(asset_dir, category),
                        ],
                        model=args.model,
                        api_key=api_key,
                    )
                    prediction = validate_prediction(
                        category,
                        joints,
                        parse_vlm_json(raw),
                        clamp_to_range=True,
                    )
                    break
                except Exception:
                    if attempt == args.retries:
                        raise
                    time.sleep(args.sleep)
            predicted_joints = {
                joint["joint_name"]: joint for joint in prediction["joints"]
            }
            previous_affordances: dict[str, dict[str, Any]] = {}
            existing_info_path = asset_dir / "info1.json"
            if existing_info_path.is_file():
                existing_info = json.loads(existing_info_path.read_text(encoding="utf-8"))
                previous_affordances = {
                    str(aff["joint_name"]): aff
                    for aff in existing_info.get("affordances", [])
                    if aff.get("joint_name")
                }

            info1 = build_info1(
                category,
                asset_id,
                asset_dir,
                vlm_base=vlm_base,
                dynamics=dynamics,
                vlm_result=scale_result,
            )
            for info_joint in info1["joints"]:
                predicted = predicted_joints[info_joint["joint_name"]]
                for field in ("damping", "stiffness", "effort_limit"):
                    info_joint[field] = predicted[field]
            for affordance in info1["affordances"]:
                previous = previous_affordances.get(str(affordance["joint_name"]))
                if not previous:
                    continue
                position = previous.get("position_xyz")
                axis = previous.get("contact_axis_xyz")
                if isinstance(position, list) and len(position) == 3 and all(
                    value is not None for value in position
                ):
                    affordance["position_xyz"] = position
                if isinstance(axis, list) and len(axis) == 3 and all(
                    value is not None for value in axis
                ):
                    affordance["contact_axis_xyz"] = axis
            write_info1(asset_dir, info1, dry_run=False)
            written += 1
            append_error(
                args.progress_log,
                {
                    "status": "ok",
                    "category": category,
                    "asset_id": asset_id,
                    "model": args.model,
                    "timestamp": utc_now(),
                },
            )
            print(f"[{index}/{len(work)}] wrote {category}/{asset_id}")
        except Exception as exc:
            errors += 1
            print(f"[error] {category}/{asset_id}: {exc}", file=sys.stderr)
            append_error(
                args.error_log,
                {
                    "category": category,
                    "asset_id": asset_id,
                    "model": args.model,
                    "timestamp": utc_now(),
                    "error": str(exc),
                    "raw_response": raw,
                },
            )
        if index < len(work):
            time.sleep(args.sleep)

    print(f"done written={written} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
