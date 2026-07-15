#!/usr/bin/env python3
"""Predict FITR-Bench joint dynamics with Qwen VLM.

Predictions directly update each asset's ``joints`` dynamics fields. Existing
``dynamics_vlm_candidate`` fields can be promoted once and are then removed.

Reproduce (cd physvla_sim/Augment_code; export DASHSCOPE_API_KEY='sk-...'):

  # Preview the selected work without calling Qwen or editing the benchmark.
  .venv/bin/python dyn/predict_joint_dynamics.py --category Laptop --asset-id 9604 --dry-run

  # Predict one asset and directly update joints' three dynamics fields.
  .venv/bin/python dyn/predict_joint_dynamics.py --category Laptop --asset-id 9604

  # Predict all FITR-Bench assets and directly update their joint dynamics.
  .venv/bin/python dyn/predict_joint_dynamics.py --sleep 1.0

  # Promote prior candidate results without another VLM call.
  .venv/bin/python dyn/predict_joint_dynamics.py --category Laptop --promote-existing-candidates
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUGMENT_ROOT = Path(__file__).resolve().parents[1]
if str(AUGMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUGMENT_ROOT))

from prompt.load_prompt import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_VLM_BASE_PATH,
    find_category_scale_image,
    list_movable_joints,
    load_vlm_base,
)


DEFAULT_BENCH = AUGMENT_ROOT / "fitr_bench.json"
DEFAULT_PROMPT = AUGMENT_ROOT / "prompt" / "joint_dynamics_vlm_prompt.txt"
DEFAULT_ERROR_LOG = AUGMENT_ROOT / "experiments" / "joint_dynamics_errors.jsonl"
DEFAULT_MODEL = "qwen3-vl-flash"

RANGES: dict[str, dict[str, tuple[float, float] | None]] = {
    "Laptop": {"damping": (80, 150), "stiffness": None, "effort_limit": (0.5, 1.2)},
    "Display": {"damping": (1000, 2000), "stiffness": None, "effort_limit": (1.2, 4.8)},
    "Microwave": {"damping": (0.3, 2), "stiffness": None, "effort_limit": (0.005, 0.04)},
    "Drawer": {"damping": (10, 90), "stiffness": None, "effort_limit": (0.3, 3)},
    "Lamp": {"damping": (40, 300), "stiffness": None, "effort_limit": (0.5, 4)},
    "Faucet": {"damping": (20, 120), "stiffness": None, "effort_limit": (0.015, 0.12)},
    "Knife": {"damping": (3, 30), "stiffness": (25, 100), "effort_limit": (0.15, 0.75)},
    "Dishwasher": {"damping": (200, 1200), "stiffness": None, "effort_limit": (3.75, 15)},
    "Door": {"damping": (30, 150), "stiffness": (50, 150), "effort_limit": (0.9, 3.6)},
    "Refrigerator": {"damping": (20, 150), "stiffness": None, "effort_limit": (0.3, 1.35)},
    "Scissors": {"damping": (2, 15), "stiffness": None, "effort_limit": (0.02, 0.15)},
    "StorageFurniture": {"damping": (3, 30), "stiffness": None, "effort_limit": (1, 8)},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_vlm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair frequent Qwen formatting errors: unquoted joint keys, omitted
        # commas between object fields, and a trailing comma before } or ].
        repaired = re.sub(
            r'([{\[,]\s*)(joint[\w]*)\s*:',
            r'\1"\2":',
            text,
        )
        # Qwen occasionally closes a joint object with `]` and places a
        # per-joint confidence after it. Retain the joint values and remove
        # that misplaced confidence field before standard JSON repair.
        repaired = re.sub(
            r'(\{[^{}]*"joint_name"[^{}]*?)\]\s*,\s*"confidence"\s*:\s*'
            r'[^,}\]]+\s*\}',
            r"\1}",
            repaired,
        )
        repaired = re.sub(
            r',\s*"confidence"\s*:\s*[^,\]]+(?=\s*\])',
            "",
            repaired,
        )
        repaired = re.sub(
            r'("(?:[^"\\]|\\.)*"|\b(?:true|false|null|-?\d+(?:\.\d+)?)\b|[}\]])'
            r'\s+("(?:[^"\\]|\\.)*"\s*:)',
            r"\1, \2",
            repaired,
        )
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return json.loads(repaired)


def append_error(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def image_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{encoded}"


def call_vlm(
    *,
    system: str,
    user: str,
    image_paths: list[Path],
    model: str,
    api_key: str,
) -> str:
    import dashscope
    from dashscope import MultiModalConversation

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    content = [{"image": image_to_data_uri(path)} for path in image_paths]
    content.append({"text": user})
    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": [{"text": system}]},
            {"role": "user", "content": content},
        ],
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"VLM call failed: status={response.status_code}, "
            f"code={getattr(response, 'code', None)}, "
            f"message={getattr(response, 'message', None)}"
        )
    message = response.output.choices[0].message.content
    if isinstance(message, list):
        for part in message:
            if isinstance(part, dict) and "text" in part:
                return str(part["text"])
        raise RuntimeError(f"Unexpected VLM content: {message!r}")
    return str(message)


def load_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    try:
        system = text.split("[SYSTEM]\n", 1)[1].split("\n[USER]\n", 1)[0].strip()
        user = text.split("\n[USER]\n", 1)[1].strip()
    except IndexError as exc:
        raise ValueError(f"Invalid prompt format: {path}") from exc
    return system, user


def render_prompt(
    user: str,
    category: str,
    asset_id: str,
    joints: list[Any],
    *,
    reference_size_cm: float,
    target_size_cm: float,
) -> str:
    joint_lines = "\n".join(
        f"- joint_name: {joint.joint_name}; link_name: {joint.link_name}; "
        f"motion_type: {joint.motion_type}"
        for joint in joints
    )
    size_ratio = target_size_cm / reference_size_cm
    return (
        user.replace("{category}", category)
        .replace("{asset_id}", asset_id)
        .replace("{joint_lines}", joint_lines)
        .replace("{reference_size_cm}", f"{reference_size_cm:.2f}")
        .replace("{target_size_cm}", f"{target_size_cm:.2f}")
        .replace("{size_ratio}", f"{size_ratio:.3f}")
    )


def validate_prediction(
    category: str,
    joints: list[Any],
    parsed: dict[str, Any],
    *,
    clamp_to_range: bool = False,
) -> dict[str, Any]:
    predicted = parsed.get("joints")
    if not isinstance(predicted, list):
        raise ValueError("Response field 'joints' must be an array.")

    by_name: dict[str, dict[str, Any]] = {}
    for item in predicted:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("joint_name", "")).strip()
        joint_name = raw_name.split(" / ", 1)[0].strip()
        if joint_name in by_name:
            raise ValueError(f"Duplicate prediction for joint '{joint_name}'.")
        by_name[joint_name] = item
    expected_names = [joint.joint_name for joint in joints]
    if set(by_name) != set(expected_names):
        raise ValueError(
            f"Response joint names {sorted(by_name)} do not match expected "
            f"{sorted(expected_names)}."
        )

    ranges = RANGES[category]
    result: list[dict[str, Any]] = []
    for joint in joints:
        item = by_name[joint.joint_name]
        values = {"joint_name": joint.joint_name}
        for field in ("damping", "stiffness", "effort_limit"):
            try:
                value = float(item[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{joint.joint_name}.{field} must be numeric.") from exc
            if value < 0:
                raise ValueError(f"{joint.joint_name}.{field} must be nonnegative.")
            allowed = ranges[field]
            if field == "stiffness" and allowed is None:
                if value != 0:
                    raise ValueError(f"{category} stiffness must be exactly 0.")
            elif field == "stiffness":
                if value != 0 and not (allowed[0] <= value <= allowed[1]):
                    raise ValueError(
                        f"{category} stiffness must be 0 or within {allowed}."
                    )
            elif allowed is not None and not (allowed[0] <= value <= allowed[1]):
                if clamp_to_range:
                    value = min(max(value, allowed[0]), allowed[1])
                else:
                    raise ValueError(
                        f"{joint.joint_name}.{field}={value} is outside {allowed}."
                    )
            values[field] = value
        confidence = item.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not 0 <= confidence <= 1:
                raise ValueError("Per-joint confidence must be in [0, 1].")
        values["confidence"] = confidence
        result.append(values)

    confidence = float(parsed.get("confidence"))
    if not 0 <= confidence <= 1:
        raise ValueError("Overall confidence must be in [0, 1].")
    return {"joints": result, "confidence": confidence}


def selected_assets(
    bench: dict[str, Any], categories: list[str] | None, asset_id: str | None
) -> list[tuple[str, dict[str, Any]]]:
    selected_categories = categories or list(bench["categories"])
    work: list[tuple[str, dict[str, Any]]] = []
    for category in selected_categories:
        if category not in bench["categories"]:
            raise KeyError(f"Unknown benchmark category: {category}")
        for asset in bench["categories"][category]["assets"]:
            if asset_id is None or str(asset["asset_id"]) == asset_id:
                work.append((category, asset))
    if asset_id is not None and not work:
        raise KeyError(f"Asset {asset_id} not found in selected category.")
    return work


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict FITR-Bench damping/stiffness/effort_limit with Qwen VLM."
    )
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--vlm-base", type=Path, default=DEFAULT_VLM_BASE_PATH)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--asset-id", default=None, help="Requires exactly one --category.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based offset in the selected work list; useful for resuming.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERROR_LOG,
        help="Append failed assets and raw VLM responses to this JSONL file.",
    )
    parser.add_argument(
        "--promote-existing-candidates",
        action="store_true",
        help="Write stored candidate results to joints, then delete candidate fields.",
    )
    args = parser.parse_args()

    if args.asset_id and len(args.category or []) != 1:
        parser.error("--asset-id requires exactly one --category.")
    if args.promote_existing_candidates and args.dry_run:
        parser.error("--promote-existing-candidates cannot be combined with --dry-run.")
    if args.start_index < 0:
        parser.error("--start-index must be nonnegative.")

    bench = json.loads(args.bench_path.read_text(encoding="utf-8"))
    vlm_base = load_vlm_base(args.vlm_base)
    system, user_template = load_prompt(args.prompt)
    work = selected_assets(bench, args.category, args.asset_id)
    work = work[args.start_index:]
    if args.limit is not None:
        work = work[: args.limit]

    if args.dry_run:
        for category, asset in work:
            category_dir = vlm_base["categories"][category]["category_dir"]
            ref_id = str(bench["categories"][category]["assets"][0]["asset_id"])
            print(f"{category}/{asset['asset_id']}  reference={ref_id}  dir={category_dir}")
        return 0

    if not args.promote_existing_candidates:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("DASHSCOPE_API_KEY is not set.", file=sys.stderr)
            return 1
    else:
        api_key = ""

    for index, (category, asset) in enumerate(work, start=1):
        if args.promote_existing_candidates:
            candidate = asset.get("dynamics_vlm_candidate", {})
            result = candidate.get("result", {})
            predicted_joints = result.get("joints")
            if not isinstance(predicted_joints, list):
                raise ValueError(
                    f"No valid dynamics_vlm_candidate for {category}/{asset['asset_id']}."
                )
            by_name = {joint["joint_name"]: joint for joint in predicted_joints}
            for bench_joint in asset["joints"]:
                bench_joint.update(
                    {
                        key: by_name[bench_joint["joint_name"]][key]
                        for key in ("damping", "stiffness", "effort_limit")
                    }
                )
            asset.pop("dynamics_vlm_candidate", None)
            args.bench_path.write_text(
                json.dumps(bench, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[{index}/{len(work)}] promoted {category}/{asset['asset_id']}")
            continue

        category_dir = vlm_base["categories"][category]["category_dir"]
        target_dir = args.dataset_root / category_dir / str(asset["asset_id"])
        reference = bench["categories"][category]["assets"][0]
        reference_dir = args.dataset_root / category_dir / str(reference["asset_id"])
        joints = list_movable_joints(target_dir)
        if not joints:
            message = f"No movable joints found for {category}/{asset['asset_id']}."
            print(f"[error] {message}", file=sys.stderr)
            append_error(
                args.error_log,
                {"category": category, "asset_id": asset["asset_id"], "error": message},
            )
            continue
        try:
            reference_size_cm = float(reference["scale"]["L_real_cm_gt"])
            target_size_cm = float(asset["scale"]["L_real_cm_gt"])
        except (KeyError, TypeError, ValueError) as exc:
            message = (
                f"Missing recovered longest-edge size for {category}/{asset['asset_id']}: {exc}"
            )
            print(f"[error] {message}", file=sys.stderr)
            append_error(
                args.error_log,
                {"category": category, "asset_id": asset["asset_id"], "error": message},
            )
            continue

        raw: str | None = None
        try:
            raw = call_vlm(
                system=system,
                user=render_prompt(
                    user_template,
                    category,
                    str(asset["asset_id"]),
                    joints,
                    reference_size_cm=reference_size_cm,
                    target_size_cm=target_size_cm,
                ),
                image_paths=[
                    find_category_scale_image(reference_dir, category),
                    find_category_scale_image(target_dir, category),
                ],
                model=args.model,
                api_key=api_key,
            )
            result = validate_prediction(category, joints, parse_vlm_json(raw))
            by_name = {joint["joint_name"]: joint for joint in result["joints"]}
            for bench_joint in asset["joints"]:
                bench_joint.update(
                    {
                        key: by_name[bench_joint["joint_name"]][key]
                        for key in ("damping", "stiffness", "effort_limit")
                    }
                )
            asset.pop("dynamics_vlm_candidate", None)
            args.bench_path.write_text(
                json.dumps(bench, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[{index}/{len(work)}] updated {category}/{asset['asset_id']}")
        except Exception as exc:
            print(f"[error] {category}/{asset['asset_id']}: {exc}", file=sys.stderr)
            append_error(
                args.error_log,
                {
                    "category": category,
                    "asset_id": asset["asset_id"],
                    "error": str(exc),
                    "raw_response": raw,
                },
            )
        if index < len(work):
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
