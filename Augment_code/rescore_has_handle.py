#!/usr/bin/env python3
"""Re-run VLM for has_handle only; preserve existing scale from vlm_results.jsonl.

Reproduce (cd physvla_sim/Augment_code; export DASHSCOPE_API_KEY='sk-...'):

  # Re-run has_handle after prompt update; keeps longest_edge_cm from latest jsonl row
  .venv/bin/python rescore_has_handle.py --category Dishwasher --bench-only --sleep 1.0

  .venv/bin/python rescore_has_handle.py --category Door --bench-only --sleep 1.0

  # Single asset
  .venv/bin/python rescore_has_handle.py --category Dishwasher --asset-id 11710 --sleep 1.0

  # Evaluate has_handle F1 vs bench
  .venv/bin/python eval_vlm_bench.py --report bench_ours

Note: full vlm_batch.py also predicts has_handle in the combined call; use this script
when only has_handle needs refresh without re-calling scale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from experiment_paths import DEFAULT_VLM_RESULTS
from prompt.load_prompt import build_prompt_for_asset, load_vlm_base
from vlm_batch import (
    DEFAULT_BENCH_PATH,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL,
    append_jsonl,
    call_vlm,
    iter_asset_ids,
    normalize_result,
    parse_vlm_json,
    utc_now,
    vlm_image_paths,
)

AUGMENT_ROOT = Path(__file__).resolve().parent


def load_latest_results(path: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["category"], str(rec["asset_id"]))
            prev = latest.get(key)
            if prev is None or rec.get("timestamp", "") >= prev.get("timestamp", ""):
                latest[key] = rec
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run VLM has_handle only; keep scale from latest vlm_results."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_VLM_RESULTS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--vlm-base", type=Path, default=AUGMENT_ROOT / "vlm_base_template.json")
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH_PATH)
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--asset-id", default=None, help="Single asset (requires one --category)")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.asset_id and (not args.category or len(args.category) != 1):
        parser.error("--asset-id requires exactly one --category")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not args.dry_run and not api_key:
        print("DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 1

    vlm_base = load_vlm_base(args.vlm_base)
    prev = load_latest_results(args.output)

    if args.asset_id:
        work = [(args.category[0], args.asset_id)]
    else:
        work = list(
            iter_asset_ids(
                vlm_base,
                dataset_root=args.dataset_root,
                categories=args.category,
                bench_only=args.bench_only,
                bench_path=args.bench_path,
            )
        )

    pending = []
    for cat, asset_id in work:
        cat_cfg = vlm_base["categories"].get(cat, {})
        if "has_handle" not in cat_cfg.get("vlm_tasks", []):
            continue
        pending.append((cat, asset_id))

    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"planned={len(pending)} output={args.output}")
    if not pending:
        return 0

    updated = 0
    errors = 0
    for cat, asset_id in pending:
        try:
            bundle = build_prompt_for_asset(
                cat,
                asset_id,
                dataset_root=args.dataset_root,
                vlm_base=vlm_base,
                bench_path=args.bench_path,
                vlm_tasks_override=["has_handle"],
            )
        except Exception as exc:
            print(f"[error] {cat}/{asset_id} prompt: {exc}", file=sys.stderr)
            errors += 1
            continue

        images = vlm_image_paths(bundle)
        print(
            f"[{'dry' if args.dry_run else 'run'}] {cat}/{asset_id} "
            f"images={len(images)} tasks=has_handle"
        )
        if args.dry_run:
            updated += 1
            continue

        old = prev.get((cat, asset_id), {})
        old_result = old.get("result") or {}
        try:
            raw = call_vlm(
                system=bundle.system,
                user=bundle.user,
                image_paths=images,
                model=args.model,
                api_key=api_key,
            )
            parsed = normalize_result(parse_vlm_json(raw), bundle)
        except Exception as exc:
            print(f"[error] {cat}/{asset_id}: {exc}", file=sys.stderr)
            errors += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue

        result = {
            "longest_edge_cm": old_result.get("longest_edge_cm"),
            "has_handle": parsed.get("has_handle") or {},
            "door_type": old_result.get("door_type"),
            "confidence": parsed.get("confidence"),
        }
        record = {
            "category": cat,
            "asset_id": asset_id,
            "model": args.model,
            "vlm_tasks": ["has_handle"],
            "scale_ref_asset_ids": [ref.asset_id for ref in bundle.scale_refs],
            "timestamp": utc_now(),
            "status": "ok",
            "skipped_vlm": False,
            "rescore_mode": "has_handle_only",
            "scale_source": old.get("scale_source"),
            "pixel_feature": old.get("pixel_feature"),
            "pixel_image": old.get("pixel_image"),
            "pixel_estimates_cm": old.get("pixel_estimates_cm"),
            "mesh_l_norm": old.get("mesh_l_norm"),
            "mesh_estimates_cm": old.get("mesh_estimates_cm"),
            "pixel_merge_mode": old.get("pixel_merge_mode"),
            "raw_response": raw,
            "result": result,
            "error": None,
        }
        append_jsonl(args.output, record)
        prev[(cat, asset_id)] = record
        print(
            f"[ok] {cat}/{asset_id} handle={result['has_handle']} "
            f"conf={result['confidence']} scale_kept={result['longest_edge_cm']}"
        )
        updated += 1
        if args.sleep > 0 and updated + errors < len(pending):
            time.sleep(args.sleep)

    print(f"finished updated={updated} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
