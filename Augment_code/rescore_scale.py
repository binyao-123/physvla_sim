#!/usr/bin/env python3
"""Recompute scale from bench pixel refs (no VLM API) and append to vlm_results.jsonl.

Reproduce (cd physvla_sim/Augment_code; no API key needed):

  # Pixel LOO scale for bench assets only (Ours scale column, no VLM)
  .venv/bin/python rescore_scale.py --category Dishwasher --bench-only

  # Rescore all assets in a category (keeps has_handle / door_type from jsonl)
  .venv/bin/python rescore_scale.py --category Dishwasher

  .venv/bin/python eval_vlm_bench.py --report dishwasher_ours

Typically run after vlm_batch.py when scale_method=pixel in scale/calibration_config.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from prompt.load_prompt import build_prompt_for_asset, load_calibration_config, load_vlm_base
from experiment_paths import DEFAULT_VLM_RESULTS
from scale.pixel_calibrate import (
    resolve_loo_scale,
    resolve_merge_vlm_mode,
    resolve_merge_vlm_scale,
    resolve_scale_method,
)
from vlm_batch import (
    append_jsonl,
    bench_scale_estimate,
    merge_drawer_vlm_scale,
    merge_pixel_vlm_scale,
    parse_vlm_json,
)

AUGMENT_ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    parser = argparse.ArgumentParser(description="Rescore scale via pixel bench refs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_VLM_RESULTS)
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--no-loo", action="store_true")
    parser.add_argument("--model", default="pixel_ref")
    args = parser.parse_args()

    vlm_base = load_vlm_base()
    calib = load_calibration_config()
    prev = load_latest_results(args.output)

    from vlm_batch import iter_asset_ids, DEFAULT_BENCH_PATH, DEFAULT_DATASET_ROOT

    work = list(
        iter_asset_ids(
            vlm_base,
            dataset_root=DEFAULT_DATASET_ROOT,
            categories=args.category,
            bench_only=args.bench_only,
            bench_path=DEFAULT_BENCH_PATH,
        )
    )

    updated = 0
    for cat, asset_id in work:
        try:
            bundle = build_prompt_for_asset(cat, asset_id)
        except Exception as exc:
            print(f"[skip] {cat}/{asset_id}: {exc}", file=sys.stderr)
            continue

        bench = bench_scale_estimate(
            bundle,
            leave_one_out=resolve_loo_scale(cat, calib, cli_leave_one_out=not args.no_loo),
            calib=calib,
        )
        if bench is None:
            print(f"[skip] {cat}/{asset_id}: bench scale unavailable (method may be vlm_refs)", file=sys.stderr)
            continue

        old = prev.get((cat, asset_id), {})
        old_result = old.get("result") or {}
        pixel_l = float(bench["longest_edge_cm"])
        scale_l = pixel_l
        pixel_merge_mode = None
        vlm_l = None
        raw = old.get("raw_response")
        if raw:
            try:
                parsed_vlm = parse_vlm_json(raw).get("longest_edge_cm")
                if parsed_vlm is not None:
                    vlm_l = float(parsed_vlm)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if (
            vlm_l is not None
            and resolve_scale_method(cat, calib) == "pixel"
            and resolve_merge_vlm_scale(cat, calib)
        ):
            if resolve_merge_vlm_mode(cat, calib) == "drawer":
                scale_l, pixel_merge_mode = merge_drawer_vlm_scale(
                    pixel_l, float(vlm_l), bundle.scale_refs, target_asset_id=asset_id
                )
            else:
                scale_l, pixel_merge_mode = merge_pixel_vlm_scale(
                    pixel_l, float(vlm_l), bundle.scale_refs
                )
        scale_source = bench["scale_source"]
        if pixel_merge_mode and pixel_merge_mode != "pixel":
            scale_source = f"{scale_source}_{pixel_merge_mode}"
        result = {
            "longest_edge_cm": scale_l,
            "has_handle": old_result.get("has_handle") or {},
            "door_type": old_result.get("door_type"),
            "confidence": 1.0,
        }
        record = {
            "category": cat,
            "asset_id": asset_id,
            "model": args.model,
            "vlm_tasks": bundle.vlm_tasks,
            "scale_ref_asset_ids": [ref.asset_id for ref in bundle.scale_refs],
            "timestamp": utc_now(),
            "status": "ok",
            "skipped_vlm": True,
            "scale_source": scale_source,
            "pixel_feature": bench.get("pixel_feature"),
            "pixel_image": bench.get("pixel_image"),
            "pixel_estimates_cm": bench.get("pixel_estimates_cm"),
            "mesh_l_norm": bench.get("mesh_l_norm"),
            "mesh_estimates_cm": bench.get("mesh_estimates_cm"),
            "pixel_merge_mode": pixel_merge_mode,
            "raw_response": old.get("raw_response"),
            "result": result,
            "error": None,
        }
        append_jsonl(args.output, record)
        print(
            f"[ok] {cat}/{asset_id} L={result['longest_edge_cm']:.1f}cm "
            f"({bench.get('scale_source')})"
        )
        updated += 1

    print(f"updated={updated}")
    return 0 if updated else 1


if __name__ == "__main__":
    raise SystemExit(main())
