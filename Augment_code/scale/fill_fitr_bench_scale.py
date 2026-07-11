#!/usr/bin/env python3
"""Fill fitr_bench.json L_norm_gt and scale_gt using FITR scale formula."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from l_norm import asset_dir_for, compute_l_norm, compute_scale

AUGMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = AUGMENT_ROOT / "fitr_bench.json"
DEFAULT_DATASET = AUGMENT_ROOT.parent / "datasets" / "data_normalized"


def fill_bench(
    bench_path: Path,
    dataset_root: Path,
    *,
    recompute_l_norm: bool = False,
    dry_run: bool = False,
) -> dict:
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    stats = {
        "l_norm_filled": 0,
        "scale_gt_filled": 0,
        "skipped_no_l_real": 0,
        "errors": [],
    }

    for category, cat_data in bench["categories"].items():
        for asset in cat_data["assets"]:
            aid = asset["asset_id"]
            scale = asset.setdefault("scale", {})
            asset_dir = asset_dir_for(category, aid, dataset_root)

            need_l_norm = scale.get("L_norm_gt") is None or recompute_l_norm
            if need_l_norm:
                try:
                    result = compute_l_norm(asset_dir)
                    scale["L_norm_gt"] = round(result["L_norm"], 6)
                    stats["l_norm_filled"] += 1
                    print(
                        f"[L_norm] {category}/{aid} = {scale['L_norm_gt']:.6f} "
                        f"({result['mesh_source']})"
                    )
                except Exception as exc:
                    stats["errors"].append(f"{category}/{aid} L_norm: {exc}")
                    print(f"[ERROR] {category}/{aid} L_norm: {exc}", file=sys.stderr)

            l_real = scale.get("L_real_cm_gt")
            l_norm = scale.get("L_norm_gt")
            if l_real is None:
                stats["skipped_no_l_real"] += 1
                continue
            if l_norm is None:
                continue

            scale["scale_gt"] = round(compute_scale(float(l_real), float(l_norm)), 6)
            stats["scale_gt_filled"] += 1

    if not dry_run:
        bench_path.write_text(
            json.dumps(bench, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill fitr_bench.json L_norm_gt and scale_gt (s = L_real / L_norm)."
    )
    parser.add_argument(
        "--bench",
        type=Path,
        default=DEFAULT_BENCH,
        help="Path to fitr_bench.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to datasets/data_normalized",
    )
    parser.add_argument(
        "--recompute-l-norm",
        action="store_true",
        help="Recompute L_norm_gt even when already set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print updates without writing fitr_bench.json",
    )
    args = parser.parse_args()

    stats = fill_bench(
        args.bench,
        args.dataset_root,
        recompute_l_norm=args.recompute_l_norm,
        dry_run=args.dry_run,
    )

    print("\n--- summary ---")
    print(f"L_norm_gt filled:  {stats['l_norm_filled']}")
    print(f"scale_gt filled:   {stats['scale_gt_filled']}")
    print(f"skipped (no L_real): {stats['skipped_no_l_real']}")
    if stats["errors"]:
        print(f"errors: {len(stats['errors'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
