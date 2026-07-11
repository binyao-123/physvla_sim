#!/usr/bin/env python3
"""Evaluate geometry-derived contact points (Phi_aff) against fitr_bench.json GT.

Reproduce (cd physvla_sim/Augment_code; no API key):

  # Full bench — contact error (cm) for all algo-filled affordances
  .venv/bin/python eval_affordance_bench.py --report affordance_ours

  # Core preserved assets only (paper Table 5.2 subset, 57 affordances)
  .venv/bin/python eval_affordance_bench.py --preserved-only --report affordance_ours_core

Reports: experiments/eval/affordance_ours.json (MAE cm per category vs bench position_xyz GT).
VLM affordance column is not implemented here; use eval_vlm_bench.py for has_handle F1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AUGMENT_ROOT = Path(__file__).resolve().parent
if str(AUGMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUGMENT_ROOT))

from affordance.contact_point import compute_contact_point
from bench_protocol import PRESERVED_ASSET_IDS
from experiment_paths import eval_report_path
from scale.l_norm import asset_dir_for

DEFAULT_BENCH = AUGMENT_ROOT / "fitr_bench.json"
DEFAULT_DATASET = AUGMENT_ROOT.parent / "datasets" / "data_normalized"

CATEGORY_ZH: dict[str, str] = {
    "Laptop": "笔记本",
    "Display": "显示器",
    "Microwave": "微波炉",
    "Drawer": "抽屉(table)",
    "Lamp": "台灯",
    "Faucet": "水龙头",
    "Knife": "小刀",
    "Dishwasher": "洗碗机",
    "Door": "门",
    "Refrigerator": "冰箱",
    "Scissors": "剪刀",
    "StorageFurniture": "储物柜",
}


def _is_filled_vec(vec: list[float | None] | None) -> bool:
    return bool(vec) and all(value is not None for value in vec)


def load_bench_gt(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    bench = json.loads(path.read_text(encoding="utf-8"))
    gt: dict[tuple[str, str], dict[str, Any]] = {}
    for category, data in bench["categories"].items():
        for asset in data["assets"]:
            gt[(category, str(asset["asset_id"]))] = asset
    return gt


def contact_error_cm(
    pred_xyz: list[float],
    gt_xyz: list[float],
    scale_gt: float,
) -> float:
    """L2 distance in calibration frame, converted to centimeters via scale_gt."""
    p = np.asarray(pred_xyz, dtype=np.float64)
    g = np.asarray(gt_xyz, dtype=np.float64)
    return float(np.linalg.norm(p - g) * float(scale_gt))


def _is_preserved(category: str, asset_id: str) -> bool:
    return asset_id in set(PRESERVED_ASSET_IDS.get(category, []))


def eval_affordance_bench(
    bench_path: Path,
    dataset_root: Path,
    *,
    preserved_only: bool = False,
) -> dict[str, Any]:
    gt_map = load_bench_gt(bench_path)

    errors_all: list[float] = []
    errors_preserved: list[float] = []
    per_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "errors": [],
            "errors_preserved": [],
            "n": 0,
            "n_preserved": 0,
            "errors_compute": [],
            "skipped_null_gt": 0,
            "skipped_filter": 0,
        }
    )
    detail_rows: list[dict[str, Any]] = []

    for (category, asset_id), gt_asset in sorted(gt_map.items()):
        if preserved_only and not _is_preserved(category, asset_id):
            per_category[category]["skipped_filter"] += 1
            continue

        asset_dir = asset_dir_for(category, asset_id, dataset_root)
        scale_gt = gt_asset.get("scale", {}).get("scale_gt")
        if scale_gt is None:
            per_category[category]["errors_compute"].append(f"{asset_id}: missing scale_gt")
            continue

        preserved = _is_preserved(category, asset_id)

        for aff in gt_asset.get("affordances", []):
            joint_name = str(aff["joint_name"])
            gt_pos = aff.get("position_xyz")
            if not _is_filled_vec(gt_pos):
                per_category[category]["skipped_null_gt"] += 1
                continue

            try:
                result = compute_contact_point(
                    asset_dir,
                    joint_name=joint_name,
                    link_name=str(aff.get("link_name", "")),
                    motion_type=str(aff.get("motion_type", "revolute")),
                    category=category,
                )
            except Exception as exc:
                per_category[category]["errors_compute"].append(f"{asset_id}/{joint_name}: {exc}")
                detail_rows.append(
                    {
                        "category": category,
                        "asset_id": asset_id,
                        "joint_name": joint_name,
                        "status": "error",
                        "error": str(exc),
                        "preserved": preserved,
                    }
                )
                continue

            err_cm = contact_error_cm(result.position_xyz, gt_pos, float(scale_gt))
            errors_all.append(err_cm)
            per_category[category]["errors"].append(err_cm)
            per_category[category]["n"] += 1

            if preserved:
                errors_preserved.append(err_cm)
                per_category[category]["errors_preserved"].append(err_cm)
                per_category[category]["n_preserved"] += 1

            detail_rows.append(
                {
                    "category": category,
                    "asset_id": asset_id,
                    "joint_name": joint_name,
                    "status": "ok",
                    "preserved": preserved,
                    "scale_gt": float(scale_gt),
                    "pred_xyz": result.position_xyz,
                    "gt_xyz": [float(v) for v in gt_pos],
                    "error_cm": err_cm,
                    "source": result.source,
                }
            )

    def _summary(errs: list[float]) -> dict[str, Any]:
        if not errs:
            return {"n": 0}
        n = len(errs)
        return {
            "n": n,
            "mae_cm": sum(errs) / n,
            "rmse_cm": math.sqrt(sum(e * e for e in errs) / n),
            "max_cm": max(errs),
            "median_cm": float(np.median(errs)),
        }

    by_category: dict[str, Any] = {}
    for category in sorted(per_category):
        cat = per_category[category]
        by_category[category] = {
            "contact": _summary(cat["errors"]),
            "contact_preserved": _summary(cat["errors_preserved"]),
            "n_affordances": cat["n"],
            "n_preserved_affordances": cat["n_preserved"],
            "skipped_null_gt": cat["skipped_null_gt"],
            "skipped_filter": cat["skipped_filter"],
            "errors_compute": cat["errors_compute"],
        }

    return {
        "bench_path": str(bench_path),
        "dataset_root": str(dataset_root),
        "method": "geometry_contact_point",
        "metric": "L2 position error (cm) = ||pred-gt||_2 * scale_gt",
        "preserved_only": preserved_only,
        "contact": _summary(errors_all),
        "contact_preserved_subset": _summary(errors_preserved),
        "by_category": by_category,
        "details": detail_rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"bench: {report['bench_path']}")
    print(f"method: {report['method']}")
    print(f"metric: {report['metric']}")
    if report["preserved_only"]:
        print("scope: preserved core bench assets only")
    print()

    overall = report["contact"]
    if overall.get("n"):
        print("=== Contact point error (all evaluated affordances) ===")
        print(f"  n        = {overall['n']}")
        print(f"  MAE      = {overall['mae_cm']:.2f} cm")
        print(f"  RMSE     = {overall['rmse_cm']:.2f} cm")
        print(f"  median   = {overall['median_cm']:.2f} cm")
        print(f"  max      = {overall['max_cm']:.2f} cm")
        print()

    preserved = report["contact_preserved_subset"]
    if preserved.get("n"):
        print("=== Contact point error (preserved core assets only) ===")
        print(f"  n        = {preserved['n']}")
        print(f"  MAE      = {preserved['mae_cm']:.2f} cm")
        print()

    print("=== Table 5.2 style — Affordance Ours (cm, ↓) ===")
    print(f"{'Category':<18} {'ZH':<12} {'n':>4} {'MAE cm':>8} {'preserved MAE':>14}")
    cat_maes: list[float] = []
    for category, stats in report["by_category"].items():
        c = stats["contact"]
        cp = stats["contact_preserved"]
        mae = c.get("mae_cm")
        mae_p = cp.get("mae_cm")
        if mae is not None:
            cat_maes.append(mae)
        mae_txt = f"{mae:.2f}" if mae is not None else "—"
        mae_p_txt = f"{mae_p:.2f}" if mae_p is not None else "—"
        zh = CATEGORY_ZH.get(category, "")
        print(
            f"{category:<18} {zh:<12} {c.get('n', 0):>4} {mae_txt:>8} {mae_p_txt:>14}"
        )
    if cat_maes:
        print(f"{'Average':<18} {'平均':<12} {'':>4} {sum(cat_maes)/len(cat_maes):>8.2f}")
    print()

    print("=== Largest errors (top 10) ===")
    ok_rows = [r for r in report["details"] if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: r["error_cm"], reverse=True)
    for row in ok_rows[:10]:
        mark = "core" if row["preserved"] else "ext"
        print(
            f"  [{mark}] {row['category']}/{row['asset_id']}/{row['joint_name']}: "
            f"{row['error_cm']:.2f} cm  pred={row['pred_xyz']} gt={row['gt_xyz']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate geometry contact-point derivation on fitr_bench GT."
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--preserved-only",
        action="store_true",
        help="Evaluate only preserved core bench assets",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--report",
        type=str,
        default="affordance_ours",
        help="Write JSON to experiments/eval/NAME.json",
    )
    args = parser.parse_args()

    if not args.bench.is_file():
        print(f"Bench not found: {args.bench}", file=sys.stderr)
        return 1

    report = eval_affordance_bench(
        args.bench,
        args.dataset_root,
        preserved_only=args.preserved_only,
    )
    print_report(report)

    json_out = args.json_out or eval_report_path(args.report)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
