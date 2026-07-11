#!/usr/bin/env python3
"""Evaluate VLM affordance position_xyz predictions against fitr_bench.json GT.

Reproduce (cd physvla_sim/Augment_code):

  .venv/bin/python eval_affordance_vlm_bench.py --report affordance_vlm_bench
  .venv/bin/python eval_affordance_vlm_bench.py --preserved-only --report affordance_vlm_core

Primary metric: per-joint L2 error on position_xyz in calibration frame × scale_gt (cm).
has_handle / contact_axis_xyz are prompt inputs, not scored here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

AUGMENT_ROOT = Path(__file__).resolve().parent
if str(AUGMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUGMENT_ROOT))

from bench_protocol import PRESERVED_ASSET_IDS
from experiment_paths import DEFAULT_VLM_AFFORDANCE_RESULTS, eval_report_path

DEFAULT_BENCH = AUGMENT_ROOT / "fitr_bench.json"

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


def load_latest_results(
    path: Path,
    *,
    model: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if model is not None and rec.get("model") != model:
                continue
            key = (rec["category"], str(rec["asset_id"]))
            prev = latest.get(key)
            if prev is None or rec.get("timestamp", "") >= prev.get("timestamp", ""):
                latest[key] = rec
    return latest


def load_bench_gt(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    bench = json.loads(path.read_text(encoding="utf-8"))
    return {
        (category, str(asset["asset_id"])): asset
        for category, data in bench["categories"].items()
        for asset in data["assets"]
    }


def contact_error_cm(pred_xyz: list[float], gt_xyz: list[float], scale_gt: float) -> float:
    p = np.asarray(pred_xyz, dtype=np.float64)
    g = np.asarray(gt_xyz, dtype=np.float64)
    return float(np.linalg.norm(p - g) * float(scale_gt))


def _is_preserved(category: str, asset_id: str) -> bool:
    return asset_id in set(PRESERVED_ASSET_IDS.get(category, []))


def eval_vlm_affordance(
    results_path: Path,
    bench_path: Path,
    *,
    preserved_only: bool = False,
    only_ok: bool = True,
    categories: list[str] | None = None,
    model: str | None = None,
    asset_ids: list[str] | None = None,
) -> dict[str, Any]:
    preds = load_latest_results(results_path, model=model)
    gt_map = load_bench_gt(bench_path)

    errors_all: list[float] = []
    per_category: dict[str, list[float]] = defaultdict(list)
    details: list[dict[str, Any]] = []

    for (category, asset_id), asset_gt in sorted(gt_map.items()):
        if categories and category not in categories:
            continue
        if asset_ids and asset_id not in asset_ids:
            continue
        if preserved_only and not _is_preserved(category, asset_id):
            continue
        scale_gt = float(asset_gt.get("scale", {}).get("scale_gt") or 0.0)
        pred_rec = preds.get((category, asset_id))
        if pred_rec is None:
            for aff in asset_gt.get("affordances", []):
                if not _is_filled_vec(aff.get("position_xyz")):
                    continue
                details.append(
                    {
                        "category": category,
                        "asset_id": asset_id,
                        "joint_name": aff["joint_name"],
                        "status": "missing_pred",
                    }
                )
            continue
        if only_ok and pred_rec.get("status") != "ok":
            continue

        pred_aff = (pred_rec.get("result") or {}).get("affordances") or {}
        for aff in asset_gt.get("affordances", []):
            joint_name = str(aff["joint_name"])
            gt_pos = aff.get("position_xyz")
            if not _is_filled_vec(gt_pos):
                continue
            entry = pred_aff.get(joint_name)
            if not entry or not _is_filled_vec(entry.get("position_xyz")):
                details.append(
                    {
                        "category": category,
                        "asset_id": asset_id,
                        "joint_name": joint_name,
                        "status": "missing_joint_pred",
                        "gt_xyz": gt_pos,
                    }
                )
                continue
            pred_pos = entry["position_xyz"]
            err = contact_error_cm(pred_pos, gt_pos, scale_gt)
            errors_all.append(err)
            per_category[category].append(err)
            details.append(
                {
                    "category": category,
                    "asset_id": asset_id,
                    "joint_name": joint_name,
                    "status": "ok",
                    "pred_xyz": pred_pos,
                    "gt_xyz": gt_pos,
                    "error_cm": err,
                    "preserved": _is_preserved(category, asset_id),
                    "has_handle_gt": aff.get("has_handle"),
                    "contact_axis_gt": aff.get("contact_axis_xyz"),
                }
            )

    def mae(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    per_cat_summary = {
        cat: {"n": len(vals), "mae_cm": mae(vals)}
        for cat, vals in sorted(per_category.items())
    }

    return {
        "results_path": str(results_path),
        "bench_path": str(bench_path),
        "model": model,
        "preserved_only": preserved_only,
        "n_errors": len(errors_all),
        "mae_cm": mae(errors_all),
        "per_category": per_cat_summary,
        "details": details,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"results: {report['results_path']}")
    print(f"bench:   {report['bench_path']}")
    if report.get("model"):
        print(f"model:   {report['model']}")
    print(f"affordances evaluated: {report['n_errors']}")
    if report["mae_cm"] is not None:
        print(f"position_xyz MAE: {report['mae_cm']:.2f} cm")
    print()
    print(f"{'Category':<18} {'中文':<12} {'n':>4} {'MAE(cm)':>8}")
    print("-" * 46)
    cat_maes: list[float] = []
    for category, row in report["per_category"].items():
        mae = row.get("mae_cm")
        if mae is not None:
            cat_maes.append(mae)
        zh = CATEGORY_ZH.get(category, "")
        mae_txt = f"{mae:.2f}" if mae is not None else "—"
        print(f"{category:<18} {zh:<12} {row.get('n', 0):>4} {mae_txt:>8}")
    if cat_maes:
        print(f"{'Average':<18} {'平均':<12} {'':>4} {sum(cat_maes)/len(cat_maes):>8.2f}")


def print_details(report: dict[str, Any]) -> None:
    rows = [d for d in report["details"] if d.get("status") == "ok"]
    if not rows:
        return
    print()
    print("Details:")
    print(f"{'Category':<18} {'Asset':<8} {'Joint':<10} {'Pred xyz':<28} {'GT xyz':<28} {'Err(cm)':>8}")
    print("-" * 104)
    for row in rows:
        pred = "[" + ", ".join(f"{v:.3g}" for v in row["pred_xyz"]) + "]"
        gt = "[" + ", ".join(f"{v:.3g}" for v in row["gt_xyz"]) + "]"
        print(
            f"{row['category']:<18} {row['asset_id']:<8} {row['joint_name']:<10} "
            f"{pred:<28} {gt:<28} {row['error_cm']:>8.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval VLM affordance position_xyz vs bench GT.")
    parser.add_argument("--results", type=Path, default=DEFAULT_VLM_AFFORDANCE_RESULTS)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--preserved-only", action="store_true")
    parser.add_argument("--category", action="append", default=None, help="Filter by category")
    parser.add_argument("--asset-id", action="append", default=None, help="Filter by asset id")
    parser.add_argument("--model", default=None, help="Only evaluate predictions from this model")
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--details", action="store_true", help="Print per-joint pred vs GT rows")
    parser.add_argument("--report", type=str, default="affordance_vlm_bench")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.results.is_file():
        print(f"Results not found: {args.results}", file=sys.stderr)
        return 1
    if not args.bench.is_file():
        print(f"Bench not found: {args.bench}", file=sys.stderr)
        return 1

    report = eval_vlm_affordance(
        args.results,
        args.bench,
        preserved_only=args.preserved_only,
        only_ok=not args.include_failed,
        categories=args.category,
        model=args.model,
        asset_ids=[str(x) for x in args.asset_id] if args.asset_id else None,
    )
    print_report(report)
    if args.details:
        print_details(report)

    out = args.json_out or eval_report_path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
