#!/usr/bin/env python3
"""Evaluate vlm_results.jsonl against fitr_bench.json ground truth.

Reproduce (cd physvla_sim/Augment_code):

  # Ours — scale sMAPE, has_handle F1, door_type accuracy vs bench GT
  .venv/bin/python eval_vlm_bench.py --report bench_ours

  # Baseline (pure VLM scale) vs bench
  .venv/bin/python eval_vlm_bench.py --results experiments/vlm_results_baseline.jsonl --report bench_baseline

  # Per-category (example: Dishwasher 3 bench assets)
  .venv/bin/python eval_vlm_bench.py --results experiments/vlm_results.jsonl --report dishwasher_ours
  .venv/bin/python eval_vlm_bench.py --results experiments/vlm_results_baseline.jsonl --report dishwasher_baseline

Reports written to experiments/eval/<NAME>.json; metrics printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiment_paths import (
    DEFAULT_VLM_RESULTS,
    DEFAULT_VLM_RESULTS_BASELINE,
    eval_report_path,
)

AUGMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = DEFAULT_VLM_RESULTS
DEFAULT_BASELINE_RESULTS = DEFAULT_VLM_RESULTS_BASELINE
DEFAULT_BENCH = AUGMENT_ROOT / "fitr_bench.json"


def load_latest_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
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


def load_bench_gt(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    bench = json.loads(path.read_text(encoding="utf-8"))
    gt: dict[tuple[str, str], dict[str, Any]] = {}
    for category, data in bench["categories"].items():
        for asset in data["assets"]:
            gt[(category, str(asset["asset_id"]))] = asset
    return gt


def pct_err(pred: float, target: float) -> float:
    if target == 0:
        return math.inf if pred != 0 else 0.0
    return abs(pred - target) / target * 100.0


def smape(pred: float, target: float) -> float:
    """FITR B.4 symmetric MAPE (%). L_hat=pred, L*=target."""
    denom = (abs(pred) + abs(target)) / 2.0
    if denom == 0:
        return 0.0 if pred == target else math.inf
    return abs(pred - target) / denom * 100.0


def binary_f1(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Binary F1 over joint-level has_handle rows."""
    tp = fp = fn = tn = 0
    for row in rows:
        pred = bool(row["pred"])
        gt = bool(row["gt"])
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None and recall is None:
        f1 = None
    elif precision is None or recall is None:
        f1 = 0.0
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def eval_bench(
    results_path: Path,
    bench_path: Path,
    *,
    only_ok: bool = True,
) -> dict[str, Any]:
    results = load_latest_results(results_path)
    gt_map = load_bench_gt(bench_path)

    scale_abs_err: list[float] = []
    scale_pct_err: list[float] = []
    scale_smape: list[float] = []
    scale_scale_abs_err: list[float] = []

    handle_total = 0
    handle_correct = 0
    handle_rows: list[dict[str, Any]] = []

    door_total = 0
    door_correct = 0
    door_rows: list[dict[str, Any]] = []

    per_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scale_abs_err": [],
            "scale_pct_err": [],
            "scale_smape": [],
            "handle_total": 0,
            "handle_correct": 0,
            "handle_rows": [],
            "door_total": 0,
            "door_correct": 0,
            "missing": [],
            "failed": [],
        }
    )

    detail_rows: list[dict[str, Any]] = []

    for (category, asset_id), gt_asset in sorted(gt_map.items()):
        cat = per_category[category]
        rec = results.get((category, asset_id))
        if rec is None:
            cat["missing"].append(asset_id)
            continue
        if only_ok and rec.get("status") != "ok":
            cat["failed"].append(asset_id)
            continue
        if rec.get("status") != "ok" or not rec.get("result"):
            cat["failed"].append(asset_id)
            continue

        pred = rec["result"]
        row: dict[str, Any] = {
            "category": category,
            "asset_id": asset_id,
            "L_pred_cm": pred.get("longest_edge_cm"),
            "L_gt_cm": gt_asset["scale"]["L_real_cm_gt"],
            "L_norm_gt": gt_asset["scale"]["L_norm_gt"],
            "scale_gt": gt_asset["scale"]["scale_gt"],
            "confidence": pred.get("confidence"),
        }

        l_pred = pred.get("longest_edge_cm")
        l_gt = gt_asset["scale"]["L_real_cm_gt"]
        if l_pred is not None and l_gt is not None:
            l_pred_f = float(l_pred)
            l_gt_f = float(l_gt)
            abs_e = abs(l_pred_f - l_gt_f)
            pct_e = pct_err(l_pred_f, l_gt_f)
            smape_e = smape(l_pred_f, l_gt_f)
            scale_abs_err.append(abs_e)
            scale_pct_err.append(pct_e)
            scale_smape.append(smape_e)
            cat["scale_abs_err"].append(abs_e)
            cat["scale_pct_err"].append(pct_e)
            cat["scale_smape"].append(smape_e)
            row["L_abs_err_cm"] = abs_e
            row["L_pct_err"] = pct_e
            row["L_smape_pct"] = smape_e

            l_norm = gt_asset["scale"]["L_norm_gt"]
            scale_gt = gt_asset["scale"]["scale_gt"]
            if l_norm and scale_gt:
                scale_pred = l_pred_f / float(l_norm)
                scale_e = abs(scale_pred - float(scale_gt))
                scale_scale_abs_err.append(scale_e)
                row["scale_pred"] = scale_pred
                row["scale_abs_err"] = scale_e

        pred_handles = pred.get("has_handle") or {}
        for aff in gt_asset.get("affordances", []):
            jn = aff["joint_name"]
            gt_handle = aff.get("has_handle")
            if gt_handle is None:
                continue
            pred_handle = bool(pred_handles.get(jn, False))
            ok = pred_handle == bool(gt_handle)
            handle_total += 1
            cat["handle_total"] += 1
            if ok:
                handle_correct += 1
                cat["handle_correct"] += 1
            handle_rows.append(
                {
                    "category": category,
                    "asset_id": asset_id,
                    "joint_name": jn,
                    "pred": pred_handle,
                    "gt": gt_handle,
                    "ok": ok,
                }
            )
            cat["handle_rows"].append(handle_rows[-1])
            row[f"has_handle_{jn}_pred"] = pred_handle
            row[f"has_handle_{jn}_gt"] = gt_handle

        if category == "Door":
            gt_door = None
            for aff in gt_asset.get("affordances", []):
                if aff.get("door_type") is not None:
                    gt_door = aff["door_type"]
                    break
            if gt_door is None:
                for j in gt_asset.get("joints", []):
                    if j.get("door_type") is not None:
                        gt_door = j["door_type"]
                        break
            pred_door = pred.get("door_type")
            if gt_door is not None:
                door_total += 1
                cat["door_total"] += 1
                ok = pred_door == gt_door
                if ok:
                    door_correct += 1
                    cat["door_correct"] += 1
                door_rows.append(
                    {
                        "category": category,
                        "asset_id": asset_id,
                        "pred": pred_door,
                        "gt": gt_door,
                        "ok": ok,
                    }
                )
                row["door_type_pred"] = pred_door
                row["door_type_gt"] = gt_door

        detail_rows.append(row)

    def scale_summary(
        abs_errs: list[float],
        pct_errs: list[float],
        smape_errs: list[float],
    ) -> dict[str, Any]:
        if not abs_errs:
            return {"n": 0}
        n = len(abs_errs)
        return {
            "n": n,
            "mae_cm": sum(abs_errs) / n,
            "mape_pct": sum(pct_errs) / n,
            "smape_pct": sum(smape_errs) / n,
            "rmse_cm": math.sqrt(sum(e * e for e in abs_errs) / n),
            "max_abs_cm": max(abs_errs),
            "max_pct": max(pct_errs),
            "max_smape_pct": max(smape_errs),
        }

    handle_f1 = binary_f1(handle_rows)

    by_category = {}
    for category, cat in sorted(per_category.items()):
        cat_handle_f1 = binary_f1(cat["handle_rows"])
        by_category[category] = {
            "scale": scale_summary(cat["scale_abs_err"], cat["scale_pct_err"], cat["scale_smape"]),
            "has_handle_acc": (
                cat["handle_correct"] / cat["handle_total"] if cat["handle_total"] else None
            ),
            "has_handle_f1": cat_handle_f1["f1"],
            "has_handle_n": cat["handle_total"],
            "door_type_acc": (
                cat["door_correct"] / cat["door_total"] if cat["door_total"] else None
            ),
            "door_type_n": cat["door_total"],
            "missing": cat["missing"],
            "failed": cat["failed"],
        }

    return {
        "results_path": str(results_path),
        "bench_path": str(bench_path),
        "bench_assets": len(gt_map),
        "evaluated_ok": len(detail_rows),
        "scale": {
            **scale_summary(scale_abs_err, scale_pct_err, scale_smape),
            "scale_factor_mae": (
                sum(scale_scale_abs_err) / len(scale_scale_abs_err)
                if scale_scale_abs_err
                else None
            ),
        },
        "has_handle": {
            "n": handle_total,
            "correct": handle_correct,
            "accuracy": handle_correct / handle_total if handle_total else None,
            **{k: handle_f1[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1")},
        },
        "door_type": {
            "n": door_total,
            "correct": door_correct,
            "accuracy": door_correct / door_total if door_total else None,
        },
        "by_category": by_category,
        "details": detail_rows,
        "has_handle_rows": handle_rows,
        "door_type_rows": door_rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"results: {report['results_path']}")
    print(f"bench:   {report['bench_path']}")
    print(f"evaluated ok: {report['evaluated_ok']} / {report['bench_assets']}")
    print()

    scale = report["scale"]
    if scale.get("n", 0):
        print("=== Scale (L_real_cm) ===")
        print(f"  n        = {scale['n']}")
        print(f"  MAE      = {scale['mae_cm']:.2f} cm")
        print(f"  MAPE     = {scale['mape_pct']:.1f} %  (|pred-gt|/gt)")
        print(f"  sMAPE    = {scale['smape_pct']:.1f} %  (FITR B.4, symmetric)")
        print(f"  RMSE     = {scale['rmse_cm']:.2f} cm")
        print(f"  max err  = {scale['max_abs_cm']:.2f} cm ({scale['max_pct']:.1f} %)")
        if scale.get("scale_factor_mae") is not None:
            print(f"  scale MAE= {scale['scale_factor_mae']:.3f}")
        print()

    handle = report["has_handle"]
    if handle["n"]:
        print("=== has_handle ===")
        print(f"  accuracy = {handle['correct']}/{handle['n']} ({handle['accuracy']:.1%})")
        if handle.get("f1") is not None:
            print(f"  F1       = {handle['f1']:.3f}")
            p, r = handle.get("precision"), handle.get("recall")
            p_txt = f"{p:.3f}" if p is not None else "—"
            r_txt = f"{r:.3f}" if r is not None else "—"
            print(f"  P/R      = {p_txt} / {r_txt}")
            print(f"  TP/FP/FN/TN = {handle['tp']}/{handle['fp']}/{handle['fn']}/{handle['tn']}")
        print()

    door = report["door_type"]
    if door["n"]:
        print("=== door_type (Door only) ===")
        print(f"  accuracy = {door['correct']}/{door['n']} ({door['accuracy']:.1%})")
        print()

    print("=== By category ===")
    for cat, stats in report["by_category"].items():
        s = stats["scale"]
        parts = [cat]
        if s.get("n"):
            parts.append(f"L sMAPE={s['smape_pct']:.1f}%")
        if stats["has_handle_n"]:
            f1 = stats.get("has_handle_f1")
            f1_txt = f"handle_F1={f1:.2f}" if f1 is not None else f"handle={stats['has_handle_acc']:.0%}"
            parts.append(f1_txt)
        if stats["door_type_n"]:
            parts.append(f"door={stats['door_type_acc']:.0%}")
        if stats["missing"]:
            parts.append(f"missing={len(stats['missing'])}")
        if stats["failed"]:
            parts.append(f"failed={len(stats['failed'])}")
        print("  " + " | ".join(parts))

    print()
    print("=== Per-asset (has_handle) ===")
    for row in report["has_handle_rows"]:
        mark = "OK" if row["ok"] else "WRONG"
        print(
            f"  {row['category']}/{row['asset_id']}/{row['joint_name']}: "
            f"pred={row['pred']} gt={row['gt']} {mark}"
        )

    print()
    print("=== Per-asset (scale) ===")
    for row in report["details"]:
        if "L_abs_err_cm" not in row:
            continue
        print(
            f"  {row['category']}/{row['asset_id']}: "
            f"pred={row['L_pred_cm']:.1f} gt={row['L_gt_cm']:.1f} "
            f"sMAPE={row['L_smape_pct']:.1f}% "
            f"(MAE={row['L_abs_err_cm']:.1f}cm)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VLM results on fitr_bench GT.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full report JSON (default: stdout only). "
        "Tip: experiments/eval/<category>_<baseline|ours>.json",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="NAME",
        help="Shortcut: write JSON to experiments/eval/NAME.json",
    )
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args()

    if not args.results.is_file():
        print(f"Results not found: {args.results}")
        return 1
    if not args.bench.is_file():
        print(f"Bench not found: {args.bench}")
        return 1

    report = eval_bench(args.results, args.bench, only_ok=not args.include_failed)
    print_report(report)

    json_out = args.json_out
    if args.report:
        json_out = eval_report_path(args.report)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
