#!/usr/bin/env python3
"""Merge geometry, dynamics, affordances, and VLM outputs into per-asset info1.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from affordance.contact_axis import compute_contact_axis
from experiment_paths import DEFAULT_VLM_RESULTS
from prompt.load_prompt import list_movable_joints
from scale.l_norm import compute_l_norm, compute_scale

AUGMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = AUGMENT_ROOT.parent / "datasets" / "data_normalized"
DEFAULT_VLM_BASE = AUGMENT_ROOT / "vlm_base_template.json"
DEFAULT_DYNAMICS = AUGMENT_ROOT / "joint_dynamics_template.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_vlm_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok" or not rec.get("result"):
                continue
            key = (rec["category"], str(rec["asset_id"]))
            prev = latest.get(key)
            if prev is None or rec.get("timestamp", "") >= prev.get("timestamp", ""):
                latest[key] = rec["result"]
    return latest


def iter_assets(
    vlm_base: dict[str, Any],
    dataset_root: Path,
    *,
    categories: list[str] | None = None,
) -> Iterator[tuple[str, str, Path]]:
    cat_names = categories or list(vlm_base["categories"])
    for category in cat_names:
        cfg = vlm_base["categories"][category]
        cat_dir = dataset_root / cfg["category_dir"]
        if not cat_dir.is_dir():
            continue
        for asset_dir in sorted(cat_dir.iterdir()):
            if asset_dir.is_dir():
                yield category, asset_dir.name, asset_dir


def build_info1(
    category: str,
    asset_id: str,
    asset_dir: Path,
    *,
    vlm_base: dict[str, Any],
    dynamics: dict[str, Any],
    vlm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    cat_cfg = vlm_base["categories"][category]
    dyn = dynamics["categories"][category]

    l_norm_val = float(compute_l_norm(asset_dir)["L_norm"])

    l_real: float | None = None
    if vlm_result and vlm_result.get("longest_edge_cm") is not None:
        l_real = float(vlm_result["longest_edge_cm"])

    scale_val: float | None = None
    if l_real is not None:
        scale_val = float(compute_scale(l_real, l_norm_val))

    door_type = vlm_result.get("door_type") if vlm_result else None

    stiffness = float(dyn["stiffness"])
    if category == "Door":
        by_type = cat_cfg.get("stiffness_by_door_type", {})
        if door_type in by_type:
            stiffness = float(by_type[door_type])

    has_handle_map: dict[str, bool] = {}
    if vlm_result and vlm_result.get("has_handle"):
        has_handle_map = {
            str(k): bool(v) for k, v in vlm_result["has_handle"].items()
        }

    joints_out: list[dict[str, Any]] = []
    affordances_out: list[dict[str, Any]] = []

    for joint in list_movable_joints(asset_dir):
        joint_entry: dict[str, Any] = {
            "joint_name": joint.joint_name,
            "link_name": joint.link_name,
            "motion_type": joint.motion_type,
            "damping": float(dyn["damping"]),
            "stiffness": stiffness,
            "effort_limit": float(dyn["effort_limit"]),
        }
        if category == "Door":
            joint_entry["door_type"] = door_type
        joints_out.append(joint_entry)

        axis, _source = compute_contact_axis(category, joint.motion_type, asset_dir)
        has_handle = has_handle_map.get(joint.joint_name)
        if has_handle is None and "has_handle" not in cat_cfg.get("vlm_tasks", []):
            has_handle = False

        affordances_out.append(
            {
                "joint_name": joint.joint_name,
                "link_name": joint.link_name,
                "motion_type": joint.motion_type,
                "position_xyz": [None, None, None],
                "contact_axis_xyz": axis,
                "has_handle": has_handle,
            }
        )

    return {
        "schema_version": "1.0",
        "asset_id": asset_id,
        "category": category,
        "scale": {
            "L_norm": l_norm_val,
            "L_real_cm": l_real,
            "scale": scale_val,
        },
        "joints": joints_out,
        "affordances": affordances_out,
    }


def write_info1(asset_dir: Path, info1: dict[str, Any], *, dry_run: bool) -> Path:
    out_path = asset_dir / "info1.json"
    if not dry_run:
        out_path.write_text(
            json.dumps(info1, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write info1.json for dataset assets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--vlm-base", type=Path, default=DEFAULT_VLM_BASE)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--vlm-results", type=Path, default=DEFAULT_VLM_RESULTS)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing info1.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.asset_id and (not args.category or len(args.category) != 1):
        parser.error("--asset-id requires exactly one --category")

    vlm_base = load_json(args.vlm_base)
    dynamics = load_json(args.dynamics)
    vlm_map = load_latest_vlm_results(args.vlm_results)

    if args.asset_id:
        cat = args.category[0]
        cfg = vlm_base["categories"][cat]
        work = [(cat, args.asset_id, args.dataset_root / cfg["category_dir"] / args.asset_id)]
    else:
        work = list(
            iter_assets(vlm_base, args.dataset_root, categories=args.category)
        )

    if args.limit is not None:
        work = work[: args.limit]

    written = 0
    skipped = 0
    errors = 0

    for category, asset_id, asset_dir in work:
        out_path = asset_dir / "info1.json"
        if not asset_dir.is_dir():
            print(f"[error] missing dir {asset_dir}", file=sys.stderr)
            errors += 1
            continue
        if out_path.is_file() and args.skip_existing and not args.force:
            skipped += 1
            continue

        try:
            info1 = build_info1(
                category,
                asset_id,
                asset_dir,
                vlm_base=vlm_base,
                dynamics=dynamics,
                vlm_result=vlm_map.get((category, asset_id)),
            )
            write_info1(asset_dir, info1, dry_run=args.dry_run)
            written += 1
            if written <= 5 or args.asset_id or args.limit:
                scale = info1["scale"]
                print(
                    f"[{'dry' if args.dry_run else 'ok'}] {category}/{asset_id} "
                    f"L_norm={scale['L_norm']:.3f} L_real={scale['L_real_cm']} "
                    f"joints={len(info1['joints'])}"
                )
        except Exception as exc:
            print(f"[error] {category}/{asset_id}: {exc}", file=sys.stderr)
            errors += 1

    if written > 5 and not args.asset_id:
        print(f"... ({written} total written)")
    print(f"done written={written} skipped={skipped} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
