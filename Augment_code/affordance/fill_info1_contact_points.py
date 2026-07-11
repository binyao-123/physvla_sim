#!/usr/bin/env python3
"""Fill info1.json affordances[].position_xyz via FITR 4.1.2 contact-point derivation.

Reproduce (cd physvla_sim/Augment_code):

  # Report contact-link meshes above density thresholds (no writes)
  .venv/bin/python affordance/fill_info1_contact_points.py --report-dense

  # Fill all pending affordances (skips dense contact-link meshes by default)
  .venv/bin/python affordance/fill_info1_contact_points.py

  # Single category / asset
  .venv/bin/python affordance/fill_info1_contact_points.py --category Dishwasher
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AUGMENT_ROOT = Path(__file__).resolve().parents[1]
if str(AUGMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUGMENT_ROOT))

from affordance.contact_point import compute_contact_point
from affordance.mesh_surface import (
    DENSE_FACE_THRESHOLD,
    DENSE_VERT_THRESHOLD,
    is_dense_contact_link_mesh,
)

DEFAULT_DATASET = AUGMENT_ROOT.parent / "datasets" / "data_normalized"
DEFAULT_VLM_BASE = AUGMENT_ROOT / "vlm_base_template.json"


def iter_assets(
    vlm_base: dict,
    dataset_root: Path,
    *,
    categories: list[str] | None,
) -> list[tuple[str, str, Path]]:
    names = categories or list(vlm_base["categories"])
    work: list[tuple[str, str, Path]] = []
    for category in names:
        cfg = vlm_base["categories"][category]
        cat_dir = dataset_root / cfg["category_dir"]
        if not cat_dir.is_dir():
            continue
        for asset_dir in sorted(cat_dir.iterdir()):
            if asset_dir.is_dir():
                work.append((category, asset_dir.name, asset_dir))
    return work


def fill_info1_contact_points(
    info1: dict,
    asset_dir: Path,
    *,
    category: str,
    overwrite: bool = False,
    skip_dense: bool = True,
) -> tuple[int, int, list[str]]:
    updated = 0
    skipped_dense = 0
    dense_notes: list[str] = []
    for aff in info1.get("affordances", []):
        pos = aff.get("position_xyz")
        if not overwrite and pos and all(v is not None for v in pos):
            continue
        joint_name = str(aff["joint_name"])
        link_name = str(aff.get("link_name", ""))
        if skip_dense:
            dense, reason = is_dense_contact_link_mesh(
                asset_dir,
                joint_name=joint_name,
                link_name=link_name,
            )
            if dense:
                skipped_dense += 1
                dense_notes.append(f"{joint_name}({reason})")
                continue
        result = compute_contact_point(
            asset_dir,
            joint_name=joint_name,
            link_name=link_name,
            motion_type=str(aff.get("motion_type", "revolute")),
            category=category,
        )
        aff["position_xyz"] = result.position_xyz
        aff["contact_axis_xyz"] = result.contact_axis_xyz
        updated += 1
    return updated, skipped_dense, dense_notes


def report_dense_meshes(
    vlm_base: dict,
    dataset_root: Path,
    *,
    categories: list[str] | None,
) -> dict[str, int]:
    work = iter_assets(vlm_base, dataset_root, categories=categories)
    heavy_rows: list[tuple[str, str, str, str]] = []
    heavy_assets: set[tuple[str, str]] = set()
    n_aff = 0
    for category, asset_id, asset_dir in work:
        info_path = asset_dir / "info1.json"
        if not info_path.is_file():
            continue
        info1 = json.loads(info_path.read_text(encoding="utf-8"))
        for aff in info1.get("affordances", []):
            n_aff += 1
            joint_name = str(aff["joint_name"])
            link_name = str(aff.get("link_name", ""))
            try:
                dense, reason = is_dense_contact_link_mesh(
                    asset_dir,
                    joint_name=joint_name,
                    link_name=link_name,
                )
            except Exception as exc:
                print(f"[warn] {category}/{asset_id}/{joint_name}: {exc}", file=sys.stderr)
                continue
            if dense:
                heavy_rows.append((category, asset_id, joint_name, reason))
                heavy_assets.add((category, asset_id))

    print(
        f"dense_threshold: verts>{DENSE_VERT_THRESHOLD} or faces>{DENSE_FACE_THRESHOLD} "
        f"(contact-link meshes only)"
    )
    print(f"dense_affordances: {len(heavy_rows)} / {n_aff}")
    print(f"dense_assets: {len(heavy_assets)} / {len(work)}")
    for row in sorted(heavy_rows, key=lambda r: (r[0], r[1], r[2])):
        print(f"  {row[0]}/{row[1]}/{row[2]}  {row[3]}")
    return {
        "dense_affordances": len(heavy_rows),
        "dense_assets": len(heavy_assets),
        "total_affordances": n_aff,
        "total_assets": len(work),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill info1 affordance contact points (FITR 4.1.2).")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vlm-base", type=Path, default=DEFAULT_VLM_BASE)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing position_xyz")
    parser.add_argument(
        "--no-skip-dense",
        action="store_true",
        help=f"Do not skip contact-link meshes with >{DENSE_VERT_THRESHOLD} verts "
        f"or >{DENSE_FACE_THRESHOLD} faces",
    )
    parser.add_argument(
        "--report-dense",
        action="store_true",
        help="Only print dense contact-link mesh statistics and exit",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.asset_id and (not args.category or len(args.category) != 1):
        parser.error("--asset-id requires exactly one --category")

    vlm_base = json.loads(args.vlm_base.read_text(encoding="utf-8"))

    if args.report_dense:
        report_dense_meshes(vlm_base, args.dataset_root, categories=args.category)
        return 0

    if args.asset_id:
        cat = args.category[0]
        cfg = vlm_base["categories"][cat]
        work = [(cat, args.asset_id, args.dataset_root / cfg["category_dir"] / args.asset_id)]
    else:
        work = iter_assets(vlm_base, args.dataset_root, categories=args.category)
    if args.limit is not None:
        work = work[: args.limit]

    total = 0
    skipped_dense = 0
    errors = 0
    skip_dense = not args.no_skip_dense
    for category, asset_id, asset_dir in work:
        info_path = asset_dir / "info1.json"
        if not info_path.is_file():
            print(f"[skip] {category}/{asset_id}: no info1.json", file=sys.stderr)
            continue
        info1 = json.loads(info_path.read_text(encoding="utf-8"))
        try:
            n, n_dense, dense_notes = fill_info1_contact_points(
                info1,
                asset_dir,
                category=category,
                overwrite=args.force,
                skip_dense=skip_dense,
            )
        except Exception as exc:
            print(f"[error] {category}/{asset_id}: {exc}", file=sys.stderr)
            errors += 1
            continue
        skipped_dense += n_dense
        if n_dense and not n:
            print(
                f"[skip-dense] {category}/{asset_id} skipped={n_dense} "
                + "; ".join(dense_notes),
                file=sys.stderr,
            )
            continue
        if n == 0:
            continue
        total += n
        sample = info1["affordances"][0]
        print(
            f"[ok] {category}/{asset_id} updated={n} "
            f"pos={sample.get('position_xyz')} axis={sample.get('contact_axis_xyz')}"
        )
        if n_dense:
            print(
                f"[skip-dense] {category}/{asset_id} skipped={n_dense} "
                + "; ".join(dense_notes),
                file=sys.stderr,
            )
        if not args.dry_run:
            info_path.write_text(json.dumps(info1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"finished affordances_updated={total} skipped_dense={skipped_dense} errors={errors}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
