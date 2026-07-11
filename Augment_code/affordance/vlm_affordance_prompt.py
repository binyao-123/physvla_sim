"""Assemble VLM prompt for affordance position_xyz prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prompt.load_prompt import DEFAULT_BENCH_PATH, DEFAULT_DATASET_ROOT, DEFAULT_VLM_BASE_PATH, find_category_scale_image, load_calibration_config

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt" / "affordance_position_xyz.txt"

@dataclass(frozen=True)
class AffordanceSpec:
    joint_name: str
    link_name: str
    motion_type: str
    contact_axis_xyz: list[float]
    has_handle: bool | None
    link_bbox_min_xyz: list[float] | None = None
    link_bbox_max_xyz: list[float] | None = None
    link_bbox_center_xyz: list[float] | None = None
    link_bbox_size_xyz: list[float] | None = None
    joint_origin_xyz: list[float] | None = None
    joint_axis_xyz: list[float] | None = None


@dataclass(frozen=True)
class AffordancePromptBundle:
    category: str
    asset_id: str
    asset_dir: Path
    image_path: Path
    affordances: tuple[AffordanceSpec, ...]
    system: str
    user: str


def parse_affordance_prompt_file(path: Path | None = None) -> dict[str, str]:
    path = Path(path or PROMPT_PATH)
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[1:-1]
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _fmt_bool(v: bool | None) -> str:
    if v is None:
        return "unknown"
    return "true" if v else "false"


def _fmt_vec(v: list[float] | None) -> str:
    if v is None:
        return "unknown"
    return ", ".join(f"{float(x):.4g}" for x in v)


def _round_vec(v: np.ndarray) -> list[float]:
    return [round(float(x), 4) for x in v.tolist()]


def _enrich_with_geometry(
    asset_dir: Path,
    specs: list[AffordanceSpec],
) -> list[AffordanceSpec]:
    """Attach URDF-derived link bbox/joint geometry without using benchmark GT."""
    try:
        from affordance.contact_point import _resolve_movable_joint
        from affordance.mesh_surface import load_link_surface_in_link_frame, load_mesh_instances_for_link
        from affordance.urdf_kinematics import (
            joint_axis_in_base,
            link_pose_in_base,
            load_urdf_root,
            movable_joints,
            parse_joints,
            transform_points,
        )

        root = load_urdf_root(asset_dir)
        joints = parse_joints(root)
        movable = movable_joints(joints)
    except Exception:
        return specs

    enriched: list[AffordanceSpec] = []
    for spec in specs:
        try:
            joint = _resolve_movable_joint(
                asset_dir,
                movable,
                root,
                joint_name=spec.joint_name,
                link_name=spec.link_name,
            )
            link_el = next(
                (el for el in root.findall("link") if el.get("name") == joint.child_link),
                None,
            )
            if link_el is None:
                raise KeyError(joint.child_link)
            instances = load_mesh_instances_for_link(link_el)
            points_link, _ = load_link_surface_in_link_frame(asset_dir, instances)
            child_pose = link_pose_in_base(root, joints, joint.child_link)
            points_base = transform_points(points_link, child_pose[:3, :3], child_pose[:3, 3])
            bbox_min = points_base.min(axis=0)
            bbox_max = points_base.max(axis=0)
            origin, axis = joint_axis_in_base(joint, root, joints)
            enriched.append(
                AffordanceSpec(
                    joint_name=spec.joint_name,
                    link_name=spec.link_name,
                    motion_type=spec.motion_type,
                    contact_axis_xyz=spec.contact_axis_xyz,
                    has_handle=spec.has_handle,
                    link_bbox_min_xyz=_round_vec(bbox_min),
                    link_bbox_max_xyz=_round_vec(bbox_max),
                    link_bbox_center_xyz=_round_vec((bbox_min + bbox_max) * 0.5),
                    link_bbox_size_xyz=_round_vec(bbox_max - bbox_min),
                    joint_origin_xyz=_round_vec(origin),
                    joint_axis_xyz=_round_vec(axis),
                )
            )
        except Exception:
            enriched.append(spec)
    return enriched


def build_affordance_block(
    affordances: list[AffordanceSpec],
    *,
    sections: dict[str, str],
) -> str:
    line_tpl = sections["AFFORDANCE_LINE"]
    lines: list[str] = []
    for aff in affordances:
        axis = aff.contact_axis_xyz or [0.0, -1.0, 0.0]
        lines.append(
            line_tpl.format(
                joint_name=aff.joint_name,
                link_name=aff.link_name,
                motion_type=aff.motion_type,
                axis_x=axis[0],
                axis_y=axis[1],
                axis_z=axis[2],
                has_handle=_fmt_bool(aff.has_handle),
                bbox_min=_fmt_vec(aff.link_bbox_min_xyz),
                bbox_max=_fmt_vec(aff.link_bbox_max_xyz),
                bbox_center=_fmt_vec(aff.link_bbox_center_xyz),
                bbox_size=_fmt_vec(aff.link_bbox_size_xyz),
                joint_origin=_fmt_vec(aff.joint_origin_xyz),
                joint_axis=_fmt_vec(aff.joint_axis_xyz),
            )
        )
    return "\n".join(lines)


def build_affordance_prompt(
    category: str,
    asset_id: str,
    affordances: list[AffordanceSpec],
    *,
    sections: dict[str, str] | None = None,
) -> tuple[str, str]:
    sections = sections or parse_affordance_prompt_file()
    block = build_affordance_block(affordances, sections=sections)
    user = sections["USER"].format(
        category=category,
        asset_id=asset_id,
        affordance_block=block,
    )
    return sections["SYSTEM"], user


def affordances_from_bench_asset(asset: dict[str, Any]) -> list[AffordanceSpec]:
    specs: list[AffordanceSpec] = []
    for aff in asset.get("affordances", []):
        pos = aff.get("position_xyz")
        if not pos or any(v is None for v in pos):
            continue
        axis = aff.get("contact_axis_xyz") or [0.0, -1.0, 0.0]
        specs.append(
            AffordanceSpec(
                joint_name=str(aff["joint_name"]),
                link_name=str(aff.get("link_name", "")),
                motion_type=str(aff.get("motion_type", "revolute")),
                contact_axis_xyz=[float(x) for x in axis],
                has_handle=aff.get("has_handle"),
            )
        )
    return specs


def build_prompt_for_bench_asset(
    category: str,
    asset: dict[str, Any],
    *,
    dataset_root: Path | None = None,
    vlm_base_path: Path | None = None,
) -> AffordancePromptBundle:
    dataset_root = Path(dataset_root or DEFAULT_DATASET_ROOT)
    vlm_base = json.loads((vlm_base_path or DEFAULT_VLM_BASE_PATH).read_text(encoding="utf-8"))
    cat_cfg = vlm_base["categories"][category]
    asset_id = str(asset["asset_id"])
    asset_dir = dataset_root / cat_cfg["category_dir"] / asset_id
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    affordances = affordances_from_bench_asset(asset)
    if not affordances:
        raise ValueError(f"No evaluable affordances for {category}/{asset_id}")
    affordances = _enrich_with_geometry(asset_dir, affordances)
    sections = parse_affordance_prompt_file()

    calib = load_calibration_config()
    image_path = find_category_scale_image(asset_dir, category, calib)
    system, user = build_affordance_prompt(
        category,
        asset_id,
        affordances,
        sections=sections,
    )
    return AffordancePromptBundle(
        category=category,
        asset_id=asset_id,
        asset_dir=asset_dir,
        image_path=image_path,
        affordances=tuple(affordances),
        system=system,
        user=user,
    )


def load_bench_assets(
    bench_path: Path,
    *,
    categories: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    out: list[tuple[str, dict[str, Any]]] = []
    for category, data in bench["categories"].items():
        if categories and category not in categories:
            continue
        for asset in data["assets"]:
            out.append((category, asset))
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Preview affordance VLM prompt for one bench asset.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH_PATH)
    args = parser.parse_args()

    bench = json.loads(args.bench_path.read_text(encoding="utf-8"))
    asset = next(
        a
        for a in bench["categories"][args.category]["assets"]
        if str(a["asset_id"]) == str(args.asset_id)
    )
    bundle = build_prompt_for_bench_asset(
        args.category,
        asset,
    )
    print("=== SYSTEM ===")
    print(bundle.system)
    print("\n=== USER ===")
    print(bundle.user)
    print(f"\nimage: {bundle.image_path}")
    print(f"joints: {[a.joint_name for a in bundle.affordances]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
