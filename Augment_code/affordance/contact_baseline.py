"""Weak baselines for affordance ablation (not used as GT)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affordance.contact_axis import compute_contact_axis
from affordance.mesh_surface import load_link_surface_in_link_frame, load_mesh_instances_for_link
from affordance.urdf_kinematics import (
    link_pose_in_base,
    load_urdf_root,
    movable_joints,
    parse_joints,
    transform_points,
)


@dataclass(frozen=True)
class BaselineContactResult:
    position_xyz: list[float]
    contact_axis_xyz: list[float]
    source: str


def _round_vec(v: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in v.tolist()]


def compute_baseline_contact_point(
    asset_dir: Path,
    *,
    joint_name: str,
    link_name: str,
    motion_type: str,
    category: str,
) -> BaselineContactResult:
    """Link-mesh centroid + category/URDF heuristic axis (Table 5.2 baseline)."""
    asset_dir = Path(asset_dir)
    root = load_urdf_root(asset_dir)
    joints = parse_joints(root)
    movable = movable_joints(joints)
    joint = movable.get(joint_name)
    if joint is None and link_name:
        matched = [j for j in movable.values() if j.child_link == link_name]
        if len(matched) == 1:
            joint = matched[0]
    if joint is None:
        raise KeyError(f"Joint {joint_name} / link {link_name} not found")

    link_el = next(el for el in root.findall("link") if el.get("name") == joint.child_link)
    instances = load_mesh_instances_for_link(link_el)
    points_link, _ = load_link_surface_in_link_frame(asset_dir, instances)
    child_pose = link_pose_in_base(root, joints, joint.child_link)
    points_base = transform_points(points_link, child_pose[:3, :3], child_pose[:3, 3])
    centroid = points_base.mean(axis=0)

    axis, _ = compute_contact_axis(category, motion_type or joint.motion_type, asset_dir)
    return BaselineContactResult(
        position_xyz=_round_vec(centroid),
        contact_axis_xyz=axis,
        source="link_centroid_heuristic_axis",
    )
