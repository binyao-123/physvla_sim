"""URDF joint kinematics helpers for affordance derivation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JointRecord:
    name: str
    parent_link: str
    child_link: str
    motion_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


def _parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _parse_rpy(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def origin_to_matrix(origin_xyz: np.ndarray, origin_rpy: np.ndarray) -> np.ndarray:
    rot = rpy_to_matrix(origin_rpy)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = origin_xyz
    return mat


def transform_points(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return points @ rot.T + trans.reshape(1, 3)


def transform_direction(direction: np.ndarray, rot: np.ndarray) -> np.ndarray:
    vec = rot @ direction
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("degenerate direction under transform")
    return vec / norm


def load_urdf_root(asset_dir: Path) -> ET.Element:
    asset_dir = Path(asset_dir)
    for name in ("test.urdf", "mobility.urdf"):
        path = asset_dir / name
        if path.is_file():
            return ET.parse(path).getroot()
    raise FileNotFoundError(f"No URDF under {asset_dir}")


def parse_joints(root: ET.Element) -> dict[str, JointRecord]:
    joints: dict[str, JointRecord] = {}
    for joint_el in root.findall("joint"):
        jtype = joint_el.get("type", "fixed")
        parent = joint_el.find("parent")
        child = joint_el.find("child")
        if parent is None or child is None:
            continue
        origin = joint_el.find("origin")
        axis_el = joint_el.find("axis")
        origin_xyz = _parse_xyz(origin.get("xyz") if origin is not None else None)
        origin_rpy = _parse_rpy(origin.get("rpy") if origin is not None else None)
        axis = _parse_xyz(axis_el.get("xyz") if axis_el is not None else "0 0 1")
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = axis / norm
        name = joint_el.get("name") or f"joint_{len(joints)}"
        if jtype == "prismatic":
            motion = "prismatic"
        elif jtype in ("revolute", "continuous"):
            motion = "revolute"
        else:
            motion = "fixed"
        joints[name] = JointRecord(
            name=name,
            parent_link=parent.get("link", ""),
            child_link=child.get("link", ""),
            motion_type=motion,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
        )
    return joints


def movable_joints(joints: dict[str, JointRecord]) -> dict[str, JointRecord]:
    return {name: j for name, j in joints.items() if j.motion_type in ("revolute", "prismatic")}


def ordered_movable_joints(root: ET.Element) -> list[JointRecord]:
    """Movable joints in URDF document order (skips fixed joints)."""
    joints = parse_joints(root)
    ordered: list[JointRecord] = []
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        if not name or name not in joints:
            continue
        joint = joints[name]
        if joint.motion_type in ("revolute", "prismatic"):
            ordered.append(joint)
    return ordered


def find_base_link(root: ET.Element, joints: dict[str, JointRecord]) -> str:
    children = {j.child_link for j in joints.values()}
    parents = {j.parent_link for j in joints.values()}
    for link in root.findall("link"):
        name = link.get("name")
        if name and name not in children:
            return name
    for name in parents:
        if name not in children:
            return name
    return "base"


def link_pose_in_base(
    root: ET.Element,
    joints: dict[str, JointRecord],
    target_link: str,
) -> np.ndarray:
    """4x4 transform from target link frame to URDF base frame."""
    base_link = find_base_link(root, joints)
    if target_link == base_link:
        return np.eye(4, dtype=np.float64)

    parent_of: dict[str, JointRecord] = {}
    for joint in joints.values():
        parent_of[joint.child_link] = joint

    chain: list[JointRecord] = []
    link = target_link
    visited: set[str] = set()
    while link != base_link:
        if link in visited:
            raise ValueError(
                f"Kinematic cycle while resolving {target_link!r} to base {base_link!r}"
            )
        visited.add(link)
        if link not in parent_of:
            raise KeyError(f"No kinematic chain from {base_link} to {target_link}")
        joint = parent_of[link]
        chain.append(joint)
        link = joint.parent_link
    chain.reverse()

    pose = np.eye(4, dtype=np.float64)
    for joint in chain:
        pose = pose @ origin_to_matrix(joint.origin_xyz, joint.origin_rpy)
    return pose


def joint_axis_in_base(joint: JointRecord, root: ET.Element, joints: dict[str, JointRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Return hinge origin and unit axis in base frame."""
    parent_pose = link_pose_in_base(root, joints, joint.parent_link)
    joint_pose = parent_pose @ origin_to_matrix(joint.origin_xyz, joint.origin_rpy)
    origin = joint_pose[:3, 3]
    axis = transform_direction(joint.axis, joint_pose[:3, :3])
    return origin, axis
