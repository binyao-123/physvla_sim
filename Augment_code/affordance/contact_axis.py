"""Derive contact_axis_xyz per FITR 4.1.2 and category conventions."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Categories whose interaction stays in the horizontal XY plane (no Z motion).
XY_PLANE_CATEGORIES = {
    "Microwave",
    "Dishwasher",
    "Refrigerator",
    "StorageFurniture",
    "Door",
    "Drawer",
    "Faucet",
}

# Fallback unit contact axes (outward push / pull direction in asset frame).
CATEGORY_DEFAULT_AXIS: dict[str, list[float]] = {
    "Laptop": [0.0, -1.0, 0.0],
    "Display": [0.0, -1.0, 0.0],
    "Microwave": [0.0, -1.0, 0.0],
    "Drawer": [0.0, 1.0, 0.0],
    "Lamp": [1.0, 0.0, 0.0],
    "Faucet": [0.0, -1.0, 0.0],
    "Knife": [1.0, 0.0, 0.0],
    "Dishwasher": [0.0, -1.0, 0.0],
    "Door": [0.0, -1.0, 0.0],
    "Refrigerator": [0.0, -1.0, 0.0],
    "Scissors": [0.0, -1.0, 0.0],
    "StorageFurniture": [0.0, -1.0, 0.0],
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero-length vector")
    return v / n


def _round_vec(v: np.ndarray, ndigits: int = 6) -> list[float]:
    return [round(float(x), ndigits) for x in v.tolist()]


def read_primary_joint_axis(asset_dir: Path) -> tuple[str, np.ndarray]:
    """First movable joint axis from test.urdf or mobility.urdf."""
    for name in ("test.urdf", "mobility.urdf"):
        path = asset_dir / name
        if not path.is_file():
            continue
        tree = ET.parse(path)
        for joint in tree.findall("joint"):
            jtype = joint.get("type")
            if jtype not in ("revolute", "prismatic", "continuous"):
                continue
            axis_el = joint.find("axis")
            if axis_el is None:
                continue
            xyz = np.array([float(x) for x in axis_el.get("xyz", "0 0 1").split()], dtype=float)
            motion = "prismatic" if jtype == "prismatic" else "revolute"
            return motion, _normalize(xyz)
    raise FileNotFoundError(f"No movable joint in {asset_dir}")


def contact_axis_for_revolute(hinge_axis: np.ndarray, category: str) -> np.ndarray:
    """
    Outward contact normal for a revolute joint.
    Perpendicular to hinge axis; for XY-plane categories force Z=0.
    """
    ax, ay, az = hinge_axis
    if abs(ay) >= max(abs(ax), abs(az), 0.5):
        vec = np.array([1.0, 0.0, 0.0])
    elif abs(ax) >= max(abs(ay), abs(az), 0.5):
        vec = np.array([0.0, -1.0, 0.0])
    else:
        vec = np.array(CATEGORY_DEFAULT_AXIS[category], dtype=float)

    if category in XY_PLANE_CATEGORIES:
        vec[2] = 0.0
        if np.linalg.norm(vec) > 1e-6:
            return _normalize(vec)
    return _normalize(vec)


def contact_axis_for_prismatic(joint_axis: np.ndarray, category: str) -> np.ndarray:
    """Prismatic: contact axis aligns with joint axis (FITR 4.1.2)."""
    if category in XY_PLANE_CATEGORIES:
        xy = np.array([joint_axis[0], joint_axis[1], 0.0])
        if np.linalg.norm(xy) > 1e-6:
            return _normalize(xy)
        return np.array(CATEGORY_DEFAULT_AXIS[category], dtype=float)
    return joint_axis


def compute_contact_axis(
    category: str,
    motion_type: str,
    asset_dir: Path,
) -> tuple[list[float], str]:
    try:
        urdf_motion, hinge_axis = read_primary_joint_axis(asset_dir)
    except FileNotFoundError:
        vec = np.array(CATEGORY_DEFAULT_AXIS[category], dtype=float)
        return _round_vec(vec), "category_default"

    motion = motion_type or urdf_motion
    if motion == "prismatic":
        vec = contact_axis_for_prismatic(hinge_axis, category)
        source = f"prismatic_axis->{_round_vec(hinge_axis)}"
    else:
        vec = contact_axis_for_revolute(hinge_axis, category)
        source = f"revolute_hinge->{_round_vec(hinge_axis)}"
    return _round_vec(vec), source
