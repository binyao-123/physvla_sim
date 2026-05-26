"""Gripper orientation helpers (adapted from ArticuBot/manipulation/grasping_utils.py)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def align_gripper_z_with_normal(
    normal: np.ndarray,
    *,
    horizontal: bool = False,
    yaw_perturb_deg: float | None = None,
    flip: bool = False,
) -> R:
    """Align gripper +Z with `normal`; gripper +Y is horizontal or world-down based on `horizontal`."""
    gz = _normalize(np.asarray(normal, dtype=np.float64).reshape(3))
    if horizontal:
        gy_hint = np.array([0.0, 1.0 if flip else -1.0, 0.0], dtype=np.float64)
    else:
        gy_hint = np.array([0.0, 0.0, 1.0 if flip else -1.0], dtype=np.float64)

    gy = gy_hint - np.dot(gy_hint, gz) * gz
    if float(np.linalg.norm(gy)) < 1e-8:
        gy = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        gy = gy - np.dot(gy, gz) * gz
    gy = _normalize(gy)
    gx = _normalize(np.cross(gy, gz))
    rot = R.from_matrix(np.column_stack((gx, gy, gz)))

    if yaw_perturb_deg is not None and abs(yaw_perturb_deg) > 1e-9:
        rot = rot * R.from_rotvec(np.deg2rad(yaw_perturb_deg) * gy)
    return rot


def rotation_matrix_to_wxyz(rot: R) -> tuple[float, float, float, float]:
    x, y, z, w = rot.as_quat()
    return (float(w), float(x), float(y), float(z))
