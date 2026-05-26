"""Articulation-relative end-effector trajectories (ArticuBot open_door math, no PyBullet)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _wxyz_to_rot(quat_wxyz: tuple[float, float, float, float]) -> R:
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w])


def _rot_to_wxyz(rot: R) -> tuple[float, float, float, float]:
    x, y, z, w = rot.as_quat()
    return (float(w), float(x), float(y), float(z))


def compose_pose(
    parent_pos: np.ndarray,
    parent_quat_wxyz: tuple[float, float, float, float],
    child_pos: np.ndarray,
    child_quat_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    r_parent = _wxyz_to_rot(parent_quat_wxyz)
    r_child = _wxyz_to_rot(child_quat_wxyz)
    pos = parent_pos + r_parent.apply(child_pos)
    rot = r_parent * r_child
    return pos, _rot_to_wxyz(rot)


def invert_pose(
    pos: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    r = _wxyz_to_rot(quat_wxyz)
    r_inv = r.inv()
    pos_inv = -r_inv.apply(pos)
    return pos_inv, _rot_to_wxyz(r_inv)


def link_pose_at_delta_angle(
    link_pos_init: np.ndarray,
    link_quat_init_wxyz: tuple[float, float, float, float],
    hinge_origin_world: np.ndarray,
    hinge_axis_world: np.ndarray,
    delta_theta_rad: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    axis = hinge_axis_world / max(np.linalg.norm(hinge_axis_world), 1e-9)
    rot_delta = R.from_rotvec(delta_theta_rad * axis)
    r_init = _wxyz_to_rot(link_quat_init_wxyz)
    r_new = rot_delta * r_init
    pos_new = hinge_origin_world + rot_delta.apply(link_pos_init - hinge_origin_world)
    return pos_new, _rot_to_wxyz(r_new)


def compute_articulation_ee_trajectory(
    *,
    eef_pos_world: np.ndarray,
    eef_quat_wxyz: tuple[float, float, float, float],
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    hinge_origin_world: np.ndarray,
    hinge_axis_world: np.ndarray,
    theta_init_rad: float,
    theta_targets_rad: tuple[float, ...],
) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    """T_rel = T_link^{-1} T_eef held constant while link rotates (ArticuBot Sec IV-A)."""
    link_pos_inv, link_quat_inv = invert_pose(link_pos_world, link_quat_wxyz)
    eef_in_link_pos, eef_in_link_quat = compose_pose(
        link_pos_inv, link_quat_inv, eef_pos_world, eef_quat_wxyz
    )

    trajectory: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
    for theta in theta_targets_rad:
        delta = float(theta) - float(theta_init_rad)
        link_pos_t, link_quat_t = link_pose_at_delta_angle(
            link_pos_world,
            link_quat_wxyz,
            hinge_origin_world,
            hinge_axis_world,
            delta,
        )
        eef_pos_t, eef_quat_t = compose_pose(link_pos_t, link_quat_t, eef_in_link_pos, eef_in_link_quat)
        trajectory.append((eef_pos_t, eef_quat_t))
    return trajectory
