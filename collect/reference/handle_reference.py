"""Resolve push handle from link-local yaml (no HDF5 runtime dependency)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from reference.contact_reference import resolve_contact_pose_world
from reference.grasping_utils import align_gripper_z_with_normal, rotation_matrix_to_wxyz


def _wxyz_to_rot(quat_wxyz: tuple[float, float, float, float]) -> R:
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w])


def derive_contact_quat_link(
    link_quat_wxyz: tuple[float, float, float, float],
    approach_direction_world: tuple[float, float, float],
    *,
    horizontal: bool = True,
) -> tuple[float, float, float, float]:
    """Gripper +Z aligned with approach direction, expressed in link frame."""
    approach = np.asarray(approach_direction_world, dtype=np.float64)
    norm = float(np.linalg.norm(approach))
    if norm < 1e-9:
        raise ValueError("approach_direction_world must be non-zero.")
    approach /= norm

    rot_world = align_gripper_z_with_normal(approach, horizontal=horizontal)
    rot_link = _wxyz_to_rot(link_quat_wxyz).inv() * rot_world
    return rotation_matrix_to_wxyz(rot_link)


def resolve_yaml_handle_world(
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    contact_pos_link: tuple[float, float, float],
    contact_quat_link: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    return resolve_contact_pose_world(
        link_pos_world,
        link_quat_wxyz,
        np.asarray(contact_pos_link, dtype=np.float64),
        contact_quat_link,
    )
