"""Offline touch calibration from keyboard HDF5 (peak EE travel on link_1).

NOT used by auto_trajectory_collection at runtime. Workflow:
  1. Record touch with Keyboard_collection.py → HDF5
  2. scripts/inspect_touch_hdf5.py → print link-local contact
  3. Copy values into task_configs/*.yaml (push_contact_offset_link, contact_quat_link)

Legacy runtime path: articulation_calibrated / debug_link_contact_probe --mode articulation_push.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from mesh_utils import distance_to_axis
from reference.opening_kinematics import compose_pose, invert_pose


@dataclass(frozen=True)
class TouchContactReference:
    """Contact pose at touch, expressed in link_1 frame (survives object XY/yaw DR)."""

    demo_key: str
    peak_frame: int
    contact_pos_world_recorded: np.ndarray
    contact_quat_wxyz_world_recorded: tuple[float, float, float, float]
    contact_pos_link: np.ndarray
    contact_quat_wxyz_link: tuple[float, float, float, float]
    hinge_lever_m: float
    hinge_lever_vec_world: np.ndarray
    ee_home_world: np.ndarray


def _wxyz_to_rot(quat_wxyz: tuple[float, float, float, float]) -> R:
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w])


def _rot_to_wxyz(rot: R) -> tuple[float, float, float, float]:
    x, y, z, w = rot.as_quat()
    return (float(w), float(x), float(y), float(z))


def _as_wxyz(raw) -> tuple[float, float, float, float]:
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 4:
        raise ValueError(f"Expected quat length 4, got {arr.shape}")
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def hinge_lever_arm(
    contact_world: np.ndarray,
    hinge_origin_world: np.ndarray,
    hinge_axis_world: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Perpendicular distance from contact to hinge axis + lever vector in world frame."""
    origin = np.asarray(hinge_origin_world, dtype=np.float64)
    axis = np.asarray(hinge_axis_world, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    rel = np.asarray(contact_world, dtype=np.float64) - origin
    along = float(np.dot(rel, axis))
    lever = rel - along * axis
    dist = float(np.linalg.norm(lever))
    return dist, lever


def outward_normal_from_lever(lever_world: np.ndarray) -> np.ndarray:
    """Unit normal on outer push face (from hinge toward contact)."""
    lever = np.asarray(lever_world, dtype=np.float64)
    norm = float(np.linalg.norm(lever))
    if norm < 1e-9:
        raise ValueError("Contact lies on hinge axis; cannot define push face normal.")
    return lever / norm


def resolve_contact_world(
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    contact_pos_link: np.ndarray,
) -> np.ndarray:
    rot = _wxyz_to_rot(link_quat_wxyz)
    return np.asarray(link_pos_world, dtype=np.float64) + rot.apply(np.asarray(contact_pos_link, dtype=np.float64))


def resolve_contact_pose_world(
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    contact_pos_link: np.ndarray,
    contact_quat_wxyz_link: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pos, quat = compose_pose(
        link_pos_world,
        link_quat_wxyz,
        contact_pos_link,
        contact_quat_wxyz_link,
    )
    return np.asarray(pos, dtype=np.float64), quat


def approach_from_contact(
    contact_world: np.ndarray,
    contact_quat_wxyz: tuple[float, float, float, float],
    backoff_m: float,
) -> np.ndarray:
    """Retreat along gripper +Z (before contact, Z points into the push surface)."""
    rot = _wxyz_to_rot(contact_quat_wxyz)
    push_into_surface = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return np.asarray(contact_world, dtype=np.float64) - push_into_surface * float(backoff_m)


def load_touch_contact_from_hdf5(
    hdf5_path: Path,
    demo_index: int,
    *,
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    hinge_origin_world: np.ndarray,
    hinge_axis_world: np.ndarray,
) -> TouchContactReference:
    """Peak EE frame in demo = user touch on link_1 (farthest-from-hinge midpoint)."""
    import h5py

    demo_key = f"demo_{int(demo_index)}"
    with h5py.File(hdf5_path, "r") as h5_file:
        if demo_key not in h5_file["data"]:
            raise KeyError(f"Demo '{demo_key}' not found in {hdf5_path}")
        ee = h5_file[f"data/{demo_key}/obs/robot_eef_pos"][:, 0].astype(float)
        quat_raw = h5_file[f"data/{demo_key}/obs/robot_eef_quat"][:, 0].astype(float)

    ee_home = ee[0]
    travel = np.linalg.norm(ee - ee_home, axis=1)
    peak_frame = int(np.argmax(travel))
    if float(travel[peak_frame]) < 1e-4:
        raise ValueError(f"{demo_key} has no EE motion; touch the lid before saving.")

    contact_w = ee[peak_frame]
    contact_quat_w = _as_wxyz(quat_raw[peak_frame])

    link_pos_inv, link_quat_inv = invert_pose(link_pos_world, link_quat_wxyz)
    contact_pos_link, contact_quat_link = compose_pose(
        link_pos_inv,
        link_quat_inv,
        contact_w,
        contact_quat_w,
    )

    lever_m, lever_vec = hinge_lever_arm(contact_w, hinge_origin_world, hinge_axis_world)

    return TouchContactReference(
        demo_key=demo_key,
        peak_frame=peak_frame,
        contact_pos_world_recorded=contact_w.copy(),
        contact_quat_wxyz_world_recorded=contact_quat_w,
        contact_pos_link=np.asarray(contact_pos_link, dtype=np.float64),
        contact_quat_wxyz_link=contact_quat_link,
        hinge_lever_m=lever_m,
        hinge_lever_vec_world=lever_vec.copy(),
        ee_home_world=ee_home.copy(),
    )


def summarize_touch_reference(ref: TouchContactReference) -> str:
    return (
        f"{ref.demo_key} peak_frame={ref.peak_frame} "
        f"contact_w={np.round(ref.contact_pos_world_recorded, 4).tolist()} "
        f"contact_link={np.round(ref.contact_pos_link, 4).tolist()} "
        f"hinge_lever={ref.hinge_lever_m:.4f}m"
    )
