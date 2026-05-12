"""
Piper SDK (3080) integer encoding versus Isaac Lab HDF5 semantics.

Canonical reference (SDK demo):
  piper_sdk/demo/V2/piper_ctrl_joint.py
    - JointCmd: angle_rad * factor,  factor ≈ 1000*180/pi (= 57295.7795 in demo).
    - Gripper:  position_scalar * (1000*1000) for GripperCtrl.

Decoding integers from GetArmJointMsgs / CAN-style feedback::
    joint_rad   = joint_int / JOINT_DECODE_DIVISOR
    grip_phys   = gripper_int / GRIPPER_SDK_SCALE

Isaac ``Keyboard_collection.py`` HDF5::
    obs/robot_joint_pos[:6]  — robot.data.joint_pos arm DOFs ⇒ **already radians**.
    obs/robot_joint_pos[6]   — mimic gripper joint angle ⇒ **radians**, not SDK ints.
    actions[:6]              — IK joint targets ⇒ **radians**.
    actions[6]               — discrete open/close **0 / 1** (NOT radians).

So: **do not** divide HDF5 floats by SDK factors unless the file explicitly stores CAN integers
cast to floats (future teleop / bag replay pipeline).
"""

from __future__ import annotations

import math

import numpy as np

JOINT_DECODE_DIVISOR = 1000.0 * 180.0 / math.pi

GRIPPER_SDK_SCALE = 1_000_000

SDK_DEMO_FACTOR_JOINT_ROUNDED = 57295.7795


def sdk_joint_int_to_rad(joint_int: np.ndarray) -> np.ndarray:
    """(6,) or (N,6) integer joints from SDK → radians."""
    j = np.asarray(joint_int, dtype=np.float64)
    return j / JOINT_DECODE_DIVISOR


def sdk_gripper_int_to_unit(gripper_int: np.ndarray) -> np.ndarray:
    """Scalar or (N,) gripper feedback integers → GripperCtrl-side continuous scale."""
    g = np.asarray(gripper_int, dtype=np.float64)
    return g / GRIPPER_SDK_SCALE


def decode_joint7_vector_if_sdk_floats(v: np.ndarray) -> np.ndarray:
    """If rows are CAN integers cast to float, decode joints to radians & gripper to Gripper scale.

    Supports shape (7,) or (T, 7). **Do not use** on default Isaac HDF5 (`Keyboard_collection.py`).
    """
    arr = np.asarray(v, dtype=np.float64).copy()
    if arr.ndim == 1:
        arr[:6] /= JOINT_DECODE_DIVISOR
        arr[6] /= GRIPPER_SDK_SCALE
        return arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 7:
        arr[:, :6] /= JOINT_DECODE_DIVISOR
        arr[:, 6] /= GRIPPER_SDK_SCALE
        return arr.astype(np.float32)
    raise ValueError(f"Expected shape (7,) or (T,7), got {arr.shape}")
