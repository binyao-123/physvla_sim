"""
Piper SDK (3080) integer encoding versus Isaac Lab HDF5 semantics.

Canonical reference (SDK demo):
  piper_sdk/demo/V2/piper_ctrl_joint.py
    - JointCmd: angle_rad * factor,  factor ≈ 1000*180/pi (= 57295.7795 in demo).
    - GripperCtrl: ``round(scalar * 1_000_000)`` where scalar is the float in Piper_buf
      (see 3080 ``piper_controller`` action path).

Gripper feedback / command range (confirmed on hardware):
    - SDK integer ``grippers_angle``: **0 = 全关**, **100_000 = 全开**
    - Unit: **0.001 mm per count** → 100 mm travel at full open
    - Normalized for π0.5 training (0/1): ``gripper_int / GRIPPER_SDK_INT_MAX``
    - Legacy Zenoh / ``record_data`` wire scalar: ``gripper_int / GRIPPER_CTRL_ENCODE_SCALE``
      (i.e. ÷1e6, open ≈ **0.1** when int=100_000)

Isaac ``Keyboard_collection.py`` HDF5 (current)::
    obs/robot_joint_pos[:6]  — arm joint **radians**.
    obs/robot_joint_pos[6]   — gripper **0 / 1** (open / close), same semantics as actions[...,6].

Legacy HDF5 (before gripper obs change)::
    obs/robot_joint_pos[6]   — mimic gripper joint angle in **radians**; convert with
    ``--gripper-obs-mode binary`` to copy from actions[...,6].
    actions[:6]              — IK joint targets ⇒ **radians**.
    actions[6]               — discrete open/close **0 / 1** (NOT SDK integers).

So: **do not** divide HDF5 floats by SDK factors unless the file explicitly stores CAN integers
cast to floats (use ``--joint-value-source piper_sdk_can_float``).
"""

from __future__ import annotations

import math

import numpy as np

JOINT_DECODE_DIVISOR = 1000.0 * 180.0 / math.pi

SDK_DEMO_FACTOR_JOINT_ROUNDED = 57295.7795

# Gripper SDK integer range (GetArmGripperMsgs / GripperCtrl)
GRIPPER_SDK_INT_MIN = 0
GRIPPER_SDK_INT_MAX = 100_000
GRIPPER_MM_PER_COUNT = 0.001
GRIPPER_OPEN_TRAVEL_MM = GRIPPER_SDK_INT_MAX * GRIPPER_MM_PER_COUNT  # 100.0 mm

# Float on Piper_buf / record_data / piper_controller state (grippers_angle ÷ 1e6)
GRIPPER_CTRL_ENCODE_SCALE = 1_000_000
GRIPPER_CTRL_SCALAR_MAX = GRIPPER_SDK_INT_MAX / GRIPPER_CTRL_ENCODE_SCALE  # 0.1

# Default open/close threshold on raw SDK counts
GRIPPER_SDK_OPEN_THRESHOLD = 50_000

# Back-compat alias (prefer GRIPPER_CTRL_ENCODE_SCALE for ÷1e6 wire format)
GRIPPER_SDK_SCALE = GRIPPER_CTRL_ENCODE_SCALE


def sdk_joint_int_to_rad(joint_int: np.ndarray) -> np.ndarray:
    """(6,) or (N,6) integer joints from SDK → radians."""
    j = np.asarray(joint_int, dtype=np.float64)
    return j / JOINT_DECODE_DIVISOR


def sdk_gripper_int_to_mm(gripper_int: np.ndarray) -> np.ndarray:
    """SDK gripper count → millimetres (0 … 100 mm)."""
    g = np.asarray(gripper_int, dtype=np.float64)
    return np.clip(g, GRIPPER_SDK_INT_MIN, GRIPPER_SDK_INT_MAX) * GRIPPER_MM_PER_COUNT


def sdk_gripper_int_to_normalized(gripper_int: np.ndarray) -> np.ndarray:
    """SDK 0…100_000 → [0, 1] (matches π0.5 training gripper dim)."""
    g = np.asarray(gripper_int, dtype=np.float64)
    return np.clip(g / GRIPPER_SDK_INT_MAX, 0.0, 1.0)


def sdk_gripper_int_to_ctrl_scalar(gripper_int: np.ndarray) -> np.ndarray:
    """SDK integer → float on Piper_buf (÷1e6, full open ≈ 0.1)."""
    g = np.asarray(gripper_int, dtype=np.float64)
    return np.clip(g / GRIPPER_CTRL_ENCODE_SCALE, 0.0, GRIPPER_CTRL_SCALAR_MAX)


def ctrl_scalar_to_sdk_gripper_int(ctrl_scalar: np.ndarray) -> np.ndarray:
    """Piper_buf gripper float → SDK command integer (for GripperCtrl)."""
    s = np.asarray(ctrl_scalar, dtype=np.float64)
    return np.clip(np.round(s * GRIPPER_CTRL_ENCODE_SCALE), GRIPPER_SDK_INT_MIN, GRIPPER_SDK_INT_MAX)


def policy_gripper_binary_to_sdk_int(open01: np.ndarray) -> np.ndarray:
    """Training / policy 0=关, 1=开 → SDK integer."""
    b = np.asarray(open01, dtype=np.float64)
    return np.where(b >= 0.5, GRIPPER_SDK_INT_MAX, GRIPPER_SDK_INT_MIN).astype(np.int64)


def policy_gripper_binary_to_ctrl_scalar(open01: np.ndarray) -> np.ndarray:
    """Training / policy 0/1 → Piper_buf scalar (0 or ≈0.1), for现有 piper_controller."""
    b = np.asarray(open01, dtype=np.float64)
    return np.where(b >= 0.5, GRIPPER_CTRL_SCALAR_MAX, 0.0).astype(np.float32)


def sdk_gripper_int_to_policy_binary(
    gripper_int: np.ndarray,
    *,
    threshold: int = GRIPPER_SDK_OPEN_THRESHOLD,
) -> np.ndarray:
    """SDK integer → policy 0/1."""
    g = np.asarray(gripper_int, dtype=np.float64)
    return (g >= threshold).astype(np.float32)


def sdk_ctrl_scalar_to_policy_binary(
    ctrl_scalar: np.ndarray,
    *,
    threshold: float = GRIPPER_CTRL_SCALAR_MAX * 0.5,
) -> np.ndarray:
    """Piper_buf ÷1e6 scalar → policy 0/1."""
    s = np.asarray(ctrl_scalar, dtype=np.float64)
    return (s >= threshold).astype(np.float32)


def sdk_gripper_int_to_unit(gripper_int: np.ndarray) -> np.ndarray:
    """Alias: normalized [0, 1] (not legacy ÷1e6 unless you need wire format)."""
    return sdk_gripper_int_to_normalized(gripper_int)


def decode_joint7_vector_if_sdk_floats(v: np.ndarray) -> np.ndarray:
    """If rows are CAN integers cast to float, decode joints to radians & gripper to [0, 1].

    Supports shape (7,) or (T, 7). **Do not use** on default Isaac HDF5 (`Keyboard_collection.py`).
    """
    arr = np.asarray(v, dtype=np.float64).copy()
    if arr.ndim == 1:
        arr[:6] /= JOINT_DECODE_DIVISOR
        arr[6] = sdk_gripper_int_to_normalized(arr[6])
        return arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 7:
        arr[:, :6] /= JOINT_DECODE_DIVISOR
        arr[:, 6] = sdk_gripper_int_to_normalized(arr[:, 6])
        return arr.astype(np.float32)
    raise ValueError(f"Expected shape (7,) or (T,7), got {arr.shape}")
