"""Sanity checks for auto trajectory collection (finite state, bounds, RGB)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from env_setup import RobotHandles
    from task_registry import TaskPreset


@dataclass(frozen=True)
class HealthLimits:
    """Conservative bounds; reject NaN/inf and obvious transform blow-ups."""

    scene_joint_deg_min: float = -5.0
    scene_joint_deg_max: float = 115.0
    scene_joint_target_tolerance_deg: float = 18.0
    ee_pos_abs_max_m: float = 5.0
    ee_pos_norm_max_m: float = 8.0
    root_pos_abs_max_m: float = 10.0
    rgb_min_std: float = 1.0
    rgb_uniform_ratio_max: float = 0.995

    @classmethod
    def from_task_preset(cls, task_preset: TaskPreset) -> HealthLimits:
        joint_min = -5.0
        joint_max = 115.0
        joint_tol = 18.0
        for spec in task_preset.joint_limit_specs:
            if spec.lower_limit is not None:
                joint_min = min(joint_min, float(spec.lower_limit) - 5.0)
            if spec.upper_limit is not None:
                joint_max = max(joint_max, float(spec.upper_limit) + 5.0)
        for spec in task_preset.joint_initial_specs:
            joint_tol = max(joint_tol, abs(float(spec.position)) * 0.05 + 5.0)
        return cls(
            scene_joint_deg_min=joint_min,
            scene_joint_deg_max=joint_max,
            scene_joint_target_tolerance_deg=joint_tol,
        )


@dataclass(frozen=True)
class HealthCheckResult:
    ok: bool
    reason: str = ""


class RecordingHealthError(RuntimeError):
    """Raised when a recorded step fails health validation."""


def _finite_reason(name: str, value: torch.Tensor | float) -> str | None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            return f"{name} has non-finite values"
        return None
    if not math.isfinite(float(value)):
        return f"{name} is non-finite ({value})"
    return None


def check_robot_state(
    robot,
    handles: RobotHandles,
    limits: HealthLimits,
) -> HealthCheckResult:
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    ee_pos = robot.data.body_pos_w[:, handles.ee_body_id]
    root_pos = robot.data.root_state_w[:, :3]

    for name, tensor in (
        ("robot_joint_pos", joint_pos),
        ("robot_joint_vel", joint_vel),
        ("robot_eef_pos", ee_pos),
        ("robot_root_pos", root_pos),
    ):
        reason = _finite_reason(name, tensor)
        if reason:
            return HealthCheckResult(False, reason)

    ee = ee_pos[0].detach().cpu()
    if float(ee.abs().max()) > limits.ee_pos_abs_max_m:
        return HealthCheckResult(
            False,
            f"robot_eef_pos out of bounds max_abs={float(ee.abs().max()):.3f}m "
            f"pos={ee.tolist()}",
        )
    if float(torch.linalg.norm(ee)) > limits.ee_pos_norm_max_m:
        return HealthCheckResult(
            False,
            f"robot_eef_pos norm too large ({float(torch.linalg.norm(ee)):.3f}m)",
        )

    root = root_pos[0].detach().cpu()
    if float(root.abs().max()) > limits.root_pos_abs_max_m:
        return HealthCheckResult(
            False,
            f"robot_root_pos out of bounds max_abs={float(root.abs().max()):.3f}m",
        )
    return HealthCheckResult(True)


def check_scene_joint_angle_deg(
    angle_deg: float | None,
    *,
    target_deg: float | None,
    limits: HealthLimits,
) -> HealthCheckResult:
    if angle_deg is None:
        return HealthCheckResult(False, "scene_joint_angle read failed")
    reason = _finite_reason("scene_joint_angle_deg", angle_deg)
    if reason:
        return HealthCheckResult(False, reason)

    angle = float(angle_deg)
    if angle < limits.scene_joint_deg_min or angle > limits.scene_joint_deg_max:
        return HealthCheckResult(
            False,
            f"scene_joint_angle_deg={angle:.3f} outside "
            f"[{limits.scene_joint_deg_min}, {limits.scene_joint_deg_max}]",
        )

    if target_deg is not None and math.isfinite(float(target_deg)):
        delta = abs(angle - float(target_deg))
        if delta > limits.scene_joint_target_tolerance_deg:
            return HealthCheckResult(
                False,
                f"scene_joint_angle_deg={angle:.2f} deviates from target "
                f"{float(target_deg):.2f} by {delta:.2f}° "
                f"(tol={limits.scene_joint_target_tolerance_deg:.1f}°)",
            )
    return HealthCheckResult(True)


def check_rgb_tensor(rgb: torch.Tensor | None, limits: HealthLimits) -> HealthCheckResult:
    if rgb is None:
        return HealthCheckResult(False, "rgb is None")

    reason = _finite_reason("rgb", rgb.float())
    if reason:
        return HealthCheckResult(False, reason)

    x = rgb.detach()
    if x.dtype == torch.uint8:
        flat = x.reshape(-1).float()
    else:
        flat = x.reshape(-1)

    if flat.numel() == 0:
        return HealthCheckResult(False, "rgb is empty")

    std = float(flat.std(unbiased=False))
    if std < limits.rgb_min_std:
        return HealthCheckResult(False, f"rgb nearly uniform (std={std:.3f})")

    mode_count = int(torch.max(torch.bincount(flat.to(torch.int64))))
    uniform_ratio = mode_count / float(flat.numel())
    if uniform_ratio > limits.rgb_uniform_ratio_max:
        return HealthCheckResult(
            False,
            f"rgb degenerate uniform_ratio={uniform_ratio:.4f}",
        )
    return HealthCheckResult(True)


def check_step_payload(
    obs_dict: dict,
    action: torch.Tensor,
    state_dict: dict | None,
    limits: HealthLimits,
) -> HealthCheckResult:
    reason = _finite_reason("action", action)
    if reason:
        return HealthCheckResult(False, reason)

    numeric_obs_keys = (
        "robot_joint_pos",
        "robot_joint_vel",
        "robot_eef_pos",
        "robot_eef_quat",
        "timestamp_sim_sec",
        "timestamp_wall_sec",
    )
    for key in numeric_obs_keys:
        value = obs_dict.get(key)
        if value is None:
            continue
        reason = _finite_reason(f"obs/{key}", value)
        if reason:
            return HealthCheckResult(False, reason)

    if state_dict is not None:
        for key, value in state_dict.items():
            reason = _finite_reason(f"states/{key}", value)
            if reason:
                return HealthCheckResult(False, reason)

    for rgb_key in ("rgb_main", "rgb_wrist"):
        rgb = obs_dict.get(rgb_key)
        if rgb is None:
            return HealthCheckResult(False, f"obs/{rgb_key} missing")
        rgb_result = check_rgb_tensor(rgb, limits)
        if not rgb_result.ok:
            return HealthCheckResult(False, f"{rgb_key}: {rgb_result.reason}")

    ee = obs_dict["robot_eef_pos"]
    ee_vec = ee[0].detach().cpu()
    if float(ee_vec.abs().max()) > limits.ee_pos_abs_max_m:
        return HealthCheckResult(
            False,
            f"obs robot_eef_pos out of bounds pos={ee_vec.tolist()}",
        )
    return HealthCheckResult(True)


def check_episode_data(data: dict, limits: HealthLimits) -> HealthCheckResult:
    """Validate stacked episode tensors before HDF5 export."""

    def _walk(prefix: str, node) -> HealthCheckResult:
        if isinstance(node, dict):
            for key, value in node.items():
                child_prefix = f"{prefix}/{key}" if prefix else key
                result = _walk(child_prefix, value)
                if not result.ok:
                    return result
            return HealthCheckResult(True)
        if isinstance(node, torch.Tensor):
            reason = _finite_reason(prefix, node)
            if reason:
                return HealthCheckResult(False, reason)
            if prefix.endswith("rgb_main") or prefix.endswith("rgb_wrist"):
                return check_rgb_tensor(node, limits)
        return HealthCheckResult(True)

    return _walk("", data)
