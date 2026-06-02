"""Shared HDF5 step recording helpers (aligned with Keyboard_collection)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

POLICY_IMAGE_HEIGHT = 224
POLICY_IMAGE_WIDTH = 224

if TYPE_CHECKING:
    from episode_collector import OfficialEpisodeCollector
    from env_setup import RobotHandles


@dataclass
class RecordingContext:
    device: torch.device
    sim_dt: float
    sim_step_count: int
    control_step_count: int
    vision_decimation: int
    episode_start_wall_time: float
    arm_joint_ids: slice | list
    ee_body_id: int
    last_rgb_main: torch.Tensor | None = None
    last_rgb_wrist: torch.Tensor | None = None
    last_vision_control_step: int = -1
    vision_frame_counter: int = 0


def capture_rgb_if_due(
    env_module,
    ctx: RecordingContext,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    should_capture = (ctx.control_step_count - 1) % ctx.vision_decimation == 0
    if not should_capture:
        return ctx.last_rgb_main, ctx.last_rgb_wrist

    new_wrist = env_module.capture_rgb("wrist", ctx.sim_dt)
    new_main = env_module.capture_rgb("main", ctx.sim_dt)
    if new_wrist is not None and new_main is not None:
        ctx.last_rgb_wrist = new_wrist
        ctx.last_rgb_main = new_main
        ctx.last_vision_control_step = ctx.control_step_count
        ctx.vision_frame_counter += 1
    return ctx.last_rgb_main, ctx.last_rgb_wrist


def resize_rgb_for_policy_storage(rgb: torch.Tensor | None) -> torch.Tensor | None:
    """Directly stretch sensor RGB to the Pi0.5/real-robot 224x224 storage shape."""

    if rgb is None:
        return None

    x = rgb.detach()
    input_was_uint8 = x.dtype == torch.uint8
    if x.dim() != 3:
        raise ValueError(f"Expected RGB tensor with 3 dims, got shape {tuple(x.shape)}.")

    channels_last = x.shape[-1] == 3
    if channels_last:
        x = x.permute(2, 0, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor with 3 channels, got shape {tuple(rgb.shape)}.")

    x = x.unsqueeze(0).float()
    x = F.interpolate(
        x,
        size=(POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if input_was_uint8:
        x = x.round().clamp(0, 255).to(torch.uint8)
    else:
        x = x.to(dtype=rgb.dtype)

    if channels_last:
        x = x.permute(1, 2, 0)
    return x.contiguous()


def build_step_tensors(
    robot,
    ctx: RecordingContext,
    arm_joint_targets: torch.Tensor,
    gripper_open: bool,
    rgb_main: torch.Tensor | None,
    rgb_wrist: torch.Tensor | None,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    device = ctx.device
    gripper_action = torch.tensor([[1.0 if gripper_open else 0.0]], dtype=torch.float32, device=device)
    actual_action = torch.cat([arm_joint_targets, gripper_action], dim=-1)

    vision_is_fresh = bool(ctx.last_vision_control_step == ctx.control_step_count)
    vision_age_steps = (
        ctx.control_step_count - ctx.last_vision_control_step if ctx.last_vision_control_step >= 0 else -1
    )

    timestamp_sim_sec = torch.tensor([ctx.sim_step_count * ctx.sim_dt], dtype=torch.float32, device=device)
    timestamp_wall_sec = torch.tensor(
        [time.perf_counter() - ctx.episode_start_wall_time],
        dtype=torch.float32,
        device=device,
    )

    gripper_open01 = torch.tensor([[1.0 if gripper_open else 0.0]], dtype=torch.float32, device=device)
    obs_joint_pos = torch.cat([robot.data.joint_pos[:, ctx.arm_joint_ids].clone(), gripper_open01], dim=-1)

    obs_dict = {
        "robot_joint_pos": obs_joint_pos,
        "robot_joint_vel": robot.data.joint_vel[:, ctx.arm_joint_ids].clone(),
        "robot_eef_pos": robot.data.body_pos_w[:, ctx.ee_body_id].clone(),
        "robot_eef_quat": robot.data.body_quat_w[:, ctx.ee_body_id].clone(),
        "timestamp_sim_sec": timestamp_sim_sec.clone(),
        "timestamp_wall_sec": timestamp_wall_sec.clone(),
        "rgb_wrist": (
            resize_rgb_for_policy_storage(rgb_wrist).to(device="cpu", dtype=torch.uint8).clone()
            if rgb_wrist is not None
            else None
        ),
        "rgb_main": (
            resize_rgb_for_policy_storage(rgb_main).to(device="cpu", dtype=torch.uint8).clone()
            if rgb_main is not None
            else None
        ),
        "vision_is_fresh": torch.tensor([vision_is_fresh], dtype=torch.bool, device=device),
        "vision_age_steps": torch.tensor([vision_age_steps], dtype=torch.int32, device=device),
        "vision_frame_counter": torch.tensor([ctx.vision_frame_counter], dtype=torch.int32, device=device),
    }

    reward = torch.zeros((1,), device=device, dtype=torch.float32)
    done = torch.zeros((1,), device=device, dtype=torch.bool)
    state_dict = {
        "robot_root_state": robot.data.root_state_w[:, :13].clone(),
        "robot_joint_pos": robot.data.joint_pos.clone(),
        "robot_joint_vel": robot.data.joint_vel.clone(),
    }
    return obs_dict, actual_action, reward, done, state_dict


def record_control_step(
    collector: OfficialEpisodeCollector,
    robot,
    env_module,
    ctx: RecordingContext,
    arm_joint_targets: torch.Tensor,
    gripper_open: bool,
    handles: RobotHandles,
) -> bool:
    rgb_main, rgb_wrist = capture_rgb_if_due(env_module, ctx)
    if rgb_main is None or rgb_wrist is None:
        return False

    obs_dict, action, reward, done, state_dict = build_step_tensors(
        robot, ctx, arm_joint_targets, gripper_open, rgb_main, rgb_wrist
    )
    collector.add_step(obs_dict, action, reward, done, state_dict)
    return True
