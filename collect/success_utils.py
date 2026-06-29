"""Rollout success criteria shared by rollout_sim and auto trajectory collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_env_module import IsaacLabEnvironmentModule
    from task_registry import TaskRolloutSuccessSpec


def _spec_satisfied(spec: TaskRolloutSuccessSpec, deg: float | None) -> bool:
    if deg is None:
        return False
    if spec.angle_gt_deg is not None and not (deg > spec.angle_gt_deg):
        return False
    if spec.angle_lt_deg is not None and not (deg < spec.angle_lt_deg):
        return False
    return spec.angle_gt_deg is not None or spec.angle_lt_deg is not None


def evaluate_rollout_success(
    env_module: IsaacLabEnvironmentModule,
    specs: tuple[TaskRolloutSuccessSpec, ...],
) -> tuple[bool, dict[str, float | None]]:
    if not specs:
        return False, {}
    joint_degs = {
        spec.joint_prim: env_module.read_scene_joint_angle_deg(spec.joint_prim)
        for spec in specs
    }
    success = all(
        _spec_satisfied(spec, joint_degs[spec.joint_prim]) for spec in specs
    )
    return success, joint_degs


def evaluate_episode_success(
    env_module: IsaacLabEnvironmentModule,
    specs: tuple[TaskRolloutSuccessSpec, ...],
    *,
    episode_step_limit_hit: bool = False,
) -> tuple[bool, dict[str, float | None]]:
    """Episode success = joint angle specs satisfied AND completed within step limit."""
    angle_ok, joint_degs = evaluate_rollout_success(env_module, specs)
    if episode_step_limit_hit:
        return False, joint_degs
    return angle_ok, joint_degs


def update_peak_joint_degs(
    peak: dict[str, float | None],
    sample: dict[str, float | None],
) -> None:
    for prim, deg in sample.items():
        if deg is None:
            continue
        prev = peak.get(prim)
        peak[prim] = deg if prev is None else max(prev, deg)
