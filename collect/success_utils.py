"""Rollout success criteria shared by rollout_sim and auto trajectory collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_env_module import IsaacLabEnvironmentModule
    from task_registry import TaskRolloutSuccessSpec


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
    def _spec_success(spec: TaskRolloutSuccessSpec, deg: float | None) -> bool:
        if deg is None:
            return False
        if spec.angle_gt_deg is not None and not (deg > spec.angle_gt_deg):
            return False
        if spec.angle_lt_deg is not None and not (deg < spec.angle_lt_deg):
            return False
        return True

    success = all(
        _spec_success(spec, deg)
        for spec, deg in ((s, joint_degs[s.joint_prim]) for s in specs)
    )
    return success, joint_degs


def update_peak_joint_degs(
    peak: dict[str, float | None],
    sample: dict[str, float | None],
) -> None:
    for prim, deg in sample.items():
        if deg is None:
            continue
        prev = peak.get(prim)
        peak[prim] = deg if prev is None else max(prev, deg)
