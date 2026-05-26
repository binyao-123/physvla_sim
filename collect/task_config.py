"""Load per-task interaction YAML configs for auto trajectory collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from grasp_sampler import HingeSpec, SamplingConfig

_COLLECT_DIR = Path(__file__).resolve().parent
TASK_CONFIGS_DIR = _COLLECT_DIR / "task_configs"


@dataclass
class PushConfig:
    push_strategy: str = "articubot"  # yaml_handle | articubot | keyboard_aligned | articulation_calibrated (legacy)
    close_ratio: float = 0.85
    num_close_steps: int = 80
    # ArticuBot demo gen: θ from init→target at this USD interval (paper ~1°). >0 overrides num_close_steps.
    close_step_deg_usd: float = 1.0
    close_anchor_err_max_m: float = 0.03
    close_ik_substeps: int = 4
    close_recovery_substeps: int = 6
    close_clamp_joints: bool = False
    close_pose_reach_tol_m: float = 0.005
    close_pose_reach_rot_rad: float = 0.12
    close_max_steps_per_waypoint: int = 48
    close_max_iters: int | None = None
    close_push_ee_step_m: float = 0.012
    max_candidates_to_try: int = 10
    approach_steps: int = 30
    contact_hold_steps: int = 4
    max_approach_distance_m: float = 0.85
    max_ee_pos_step_m: float = 0.005
    max_joint_step_rad: float = 0.02
    keyboard_reference_hdf5: str | None = None
    keyboard_reference_demo: int = 0
    keyboard_control_mode: str = "joint_replay"  # joint_replay | ee_servo
    approach_backoff_m: float = 0.04
    close_push_distance_m: float = 0.10
    max_servo_steps_per_phase: int = 250
    close_steps_per_waypoint: int | None = None
    approach_clearance_z_m: float = 0.14
    debug_hardcoded_push: bool = False
    debug_steps_per_waypoint: int = 50
    debug_joint_waypoints: tuple[tuple[float, float, float, float, float, float], ...] = ()
    debug_reference_hdf5: str | None = None
    debug_reference_demo: int = 0
    debug_reference_demos: tuple[int, ...] = ()
    debug_reference_stride: int = 1
    debug_reference_max_frames: int | None = None

    def resolve_debug_reference_demo(self, attempt_index: int) -> int:
        """Pick reference demo index (1-based attempt_index cycles debug_reference_demos)."""
        if self.debug_reference_demos:
            return self.debug_reference_demos[(max(1, attempt_index) - 1) % len(self.debug_reference_demos)]
        return int(self.debug_reference_demo)

    def resolve_keyboard_reference_demo(self, attempt_index: int) -> int:
        if self.debug_reference_demos:
            return self.debug_reference_demos[(max(1, attempt_index) - 1) % len(self.debug_reference_demos)]
        return int(self.keyboard_reference_demo)


@dataclass
class TaskInteractionConfig:
    task_id: str
    interaction_mode: str
    link_prim: str
    joint_prim: str
    joint_type: str = "revolute"
    sampling: SamplingConfig | None = None
    push: PushConfig | None = None
    defaults: dict[str, Any] = field(default_factory=dict)


def _parse_hinge(raw: dict[str, Any]) -> HingeSpec:
    return HingeSpec(
        origin=tuple(float(v) for v in raw.get("origin", (0.0, 0.0, 0.0))),
        axis=tuple(float(v) for v in raw.get("axis", (1.0, 0.0, 0.0))),
    )


def _parse_sampling(raw: dict[str, Any]) -> SamplingConfig:
    assets = raw.get("assets", {})
    hinge_raw = raw.get("hinge", {})
    mesh_origin_raw = assets.get("mesh_origin", [0.0, 0.0, 0.0])
    return SamplingConfig(
        asset_subdir=str(assets.get("subdir", assets.get("asset_subdir", ""))),
        mesh_filename=str(assets.get("mesh", assets.get("mesh_filename", ""))),
        movable_link=str(assets.get("movable_link", "link_1")),
        base_link=str(assets.get("base_link", "link_0")),
        hinge=_parse_hinge(hinge_raw),
        mesh_origin=tuple(float(v) for v in mesh_origin_raw),
        num_fps_points=int(raw.get("num_fps_points", 15)),
        num_yaw_perturbations=int(raw.get("num_yaw_perturbations", 8)),
        max_yaw_perturb_deg=float(raw.get("max_yaw_perturb_deg", 30.0)),
        approach_offset_m=float(raw.get("approach_offset_m", 0.04)),
        contact_offset_m=float(raw.get("contact_offset_m", 0.02)),
        min_hinge_distance_percentile=float(raw.get("min_hinge_distance_percentile", 60.0)),
        horizontal_grasp=bool(raw.get("horizontal_grasp", True)),
        max_surface_points=int(raw.get("max_surface_points", 4000)),
        approach_direction_world=(
            tuple(float(v) for v in raw["approach_direction_world"])
            if raw.get("approach_direction_world") is not None
            else None
        ),
        min_approach_direction_dot=float(raw.get("min_approach_direction_dot", 0.05)),
        use_scene_approach_direction=bool(raw.get("use_scene_approach_direction", True)),
        max_contact_world_y_abs_m=(
            float(raw["max_contact_world_y_abs_m"])
            if raw.get("max_contact_world_y_abs_m") is not None
            else None
        ),
        min_contact_world_x_m=(
            float(raw["min_contact_world_x_m"])
            if raw.get("min_contact_world_x_m") is not None
            else None
        ),
        max_contact_world_x_m=(
            float(raw["max_contact_world_x_m"])
            if raw.get("max_contact_world_x_m") is not None
            else None
        ),
        min_contact_world_z_m=(
            float(raw["min_contact_world_z_m"])
            if raw.get("min_contact_world_z_m") is not None
            else None
        ),
        max_contact_world_z_m=(
            float(raw["max_contact_world_z_m"])
            if raw.get("max_contact_world_z_m") is not None
            else None
        ),
        max_contact_dist_from_link_m=(
            float(raw["max_contact_dist_from_link_m"])
            if raw.get("max_contact_dist_from_link_m") is not None
            else None
        ),
        max_contact_link_local_y_m=(
            float(raw["max_contact_link_local_y_m"])
            if raw.get("max_contact_link_local_y_m") is not None
            else None
        ),
        max_contact_delta_y_from_link_m=(
            float(raw["max_contact_delta_y_from_link_m"])
            if raw.get("max_contact_delta_y_from_link_m") is not None
            else None
        ),
        push_anchor_dist_m=float(raw.get("push_anchor_dist_m", 0.10)),
        push_contact_offset_link=(
            tuple(float(v) for v in raw["push_contact_offset_link"])
            if raw.get("push_contact_offset_link") is not None
            else None
        ),
        contact_quat_link=(
            tuple(float(v) for v in raw["contact_quat_link"])
            if raw.get("contact_quat_link") is not None
            else None
        ),
        use_push_anchor_fallback=bool(raw.get("use_push_anchor_fallback", True)),
        reference_contact_world=(
            tuple(float(v) for v in raw["reference_contact_world"])
            if raw.get("reference_contact_world") is not None
            else None
        ),
        reference_contact_max_dist_m=(
            float(raw["reference_contact_max_dist_m"])
            if raw.get("reference_contact_max_dist_m") is not None
            else None
        ),
    )


def _parse_push(raw: dict[str, Any]) -> PushConfig:
    waypoints_raw = raw.get("debug_joint_waypoints", [])
    waypoints: tuple[tuple[float, float, float, float, float, float], ...] = tuple(
        tuple(float(v) for v in wp) for wp in waypoints_raw
    )
    demos_raw = raw.get("debug_reference_demos")
    debug_reference_demos: tuple[int, ...] = (
        tuple(int(d) for d in demos_raw) if demos_raw is not None else ()
    )
    return PushConfig(
        push_strategy=str(raw.get("push_strategy", "articubot")),
        close_ratio=float(raw.get("close_ratio", 0.85)),
        num_close_steps=int(raw.get("num_close_steps", 80)),
        close_step_deg_usd=float(raw.get("close_step_deg_usd", 1.0)),
        close_anchor_err_max_m=float(raw.get("close_anchor_err_max_m", 0.03)),
        close_ik_substeps=int(raw.get("close_ik_substeps", 4)),
        close_recovery_substeps=int(raw.get("close_recovery_substeps", 6)),
        close_clamp_joints=bool(raw.get("close_clamp_joints", False)),
        close_pose_reach_tol_m=float(raw.get("close_pose_reach_tol_m", 0.005)),
        close_pose_reach_rot_rad=float(raw.get("close_pose_reach_rot_rad", 0.12)),
        close_max_steps_per_waypoint=int(raw.get("close_max_steps_per_waypoint", 48)),
        close_max_iters=(
            int(raw["close_max_iters"])
            if raw.get("close_max_iters") is not None
            else (
                int(raw["close_max_steps"])
                if raw.get("close_max_steps") is not None
                else None
            )
        ),
        close_push_ee_step_m=float(raw.get("close_push_ee_step_m", 0.012)),
        max_candidates_to_try=int(raw.get("max_candidates_to_try", 10)),
        approach_steps=int(raw.get("approach_steps", 30)),
        contact_hold_steps=int(raw.get("contact_hold_steps", 4)),
        max_approach_distance_m=float(raw.get("max_approach_distance_m", 0.85)),
        max_ee_pos_step_m=float(raw.get("max_ee_pos_step_m", 0.005)),
        max_joint_step_rad=float(raw.get("max_joint_step_rad", 0.02)),
        keyboard_reference_hdf5=raw.get("keyboard_reference_hdf5") or raw.get("debug_reference_hdf5"),
        keyboard_reference_demo=int(raw.get("keyboard_reference_demo", raw.get("debug_reference_demo", 0))),
        keyboard_control_mode=str(raw.get("keyboard_control_mode", "joint_replay")),
        approach_backoff_m=float(raw.get("approach_backoff_m", 0.04)),
        close_push_distance_m=float(raw.get("close_push_distance_m", 0.10)),
        max_servo_steps_per_phase=int(raw.get("max_servo_steps_per_phase", 250)),
        close_steps_per_waypoint=(
            int(raw["close_steps_per_waypoint"])
            if raw.get("close_steps_per_waypoint") is not None
            else None
        ),
        approach_clearance_z_m=float(raw.get("approach_clearance_z_m", 0.14)),
        debug_hardcoded_push=bool(raw.get("debug_hardcoded_push", False)),
        debug_steps_per_waypoint=int(raw.get("debug_steps_per_waypoint", 50)),
        debug_joint_waypoints=waypoints,
        debug_reference_hdf5=raw.get("debug_reference_hdf5"),
        debug_reference_demo=int(raw.get("debug_reference_demo", 0)),
        debug_reference_demos=debug_reference_demos,
        debug_reference_stride=max(1, int(raw.get("debug_reference_stride", 1))),
        debug_reference_max_frames=(
            int(raw["debug_reference_max_frames"])
            if raw.get("debug_reference_max_frames") is not None
            else None
        ),
    )


def load_task_interaction_config(task_id: str, config_path: Path | None = None) -> TaskInteractionConfig:
    path = config_path or (TASK_CONFIGS_DIR / f"{task_id}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Task interaction config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw.get("task_id") and raw["task_id"] != task_id:
        raise ValueError(f"Config task_id '{raw['task_id']}' != requested '{task_id}'.")

    interaction = raw.get("interaction", {})
    sampling = _parse_sampling(raw["sampling"]) if "sampling" in raw else None
    push_cfg = _parse_push(raw["push"]) if "push" in raw else None

    return TaskInteractionConfig(
        task_id=task_id,
        interaction_mode=str(raw.get("interaction_mode", "none")),
        link_prim=str(interaction.get("link_prim", "")),
        joint_prim=str(interaction.get("joint_prim", "")),
        joint_type=str(interaction.get("joint_type", "revolute")),
        sampling=sampling,
        push=push_cfg,
        defaults=dict(raw.get("defaults", {})),
    )
