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
    push_strategy: str = "yaml_handle"  # yaml_handle | articulation_calibrated (debug probe only)
    close_ratio: float = 0.85
    num_close_steps: int = 80
    # ArticuBot demo gen: θ from init→target at this USD interval (paper ~1°). >0 overrides num_close_steps.
    close_step_deg_usd: float = 1.0
    close_sampled_waypoints: int = 12
    close_final_hold_steps: int = 120
    close_auto_flip_hinge_axis: bool = False
    close_expected_delta_dir_world: tuple[float, float, float] | None = None
    close_ik_substeps: int = 4
    close_clamp_joints: bool = False
    close_push_ee_step_m: float = 0.012
    approach_steps: int = 30
    contact_hold_steps: int = 4
    max_approach_distance_m: float = 0.85
    max_ee_pos_step_m: float = 0.005
    max_joint_step_rad: float = 0.02
    # articulation_calibrated debug probe: touch HDF5 for live contact pose (not used by yaml_handle auto-collect).
    keyboard_reference_hdf5: str | None = None
    keyboard_reference_demo: int = 0
    approach_backoff_m: float = 0.04
    max_servo_steps_per_phase: int = 250
    close_steps_per_waypoint: int | None = None
    approach_clearance_z_m: float = 0.14
    contact_pos_tol_m: float | None = None


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
    fit_range_raw = raw.get("push_contact_joint_fit_range_deg")
    fit_range: tuple[float, float] | None = None
    if fit_range_raw is not None:
        if len(fit_range_raw) != 2:
            raise ValueError("push_contact_joint_fit_range_deg must be [min_deg, max_deg].")
        fit_range = (float(fit_range_raw[0]), float(fit_range_raw[1]))
    arc_points_raw = raw.get("push_contact_joint_arc_points", [])
    arc_points: tuple[tuple[float, float, float, float], ...] = tuple(
        tuple(float(v) for v in point) for point in arc_points_raw
    )
    for point in arc_points:
        if len(point) != 4:
            raise ValueError("push_contact_joint_arc_points entries must be [joint_deg, x, y, z].")
    yaw_anchors_raw = raw.get("yaw_contact_anchors", [])
    yaw_anchors: tuple[tuple[float, float, float, float, float, float, float, float], ...] = tuple(
        tuple(float(v) for v in point) for point in yaw_anchors_raw
    )
    for point in yaw_anchors:
        if len(point) != 8:
            raise ValueError(
                "yaw_contact_anchors entries must be [yaw_deg, x, y, z, qw, qx, qy, qz]."
            )
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
        push_contact_reference_joint_deg=float(raw.get("push_contact_reference_joint_deg", 15.0)),
        push_contact_joint_fit_deg=(
            float(raw["push_contact_joint_fit_deg"])
            if raw.get("push_contact_joint_fit_deg") is not None
            else None
        ),
        push_contact_joint_fit_world=(
            tuple(float(v) for v in raw["push_contact_joint_fit_world"])
            if raw.get("push_contact_joint_fit_world") is not None
            else None
        ),
        push_contact_joint_fit_range_deg=fit_range,
        push_contact_joint_arc_points=arc_points,
        yaw_contact_anchors=yaw_anchors,
    )


def _parse_push(raw: dict[str, Any]) -> PushConfig:
    return PushConfig(
        push_strategy=str(raw.get("push_strategy", "yaml_handle")),
        close_ratio=float(raw.get("close_ratio", 0.85)),
        num_close_steps=int(raw.get("num_close_steps", 80)),
        close_step_deg_usd=float(raw.get("close_step_deg_usd", 1.0)),
        close_sampled_waypoints=int(raw.get("close_sampled_waypoints", 12)),
        close_final_hold_steps=int(raw.get("close_final_hold_steps", 120)),
        close_auto_flip_hinge_axis=bool(raw.get("close_auto_flip_hinge_axis", False)),
        close_expected_delta_dir_world=(
            tuple(float(v) for v in raw["close_expected_delta_dir_world"])
            if raw.get("close_expected_delta_dir_world") is not None
            else None
        ),
        close_ik_substeps=int(raw.get("close_ik_substeps", 4)),
        close_clamp_joints=bool(raw.get("close_clamp_joints", False)),
        close_push_ee_step_m=float(raw.get("close_push_ee_step_m", 0.012)),
        approach_steps=int(raw.get("approach_steps", 30)),
        contact_hold_steps=int(raw.get("contact_hold_steps", 4)),
        max_approach_distance_m=float(raw.get("max_approach_distance_m", 0.85)),
        max_ee_pos_step_m=float(raw.get("max_ee_pos_step_m", 0.005)),
        max_joint_step_rad=float(raw.get("max_joint_step_rad", 0.02)),
        keyboard_reference_hdf5=raw.get("keyboard_reference_hdf5"),
        keyboard_reference_demo=int(raw.get("keyboard_reference_demo", 0)),
        approach_backoff_m=float(raw.get("approach_backoff_m", 0.04)),
        max_servo_steps_per_phase=int(raw.get("max_servo_steps_per_phase", 250)),
        close_steps_per_waypoint=(
            int(raw["close_steps_per_waypoint"])
            if raw.get("close_steps_per_waypoint") is not None
            else None
        ),
        approach_clearance_z_m=float(raw.get("approach_clearance_z_m", 0.14)),
        contact_pos_tol_m=(
            float(raw["contact_pos_tol_m"])
            if raw.get("contact_pos_tol_m") is not None
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
