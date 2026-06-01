"""Shared Isaac Lab env / Piper robot setup for collect, rollout, and auto collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_env_module import (
    CameraPrimSpec,
    EnvironmentModuleConfig,
    JointDrivePrimSpec,
    JointInitialPrimSpec,
    JointLimitPrimSpec,
    SceneRootPrimSpec,
)
from task_registry import SCENE_ARTICULATION_PRIM_PATH, TaskPreset

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

_PIPER_CFG: ArticulationCfg | None = None


def get_piper_cfg() -> ArticulationCfg:
    """Lazy-load Piper articulation cfg (requires AppLauncher before first call)."""
    global _PIPER_CFG
    if _PIPER_CFG is None:
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        _PIPER_CFG = ArticulationCfg(
            prim_path="/World/piper_description",
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "joint[1-6]": 0.0,
                    "joint[7-8]": 0.0,
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint[1-6]"],
                    effort_limit=80.0,
                    stiffness=550.0,
                    damping=40.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["joint[7-8]"],
                    effort_limit=20.0,
                    stiffness=200.0,
                    damping=20.0,
                ),
            },
        )
    return _PIPER_CFG


def __getattr__(name: str):
    if name == "PIPER_CFG":
        return get_piper_cfg()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class ControlLoopTiming:
    control_hz: int
    vision_hz: int
    sim_dt: float
    control_decimation: int
    vision_decimation: int


@dataclass
class RobotHandles:
    arm_joint_ids: slice | list
    gripper_joint_ids: slice | list
    ee_body_id: int


def build_environment_module_config(
    task_preset: TaskPreset,
    *,
    quiet_logging: bool = False,
) -> EnvironmentModuleConfig:
    return EnvironmentModuleConfig(
        quiet_logging=quiet_logging,
        usd_path=task_preset.usd_path,
        camera_width=max(32, int(task_preset.camera_width)),
        camera_height=max(32, int(task_preset.camera_height)),
        camera_sensor_type=task_preset.camera_sensor_type.lower(),
        warmup_render_steps=6,
        camera_specs=[
            CameraPrimSpec(
                name=spec.name,
                prim_path=spec.prim_path,
                translation=spec.translation,
                rotation_xyz=spec.rotation_xyz,
                focal_length=spec.focal_length,
                enable_sensor_capture=spec.enable_sensor_capture,
            )
            for spec in task_preset.camera_specs
        ],
        joint_drive_specs=[
            JointDrivePrimSpec(
                prim_path=spec.prim_path,
                damping=spec.damping,
                stiffness=spec.stiffness,
                max_force=spec.max_force,
                target_position=spec.target_position,
                target_velocity=spec.target_velocity,
            )
            for spec in task_preset.joint_drive_specs
        ],
        joint_limit_specs=[
            JointLimitPrimSpec(
                prim_path=spec.prim_path,
                lower_limit=spec.lower_limit,
                upper_limit=spec.upper_limit,
            )
            for spec in task_preset.joint_limit_specs
        ],
        joint_initial_specs=[
            JointInitialPrimSpec(prim_path=spec.prim_path, position=spec.position)
            for spec in task_preset.joint_initial_specs
        ],
        scene_root_specs=[
            SceneRootPrimSpec(
                prim_path=spec.prim_path,
                translation=spec.translation,
                rotation_xyz=spec.rotation_xyz,
                scale=spec.scale,
            )
            for spec in task_preset.scene_root_specs
        ],
    )


def build_piper_robot_cfg(task_preset: TaskPreset) -> ArticulationCfg:
    from isaaclab.assets import ArticulationCfg

    return get_piper_cfg().replace(prim_path=task_preset.robot_prim_path).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=task_preset.robot_init_root_pos,
            rot=task_preset.robot_init_root_rot,
            joint_pos=dict(task_preset.robot_init_joint_pos),
        )
    )


def compute_control_loop_timing(task_preset: TaskPreset, sim_dt: float) -> ControlLoopTiming:
    control_hz = max(1, int(task_preset.control_hz))
    vision_hz_arg = max(1, int(task_preset.vision_hz))
    vision_hz = min(vision_hz_arg, control_hz)
    if vision_hz_arg > control_hz:
        print(
            f"[WARN] vision_hz ({vision_hz_arg}) > control_hz ({control_hz}). "
            f"Clamp vision_hz to {vision_hz}."
        )
    control_decimation = max(1, int(round((1.0 / control_hz) / sim_dt)))
    vision_decimation = max(1, int(round(control_hz / vision_hz)))
    return ControlLoopTiming(
        control_hz=control_hz,
        vision_hz=vision_hz,
        sim_dt=sim_dt,
        control_decimation=control_decimation,
        vision_decimation=vision_decimation,
    )


def resolve_robot_handles(robot) -> RobotHandles:
    arm_joint_ids = robot.find_joints("joint[1-6]")[0]
    gripper_joint_ids = robot.find_joints("joint[7-8]")[0]
    ee_body_id = robot.find_bodies("link6")[0][0]
    return RobotHandles(
        arm_joint_ids=arm_joint_ids,
        gripper_joint_ids=gripper_joint_ids,
        ee_body_id=ee_body_id,
    )


def build_scene_hinge_cfg(task_preset: TaskPreset) -> ArticulationCfg | None:
    """Scene articulation cfg for hinge physics (shared by rollout and auto collect)."""
    import math

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    if not task_preset.rollout_success_specs:
        return None

    scene_prefix = f"{SCENE_ARTICULATION_PRIM_PATH}/"
    joint_pos = {
        spec.prim_path.rsplit("/", 1)[-1]: math.radians(float(spec.position))
        for spec in task_preset.joint_initial_specs
        if spec.prim_path.startswith(scene_prefix)
    }
    actuators = {
        drive.prim_path.rsplit("/", 1)[-1]: ImplicitActuatorCfg(
            joint_names_expr=[drive.prim_path.rsplit("/", 1)[-1]],
            effort_limit=float(drive.max_force if drive.max_force is not None else 30.0),
            stiffness=float(drive.stiffness if drive.stiffness is not None else 0.0),
            damping=float(drive.damping if drive.damping is not None else 100.0),
        )
        for drive in task_preset.joint_drive_specs
        if drive.prim_path.startswith(scene_prefix)
    }
    if not joint_pos and not actuators:
        return None

    return ArticulationCfg(
        prim_path=SCENE_ARTICULATION_PRIM_PATH,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_pos),
        actuators=actuators,
    )
