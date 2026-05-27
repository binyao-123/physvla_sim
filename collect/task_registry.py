from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PHYSVLA_SIM_ROOT: Path = Path(__file__).resolve().parent.parent
PHYSVLA_ASSETS_DIR: Path = PHYSVLA_SIM_ROOT / "assets"


def _tasks_scene_usd(*parts: str) -> str:
    """拼接 tasks/<任务>/.../scene.usd 的绝对路径。"""
    return str((PHYSVLA_SIM_ROOT / "tasks").joinpath(*parts).resolve())


# ---------------------------------------------------------------------------
# 任务级配置片段（相机、场景关节等）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskCameraSpec:
    name: str
    prim_path: str
    # Isaac Sim → Transform → Translate（米）
    translation: tuple[float, float, float] | None
    # Isaac Sim → Transform → Orient（度，与 isaaclab_env_module 一致）
    rotation_xyz: tuple[float, float, float] | None
    # Isaac Sim → Camera → Lens → Focal Length
    focal_length: float | None
    enable_sensor_capture: bool = True


@dataclass(frozen=True)
class TaskJointDriveSpec:
    prim_path: str
    damping: float | None = None
    stiffness: float | None = None
    max_force: float | None = None
    target_position: float | None = None
    target_velocity: float | None = None


@dataclass(frozen=True)
class TaskJointLimitSpec:
    prim_path: str
    lower_limit: float | None = None
    upper_limit: float | None = None


@dataclass(frozen=True)
class TaskJointInitialSpec:
    prim_path: str
    position: float


@dataclass(frozen=True)
class TaskRolloutSuccessSpec:
    """仿真 Rollout 单关节成功条件：角度（度）大于 angle_gt_deg。"""
    joint_prim: str
    angle_gt_deg: float


@dataclass(frozen=True)
class TaskRandomizationSpec:
    """Domain randomization defaults (ArticuBot paper-style). CLI may override enable flags."""

    obj_xy_enable: bool = False
    obj_x_range: tuple[float, float] = (-0.04, 0.04)
    obj_y_range: tuple[float, float] = (-0.03, 0.03)

    obj_yaw_enable: bool = False
    obj_yaw_range_deg: tuple[float, float] = (-12.0, 12.0)

    obj_scale_enable: bool = False
    obj_scale_delta: float = 0.3

    joint_initial_enable: bool = False
    joint_initial_delta_deg: float = 5.0

    camera_main_enable: bool = False
    camera_translation_std: float = 0.02
    camera_rotation_std_deg: float = 3.0


SCENE_ARTICULATION_PRIM_PATH = "/World/generated"


# ---------------------------------------------------------------------------
# 共享：Piper 键盘遥操作（机器人 + 双相机）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedTeleopPiperCfg:
    articulation_prim_path: str
    root_pos: tuple[float, float, float]
    root_rot: tuple[float, float, float, float]
    joint_pos_entries: tuple[tuple[str, float], ...]

    def joint_pos_dict(self) -> dict[str, float]:
        return dict(self.joint_pos_entries)


SHARED_TELEOP_PIPER = SharedTeleopPiperCfg(
    articulation_prim_path="/World/piper_description/root_joint",
    root_pos=(0.0, 0.0, 0.0),
    root_rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos_entries=(
        ("joint1", 0.0),
        ("joint2", 0.0),
        ("joint3", 0.0),
        ("joint4", 0.0),
        ("joint5", 0.0),
        ("joint6", 0.0),
        ("joint7", 0.0),
        ("joint8", 0.0),
    ),
)

SHARED_CAMERA_SPECS: tuple[TaskCameraSpec, ...] = (
    TaskCameraSpec(
        name="main",
        prim_path="/World/Camera",
        translation=(0.4, -0.6, 0.8),
        rotation_xyz=(0.0, -7.0, -90.0),
        focal_length=7.0,
        enable_sensor_capture=True,
    ),
    TaskCameraSpec(
        name="wrist",
        prim_path="/World/piper_description/gripper_base/WristCamera",
        translation=(-0.25, 0.2, -0.88),
        rotation_xyz=(-180.0, -7.0, 90.0),
        focal_length=45.0,
        enable_sensor_capture=True,
    ),
)


def _shared_teleop_kwargs() -> dict:
    """各键盘遥操作任务共用的机器人初始位姿与相机配置。"""

    return {
        "robot_prim_path": SHARED_TELEOP_PIPER.articulation_prim_path,
        "robot_init_root_pos": SHARED_TELEOP_PIPER.root_pos,
        "robot_init_root_rot": SHARED_TELEOP_PIPER.root_rot,
        "robot_init_joint_pos": SHARED_TELEOP_PIPER.joint_pos_dict(),
        "camera_specs": SHARED_CAMERA_SPECS,
    }


# ---------------------------------------------------------------------------
# 任务场景预设（TaskPreset）字段定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskPreset:
    task_id: str
    description: str
    usd_path: str
    env_name: str
    dataset_file: str
    language_instruction: str
    sensitivity: float = 4.0
    # action频率和视觉频率
    control_hz: int = 30
    vision_hz: int = 30
    camera_width: int = 400
    camera_height: int = 400
    camera_sensor_type: str = "camera"
    robot_prim_path: str = "/World/piper_description/root_joint"
    robot_init_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_init_root_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    robot_init_joint_pos: dict[str, float] = field(default_factory=dict)
    camera_specs: tuple[TaskCameraSpec, ...] = field(default_factory=tuple)
    joint_drive_specs: tuple[TaskJointDriveSpec, ...] = field(default_factory=tuple)
    joint_limit_specs: tuple[TaskJointLimitSpec, ...] = field(default_factory=tuple)
    joint_initial_specs: tuple[TaskJointInitialSpec, ...] = field(default_factory=tuple)
    # 每任务单独定义；多 joint 时列多条，全部满足才算 success（AND）
    rollout_success_specs: tuple[TaskRolloutSuccessSpec, ...] = field(default_factory=tuple)
    randomization: TaskRandomizationSpec = field(default_factory=TaskRandomizationSpec)


# ---------------------------------------------------------------------------
# 任务 ID 与 喂给 VLM 的提示词（须与 convert_hdf5_to_lerobot.py --task 一致）
# ---------------------------------------------------------------------------

CLOSE_LAPTOP_TASK_ID = "close_laptop_lid"
CLOSE_LAPTOP_LANGUAGE_INSTRUCTION = "Close the laptop lid until it is fully closed."

ADJUST_MONITOR_TASK_ID = "adjust_the_monitor"
ADJUST_MONITOR_LANGUAGE_INSTRUCTION = "adjust the display."


# ---------------------------------------------------------------------------
# 已注册任务
# ---------------------------------------------------------------------------

TASK_PRESETS: dict[str, TaskPreset] = {
    CLOSE_LAPTOP_TASK_ID: TaskPreset(
        task_id=CLOSE_LAPTOP_TASK_ID,
        description="Keyboard teleoperation collection for closing the laptop lid (open_laptop scene).",
        usd_path=_tasks_scene_usd("open_laptop", "data", "scene.usd"),
        env_name="OpenLaptopTask",
        dataset_file="./datasets/close_laptop_lid.hdf5",
        language_instruction=CLOSE_LAPTOP_LANGUAGE_INSTRUCTION,
        sensitivity=2.0,
        joint_drive_specs=(
            TaskJointDriveSpec(
                prim_path="/World/generated/joints/joint_1",
                damping=60.0,
                stiffness=0.0,
                max_force=10.0,
            ),
        ),
        joint_limit_specs=(
            TaskJointLimitSpec(
                prim_path="/World/generated/joints/joint_1",
                upper_limit=104.0,
            ),
        ),
        joint_initial_specs=(
            TaskJointInitialSpec(
                prim_path="/World/generated/joints/joint_1",
                # 笔记本盖初始开合角，由于数字资产初始化配置有偏差，position为15度对应真实世界90度,104度对应完全关闭笔记本盖
                position=15.0,
            ),
        ),
        rollout_success_specs=(
            TaskRolloutSuccessSpec(
                joint_prim="/World/generated/joints/joint_1",
                angle_gt_deg=98.0,  # 当笔记本闭合角度大于98时，判定任务成功
            ),
        ),
        randomization=TaskRandomizationSpec(
            joint_initial_delta_deg=5.0,
        ),
        **_shared_teleop_kwargs(),
    ),
    ADJUST_MONITOR_TASK_ID: TaskPreset(
        task_id=ADJUST_MONITOR_TASK_ID,
        description="Keyboard teleoperation for adjusting the monitor (generated URDF hinge).",
        usd_path=_tasks_scene_usd("adjust_the_display", "data", "scene.usd"),
        env_name="AdjustMonitorTask",
        dataset_file="./datasets/adjust_the_monitor.hdf5",
        language_instruction=ADJUST_MONITOR_LANGUAGE_INSTRUCTION,
        sensitivity=2.0,
        joint_drive_specs=(
            TaskJointDriveSpec(
                prim_path="/World/generated/joints/joint_1",
                damping=100.0,
                stiffness=0.0,
                max_force=30.0,
            ),
        ),
        joint_limit_specs=(),
        joint_initial_specs=(
            TaskJointInitialSpec(
                prim_path="/World/generated/joints/joint_1",
                position=-20.0,
            ),
        ),
        rollout_success_specs=(
            TaskRolloutSuccessSpec(
                joint_prim="/World/generated/joints/joint_1",
                angle_gt_deg=0.0,   # 显示器角度大于0度时，判定任务成功
            ),
        ),
        **_shared_teleop_kwargs(),
    ),
}


def get_task_preset(task_id: str) -> TaskPreset:
    if task_id not in TASK_PRESETS:
        known = ", ".join(sorted(TASK_PRESETS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {known}")
    return TASK_PRESETS[task_id]


def list_task_presets() -> list[TaskPreset]:
    return [TASK_PRESETS[k] for k in sorted(TASK_PRESETS.keys())]
