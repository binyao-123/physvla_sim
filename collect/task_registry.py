from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PHYSVLA_SIM_ROOT: Path = Path(__file__).resolve().parent.parent
PHYSVLA_ASSETS_DIR: Path = PHYSVLA_SIM_ROOT / "assets"
# Scene /World/generated payload (from tasks/*/data/): use relpaths into assets/, e.g.


def _tasks_scene_usd(*parts: str) -> str:
    return str((PHYSVLA_SIM_ROOT / "tasks").joinpath(*parts).resolve())


@dataclass(frozen=True)
class TaskCameraSpec:
    name: str
    prim_path: str
    translation: tuple[float, float, float] | None
    rotation_xyz: tuple[float, float, float] | None
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
class TaskPreset:
    task_id: str
    description: str
    usd_path: str
    env_name: str
    dataset_file: str
    language_instruction: str
    sensitivity: float = 4.0
    control_hz: int = 30
    vision_hz: int = 10
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


OPEN_LAPTOP_TASK_ID = "open_laptop_lid"
ADJUST_MONITOR_TASK_ID = "adjust_the_monitor"


@dataclass(frozen=True)
class SharedTeleopPiperCfg:
    articulation_prim_path: str
    root_pos: tuple[float, float, float]
    root_rot: tuple[float, float, float, float]
    joint_pos_entries: tuple[tuple[str, float], ...]

    def joint_pos_dict(self) -> dict[str, float]:
        return dict(self.joint_pos_entries)


# ---------------------------------------------------------------------------
# 公共配置注册
# ---------------------------------------------------------------------------
SHARED_TELEOP_PIPER = SharedTeleopPiperCfg(
    articulation_prim_path="/World/piper_description/root_joint",
    root_pos=(0.0, 0.0, 0.0),
    root_rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos_entries=(
        ("joint1", 0.0),
        ("joint2", 0.3),  # 约 [0, π]
        ("joint3", -0.5),  # 约 [-2.967, 0]
        ("joint4", 0.0),
        ("joint5", 0.5),
        ("joint6", 0.0),
        ("joint7", 0.0),
        ("joint8", 0.0),
    ),
)

SHARED_CAMERA_SPECS: tuple[TaskCameraSpec, ...] = (
    TaskCameraSpec(
        name="main",
        prim_path="/World/Camera",
        translation=None,
        rotation_xyz=None,
        focal_length=None,
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

# ---------------------------------------------------------------------------
# 任务资产注册
# ---------------------------------------------------------------------------

TASK_PRESETS: dict[str, TaskPreset] = {
    OPEN_LAPTOP_TASK_ID: TaskPreset(
        task_id=OPEN_LAPTOP_TASK_ID,
        description="Keyboard teleoperation data collection for opening laptop lid task.",
        usd_path=_tasks_scene_usd("open_laptop", "data", "scene.usd"),
        env_name="OpenLaptopTask",
        dataset_file="./datasets/open_laptop_lid.hdf5",
        language_instruction="open_laptop_lid",
        sensitivity=2.0,
        control_hz=30,
        vision_hz=10,
        camera_width=400,
        camera_height=400,
        camera_sensor_type="camera",
        robot_prim_path=SHARED_TELEOP_PIPER.articulation_prim_path,
        robot_init_root_pos=SHARED_TELEOP_PIPER.root_pos,
        robot_init_root_rot=SHARED_TELEOP_PIPER.root_rot,
        robot_init_joint_pos=SHARED_TELEOP_PIPER.joint_pos_dict(),
        camera_specs=SHARED_CAMERA_SPECS,
        joint_drive_specs=(
            TaskJointDriveSpec(
                prim_path="/World/generated/joints/joint_1",
                damping=50.0,
                stiffness=0.0,
                max_force=15.0,
            ),
        ),
        joint_limit_specs=(
            TaskJointLimitSpec(
                prim_path="/World/generated/joints/joint_1",
                upper_limit=104.0,
            ),
        ),
    ),
    ADJUST_MONITOR_TASK_ID: TaskPreset(
        task_id=ADJUST_MONITOR_TASK_ID,
        description="Keyboard teleoperation data collection for adjusting the monitor (generated URDF hinge).",
        usd_path=_tasks_scene_usd("adjust_the_display", "data", "scene.usd"),
        env_name="AdjustMonitorTask",
        dataset_file="./datasets/adjust_the_monitor.hdf5",
        language_instruction="adjust the monitor",
        sensitivity=2.0,
        control_hz=30,
        vision_hz=10,
        camera_width=400,
        camera_height=400,
        camera_sensor_type="camera",
        robot_prim_path=SHARED_TELEOP_PIPER.articulation_prim_path,
        robot_init_root_pos=SHARED_TELEOP_PIPER.root_pos,
        robot_init_root_rot=SHARED_TELEOP_PIPER.root_rot,
        robot_init_joint_pos=SHARED_TELEOP_PIPER.joint_pos_dict(),
        camera_specs=SHARED_CAMERA_SPECS,
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
    ),
}

DEFAULT_TASK_ID = OPEN_LAPTOP_TASK_ID


def get_task_preset(task_id: str) -> TaskPreset:
    if task_id not in TASK_PRESETS:
        known = ", ".join(sorted(TASK_PRESETS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {known}")
    return TASK_PRESETS[task_id]


def list_task_presets() -> list[TaskPreset]:
    return [TASK_PRESETS[k] for k in sorted(TASK_PRESETS.keys())]
