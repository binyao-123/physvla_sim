from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskCameraSpec:
    name: str
    prim_path: str
    # None means keeping the value authored in the USD scene.
    translation: tuple[float, float, float] | None
    # Isaac Sim Transform panel Orient XYZ values, in degrees.
    # None means keeping the value authored in the USD scene.
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
    # None means keeping the value authored in the USD scene.
    lower_limit: float | None = None
    upper_limit: float | None = None


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
    # robot_prim_path: str = "/World/piper_description"
    robot_prim_path: str = "/World/piper_description/root_joint"
    robot_init_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_init_root_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    robot_init_joint_pos: dict[str, float] = field(default_factory=dict)
    camera_specs: tuple[TaskCameraSpec, ...] = field(default_factory=tuple)
    joint_drive_specs: tuple[TaskJointDriveSpec, ...] = field(default_factory=tuple)
    joint_limit_specs: tuple[TaskJointLimitSpec, ...] = field(default_factory=tuple)


OPEN_LAPTOP_TASK_ID = "open_laptop_lid"

TASK_PRESETS: dict[str, TaskPreset] = {
    OPEN_LAPTOP_TASK_ID: TaskPreset(
        task_id=OPEN_LAPTOP_TASK_ID,
        description="Keyboard teleoperation data collection for opening laptop lid task.",
        usd_path="/home/ubuntu/workspace/physvla_sim/tasks/open_laptop/data/scene.usd",
        env_name="OpenLaptopTask",
        dataset_file="./datasets/open_laptop_lid.hdf5",
        language_instruction="open_laptop_lid",
        sensitivity=2.0,
        control_hz=30,
        vision_hz=10,
        camera_width=400,
        camera_height=400,
        camera_sensor_type="camera",
        # robot_prim_path="/World/piper_description",
        robot_prim_path="/World/piper_description/root_joint",
        robot_init_root_pos=(0.0, 0.0, 0.0),
        robot_init_root_rot=(1.0, 0.0, 0.0, 0.0),
        robot_init_joint_pos={
            "joint1": 0.0,
            "joint2": 0.3,    # [0, 3.14] 范围内
            "joint3": -0.5,   # [-2.967, 0] 范围内
            "joint4": 0.0,
            "joint5": 0.5,
            "joint6": 0.0,
            "joint7": 0.0,
            "joint8": 0.0,
        },
        camera_specs=(
            # 主俯视相机：固定于场景右侧，斜俯视机械臂+笔记本
            # rotation_xyz 对应 Isaac Sim Transform 面板里的 Orient XYZ
            TaskCameraSpec(
                name="main",
                prim_path="/World/Camera",
				translation=(0.1, -0.5, 0.8),
				rotation_xyz=(0.0, 0.0, -90.0),
				focal_length=6.9,
                enable_sensor_capture=True,
            ),
            # 腕部相机：挂载于 gripper_base，朝向夹爪操作目标
            # translation 为局部挂载偏移，rotation_xyz 对应 Transform 面板里的 Orient XYZ
            TaskCameraSpec(
                name="wrist",
                prim_path="/World/piper_description/gripper_base/WristCamera",
				translation=(-0.25, 0.0, -0.88),
				rotation_xyz=(-180.0, -7.0, 90.0),
				focal_length= 45.0,
                enable_sensor_capture=True,
            ),
        ),
        joint_drive_specs=(
            # 笔记本转轴：调节 Angular Drive 阻尼，让机械臂能推动盖子。
            TaskJointDriveSpec(
                prim_path="/World/generated/joints/joint_1",
                damping=0.5,
                stiffness=0.0,
                max_force=20.0,
            ),
        ),
        joint_limit_specs=(
            # 笔记本转轴：保留原始 lower limit，仅限制最高闭合角度。
            TaskJointLimitSpec(
                prim_path="/World/generated/joints/joint_1",
                upper_limit=104.0,
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