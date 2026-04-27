from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskCameraSpec:
    name: str
    prim_path: str
    translation: tuple[float, float, float]
    rotation_xyz: tuple[float, float, float]
    focal_length: float
    enable_sensor_capture: bool = True


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


OPEN_LAPTOP_TASK_ID = "open_laptop_lid"

TASK_PRESETS: dict[str, TaskPreset] = {
    OPEN_LAPTOP_TASK_ID: TaskPreset(
        task_id=OPEN_LAPTOP_TASK_ID,
        description="Keyboard teleoperation data collection for opening laptop lid task.",
        usd_path="/home/ubuntu/workspace/physvla_sim/tasks/open_laptop/data/scene.usd",
        env_name="OpenLaptopTask",
        dataset_file="./datasets/open_laptop_lid.hdf5",
        language_instruction="open_laptop_lid",
        sensitivity=4.0,
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
            # 参数来自 Isaac Sim Property 面板实测
            TaskCameraSpec(
                name="main",
                prim_path="/World/Camera",
                translation=(0.0, -1.1, 0.3),
                rotation_xyz=(75.0, 0.0, 0.0),
                focal_length=18.15,
                enable_sensor_capture=True,
            ),
            # 腕部相机：挂载于 gripper_base，朝向夹爪操作目标
            # translation/rotation 为初始估计值，需跑起来后根据实际视角微调
            TaskCameraSpec(
                name="wrist",
                prim_path="/World/piper_description/gripper_base/WristCamera",
                translation=(0.0, 0.05, -0.08),
                rotation_xyz=(180.0, 0.0, 90.0),
                focal_length=28.0,
                enable_sensor_capture=True,
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