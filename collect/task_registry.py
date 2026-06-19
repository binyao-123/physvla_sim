from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
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
# 任务配置片段（相机、场景关节等）
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
class TaskSceneRootSpec:
    """场景根节点位姿；旋转与相机一致，用 Isaac Transform → Orient 欧拉角 XYZ（度）。"""
    prim_path: str
    translation: tuple[float, float, float]
    rotation_xyz: tuple[float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class TaskRolloutSuccessSpec:
    """仿真 Rollout 单关节成功条件：角度（度）大于 angle_gt_deg。"""
    joint_prim: str
    angle_gt_deg: float


@dataclass(frozen=True)
class TaskRandomizationSpec:
    """Visual / environment domain randomization (see domain_randomization_robotwin2.py).

    Object pose (joint, scene root XY/yaw/scale) is registered per task in this file
    (e.g. CLOSE_LAPTOP_JOINT_INITIAL_* / CLOSE_LAPTOP_SCENE_ROOT_*), not here.
    RobotWin2-style visual fields below are shared defaults for all collection tasks.
    """

    camera_main_enable: bool = False
    camera_translation_std: float = 0.02
    camera_rotation_std_deg: float = 3.0
    camera_main_translation_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    camera_main_focal_length_range: tuple[float, float] | None = None
    lighting_enable: bool = False
    # Isaac viewport lighting menu: stage | off | camera | rig
    lighting_mode: str = "stage"
    # Empty means use lighting_mode; otherwise sample one mode per episode.
    lighting_mode_candidates: tuple[str, ...] = ()
    # Used when lighting_mode == "rig"; Isaac built-ins include:
    # "Colored Lights", "Default", "Grey Studio".
    lighting_rig_name: str = "Default"
    # Empty means use lighting_rig_name; otherwise sample one rig when mode == "rig".
    lighting_rig_candidates: tuple[str, ...] = ()
    lighting_auto_light_rig_on_startup: bool = True
    lighting_import_rig_to_stage: bool = False
    # Empty means all UsdLux lights on the stage.
    lighting_prim_paths: tuple[str, ...] = ()
    lighting_intensity_scale_range: tuple[float, float] = (0.65, 1.05)
    lighting_exposure_delta_range: tuple[float, float] = (-0.6, 0.15)
    lighting_color_temperature_range: tuple[float, float] = (4500.0, 7500.0)
    lighting_enable_color_temperature: bool = True
    # Environment scene DR. Disabled -> use environment_default_asset.
    environment_enable: bool = False
    environment_prim_path: str = "/World/RobotWin2Environment"
    environment_ground_plane_prim_paths: tuple[str, ...] = ()
    # Uniform scale for referenced Grid/Terrain USD scenes (not ground_plane).
    environment_usd_scale: float = 1.0
    environment_default_asset: tuple[str, str | None] | None = None
    environment_asset_candidates: tuple[tuple[str, str | None], ...] = ()
    # Foreground clutter DR: randomly reference small USD props around the task object.
    clutter_enable: bool = False
    clutter_prim_path: str = "/World/RobotWin2Clutter"
    clutter_asset_candidates: tuple[tuple[str, str] | tuple[str, str, float], ...] = ()
    # (zone_name, count, x_offset_range, y_offset_range, z_world)
    clutter_slot_specs: tuple[
        tuple[str, int, tuple[float, float], tuple[float, float], float],
        ...,
    ] = ()
    clutter_yaw_range_deg: tuple[float, float] = (-180.0, 180.0)


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
    camera_width: int = 640
    camera_height: int = 480
    camera_sensor_type: str = "camera"
    robot_prim_path: str = "/World/piper_description/root_joint"
    robot_init_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_init_root_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    robot_init_joint_pos: dict[str, float] = field(default_factory=dict)
    camera_specs: tuple[TaskCameraSpec, ...] = field(default_factory=tuple)
    joint_drive_specs: tuple[TaskJointDriveSpec, ...] = field(default_factory=tuple)
    joint_limit_specs: tuple[TaskJointLimitSpec, ...] = field(default_factory=tuple)
    joint_initial_specs: tuple[TaskJointInitialSpec, ...] = field(default_factory=tuple)
    scene_root_specs: tuple[TaskSceneRootSpec, ...] = field(default_factory=tuple)
    # 每任务单独定义；多 joint 时列多条，全部满足才算 success（AND）
    rollout_success_specs: tuple[TaskRolloutSuccessSpec, ...] = field(default_factory=tuple)
    randomization: TaskRandomizationSpec = field(default_factory=TaskRandomizationSpec)



# ---------------------------------------------------------------------------
# 任务 ID 与 喂给 VLM 的提示词（须与 convert_hdf5_to_lerobot.py --task 一致）
# ---------------------------------------------------------------------------
CLOSE_LAPTOP_TASK_ID = "close_laptop_lid"
CLOSE_LAPTOP_LANGUAGE_INSTRUCTION = "close the laptop lid."

ADJUST_MONITOR_TASK_ID = "adjust_the_monitor"
ADJUST_MONITOR_LANGUAGE_INSTRUCTION = "adjust the display."

# ---------------------------------------------------------------------------
# *******************************随机初始化参数*******************************
# ---------------------------------------------------------------------------

'''RoboTwin2 场景随机化'''
# 视觉 DR：主俯视相机Translate、Focal Length随机化（腕部相机不动，rotation 不随机）
ROBOTWIN2_CAMERA_MAIN_RANDOMIZATION_ENABLE = True  # ENABLE总开关
# Y轴-0.6代表最右边界，Z轴代表上下距离，焦距6.9表示超广角，视野更广
ROBOTWIN2_CAMERA_MAIN_TRANSLATION_RANGES = (
    (0.0, 0.5),
    (-0.6, -0.2),
    (0.7, 0.95),
)
ROBOTWIN2_CAMERA_MAIN_FOCAL_LENGTH_RANGE = (6.9, 8.0)

# 视觉 DR：Isaac UsdLux 光照随机化  
ROBOTWIN2_LIGHTING_RANDOMIZATION_ENABLE = True  # ENABLE总开关
ROBOTWIN2_LIGHTING_MODE = "stage"   
ROBOTWIN2_LIGHTING_MODE_CANDIDATES = ("stage", "camera", "rig")
ROBOTWIN2_LIGHTING_RIG_NAME = "Default"
ROBOTWIN2_LIGHTING_RIG_CANDIDATES = ("Colored Lights", "Default", "Grey Studio")
ROBOTWIN2_LIGHTING_AUTO_RIG_ON_STARTUP = True   # 对应 “Use auto light rig on startup
ROBOTWIN2_LIGHTING_IMPORT_RIG_TO_STAGE = False  # 对应 “Add Current Light Rig to Stage (+)”
# UsdLux: final exposure = baseline + delta; lower delta upper bound reduces overexposure.
ROBOTWIN2_LIGHTING_INTENSITY_SCALE_RANGE = (0.65, 1.05)
ROBOTWIN2_LIGHTING_EXPOSURE_DELTA_RANGE = (-0.6, 0.15)

# 场景纹理DR：ground plane（无场景贴图） 也作为一个候选项参与随机
ROBOTWIN2_ENVIRONMENT_RANDOMIZATION_ENABLE = True     # ENABLE总开关
ROBOTWIN2_ENVIRONMENT_PRIM_PATH = "/World/RobotWin2Environment"
ROBOTWIN2_ENVIRONMENT_GROUND_PLANE_PRIM_PATHS = (
    "/World/defaultGroundPlane",
    "/World/GroundPlane",
    "/World/groundPlane",
    "/World/ground_plane",
    "/World/Ground",
)
ROBOTWIN2_ENVIRONMENT_ASSETS_DIR = PHYSVLA_ASSETS_DIR / "robotwin2_environments"
ROBOTWIN2_ENVIRONMENT_USD_SCALE = 0.35  # 背景缩放系数
ROBOTWIN2_ENVIRONMENT_SCENE_CANDIDATES = (
    ("ground_plane", None),
    ("grid_default", str((ROBOTWIN2_ENVIRONMENT_ASSETS_DIR / "grid_default.usd").resolve())),
    ("gridroom_black", str((ROBOTWIN2_ENVIRONMENT_ASSETS_DIR / "gridroom_black.usd").resolve())),
    # ("gridroom_curved", str((ROBOTWIN2_ENVIRONMENT_ASSETS_DIR / "gridroom_curved.usd").resolve())),
    ("terrain_flat_plane", str((ROBOTWIN2_ENVIRONMENT_ASSETS_DIR / "terrain_flat_plane.usd").resolve())),
)
ROBOTWIN2_ENVIRONMENT_DEFAULT_SCENE = ("ground_plane", None)

# 场景物品DR：每次从15个资产中随机选5个，围绕任务物体放置
ROBOTWIN2_CLUTTER_RANDOMIZATION_ENABLE = True  # ENABLE总开关
ROBOTWIN2_CLUTTER_PRIM_PATH = "/World/RobotWin2Clutter"
ROBOTWIN2_CLUTTER_ASSETS_DIR = PHYSVLA_ASSETS_DIR / "robotwin2_clutter"
ROBOTWIN2_CLUTTER_ASSET_CANDIDATES = (
    # Assets exported with metersPerUnit=0.01 need a 0.01 wrapper scale in the meter scene.
    ("apple", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "Apple.usd").resolve()), 0.01),
    ("alarm_clock_retro", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "AlarmClock_Retro.usd").resolve()), 0.01),
    ("lemon_01", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "Lemon_01.usd").resolve()), 0.01),
    ("utility_jug_a02", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "UtilityJug_A02_PR_V_NVD_01.usd").resolve()), 0.01),
    ("plant_succulent_02", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "Plant_Succulent_02.usd").resolve()), 0.01),
    ("sorting_bowl_yellow", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "sorting_bowl_yellow.usd").resolve())),
    ("sorting_beaker_red", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "sorting_beaker_red.usd").resolve())),
    ("tomato_soup_can", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "005_tomato_soup_can.usd").resolve())),
    ("banana", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "011_banana.usd").resolve())),
    ("bleach_cleanser", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "021_bleach_cleanser.usd").resolve())),
    ("mug_d1", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "SM_Mug_D1.usd").resolve())),
    ("nvidia_cube", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "nvidia_cube.usd").resolve())),
    ("natural_boston_round_bottle", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "NaturalBostonRoundBottle_A01_PR_NVD_01.usd").resolve()), 0.01),
    ("block", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "block.usd").resolve())),
    ("mug_a2", str((ROBOTWIN2_CLUTTER_ASSETS_DIR / "SM_Mug_A2.usd").resolve())),
)
# 坐标以当前任务物体 scene root 的XY为中心；右侧=主俯视相机近侧（Y负方向）
ROBOTWIN2_CLUTTER_SLOT_SPECS = (
    # Split the right side into three non-overlapping bands to avoid clutter overlap.
    # 相对原始基准再远离原点 10cm：right Y 更负，front X 更大，left Y 更大
    ("right", 1, (-0.36, -0.12), (-0.74, -0.38), 0.02),
    ("right", 1, (-0.12, 0.12), (-0.74, -0.38), 0.02),
    ("right", 1, (0.12, 0.36), (-0.74, -0.38), 0.02),
    ("front", 1, (0.38, 0.56), (-0.10, 0.10), 0.02),
    ("left", 1, (-0.18, 0.18), (0.38, 0.56), 0.02),
)
ROBOTWIN2_CLUTTER_YAW_RANGE_DEG = (-180.0, 180.0)

ROBOTWIN2_COMMON_RANDOMIZATION = TaskRandomizationSpec(
    camera_main_enable=ROBOTWIN2_CAMERA_MAIN_RANDOMIZATION_ENABLE,
    camera_rotation_std_deg=0.0,
    camera_main_translation_ranges=ROBOTWIN2_CAMERA_MAIN_TRANSLATION_RANGES,
    camera_main_focal_length_range=ROBOTWIN2_CAMERA_MAIN_FOCAL_LENGTH_RANGE,
    lighting_enable=ROBOTWIN2_LIGHTING_RANDOMIZATION_ENABLE,
    lighting_mode=ROBOTWIN2_LIGHTING_MODE,
    lighting_mode_candidates=ROBOTWIN2_LIGHTING_MODE_CANDIDATES,
    lighting_rig_name=ROBOTWIN2_LIGHTING_RIG_NAME,
    lighting_rig_candidates=ROBOTWIN2_LIGHTING_RIG_CANDIDATES,
    lighting_auto_light_rig_on_startup=ROBOTWIN2_LIGHTING_AUTO_RIG_ON_STARTUP,
    lighting_import_rig_to_stage=ROBOTWIN2_LIGHTING_IMPORT_RIG_TO_STAGE,
    lighting_intensity_scale_range=ROBOTWIN2_LIGHTING_INTENSITY_SCALE_RANGE,
    lighting_exposure_delta_range=ROBOTWIN2_LIGHTING_EXPOSURE_DELTA_RANGE,
    environment_enable=ROBOTWIN2_ENVIRONMENT_RANDOMIZATION_ENABLE,
    environment_prim_path=ROBOTWIN2_ENVIRONMENT_PRIM_PATH,
    environment_ground_plane_prim_paths=ROBOTWIN2_ENVIRONMENT_GROUND_PLANE_PRIM_PATHS,
    environment_usd_scale=ROBOTWIN2_ENVIRONMENT_USD_SCALE,
    environment_default_asset=ROBOTWIN2_ENVIRONMENT_DEFAULT_SCENE,
    environment_asset_candidates=ROBOTWIN2_ENVIRONMENT_SCENE_CANDIDATES,
    clutter_enable=ROBOTWIN2_CLUTTER_RANDOMIZATION_ENABLE,
    clutter_prim_path=ROBOTWIN2_CLUTTER_PRIM_PATH,
    clutter_asset_candidates=ROBOTWIN2_CLUTTER_ASSET_CANDIDATES,
    clutter_slot_specs=ROBOTWIN2_CLUTTER_SLOT_SPECS,
    clutter_yaw_range_deg=ROBOTWIN2_CLUTTER_YAW_RANGE_DEG,
)


'''笔记本采集任务'''
# 基准初始化角度：可配置自由度为5-40度（对应真实65～100度）
CLOSE_LAPTOP_JOINT_INITIAL_BASE_DEG = 22.5
CLOSE_LAPTOP_JOINT_INITIAL_RANDOM_RANGE_DEG = 16.5  # position = base + x

# 基准初始化坐标： X、Y轴偏移量±10 cm
CLOSE_LAPTOP_SCENE_ROOT_BASE_TRANSLATION = (
    0.4423939426468149,
    0.0,
    0.10850162732369013,
)
CLOSE_LAPTOP_SCENE_ROOT_RANDOM_X_RANGE_M = 0.10
CLOSE_LAPTOP_SCENE_ROOT_RANDOM_Y_RANGE_M = 0.15

# 基准初始化yaw角：Z轴偏移量±20度
CLOSE_LAPTOP_SCENE_ROOT_ROTATION_XYZ = (0.0, -10.0, 180.0)
CLOSE_LAPTOP_SCENE_ROOT_RANDOM_YAW_RANGE_DEG = 20.0

# 基准初始化尺寸
CLOSE_LAPTOP_SCENE_ROOT_SCALE = (0.15, 0.15, 0.15)

# 标定笔记本模型基于该坐标平移
CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_TRANSLATION = (
    0.4423939426468149,
    0.0,
    0.10850162732369013,
)
CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ = CLOSE_LAPTOP_SCENE_ROOT_ROTATION_XYZ

def scene_root_translation_delta(
    current: tuple[float, float, float],
    calibration: tuple[float, float, float],
) -> tuple[float, float, float]:
    #当前 scene root 相对标定平移的 Δ，用于把 yaml 世界坐标轨迹映射到新桌面位置。
    return tuple(float(c) - float(b) for c, b in zip(current, calibration))

'''调节显示器采集任务'''
# 基准初始化角度：显示器铰链初始角度 -17度，随机范围 [-22, -12]
ADJUST_MONITOR_JOINT_INITIAL_BASE_DEG = -17.0
ADJUST_MONITOR_JOINT_INITIAL_RANDOM_RANGE_DEG = 5.0  # position = base + x → [-22, -12]°

# 基准初始化坐标，X轴10cm，Y轴±15cm
ADJUST_MONITOR_SCENE_ROOT_BASE_TRANSLATION = (
    0.45,
    0.0,
    0.2326,
)
ADJUST_MONITOR_SCENE_ROOT_RANDOM_X_RANGE_M = 0.10
ADJUST_MONITOR_SCENE_ROOT_RANDOM_Y_RANGE_M = 0.15

# 基准初始化yaw角：Z轴±20度
ADJUST_MONITOR_SCENE_ROOT_ROTATION_XYZ = (0.0, 0.0, 180.0)
ADJUST_MONITOR_SCENE_ROOT_RANDOM_YAW_RANGE_DEG = 20.0

# 基准初始化尺寸
ADJUST_MONITOR_SCENE_ROOT_SCALE = (0.275, 0.275, 0.275)

# 标定显示器模型基于该坐标平移
ADJUST_MONITOR_HANDLE_CALIBRATION_SCENE_TRANSLATION = (
    0.4,
    0.0,
    0.2326,
)
ADJUST_MONITOR_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ = ADJUST_MONITOR_SCENE_ROOT_ROTATION_XYZ


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
                damping=80.0,
                stiffness=0.0,
                max_force=10.0,
            ),
        ),
        joint_limit_specs=(
            TaskJointLimitSpec(
                prim_path="/World/generated/joints/joint_1",
                upper_limit=105.0,
            ),
        ),
        joint_initial_specs=(
            TaskJointInitialSpec(
                prim_path="/World/generated/joints/joint_1",
                # 笔记本盖初始开合角，由于数字资产初始化配置有偏差，position为15度对应真实世界90度,105度对应完全关闭笔记本盖
                position=CLOSE_LAPTOP_JOINT_INITIAL_BASE_DEG,
            ),
        ),
        scene_root_specs=(
            TaskSceneRootSpec(
                # 定义笔记本场景中的默认位姿、尺寸大小
                prim_path=SCENE_ARTICULATION_PRIM_PATH,
                translation=CLOSE_LAPTOP_SCENE_ROOT_BASE_TRANSLATION,
                rotation_xyz=CLOSE_LAPTOP_SCENE_ROOT_ROTATION_XYZ,
                scale=CLOSE_LAPTOP_SCENE_ROOT_SCALE,
            ),
        ),
        rollout_success_specs=(
            TaskRolloutSuccessSpec(
                joint_prim="/World/generated/joints/joint_1",
                angle_gt_deg=100.0,  # 当笔记本闭合角度大于angle_gt_deg时，判定任务成功
            ),
        ),
        randomization=ROBOTWIN2_COMMON_RANDOMIZATION,
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
                damping=1500.0,
                stiffness=0.0,
                max_force=30.0,
            ),
        ),
        joint_limit_specs=(
            TaskJointLimitSpec(
                prim_path="/World/generated/joints/joint_1",
                lower_limit=-22.71,  # USD 默认值 physics:lowerLimit
                # 比 USD 实际上限(3.58°)略大，让 T_rel 规划更长的推动弧线，
                # 确保 EE 持续贴合推面直到 joint 稳定达到成功条件。
                # USD 物理限位仍会阻止 joint 超过 3.58°。
                upper_limit=10.0,
            ),
        ),
        joint_initial_specs=(
            TaskJointInitialSpec(
                prim_path="/World/generated/joints/joint_1",
                position=ADJUST_MONITOR_JOINT_INITIAL_BASE_DEG,
            ),
        ),
        scene_root_specs=(
            TaskSceneRootSpec(
                prim_path=SCENE_ARTICULATION_PRIM_PATH,
                translation=ADJUST_MONITOR_SCENE_ROOT_BASE_TRANSLATION,
                rotation_xyz=ADJUST_MONITOR_SCENE_ROOT_ROTATION_XYZ,
                scale=ADJUST_MONITOR_SCENE_ROOT_SCALE,
            ),
        ),
        rollout_success_specs=(
            TaskRolloutSuccessSpec(
                joint_prim="/World/generated/joints/joint_1",
                angle_gt_deg=-2.0,   # 显示器角度大于-2度时，判定任务成功
            ),
        ),
        randomization=ROBOTWIN2_COMMON_RANDOMIZATION,
        **_shared_teleop_kwargs(),
    ),
}







# ---------------------------------------------------------------------------
# 计算函数
# ---------------------------------------------------------------------------
def sample_close_laptop_joint_initial_deg() -> float:
    """Each call: x ~ U[-range, +range], return base + x (default 5°..40°)."""
    x = random.uniform(
        -CLOSE_LAPTOP_JOINT_INITIAL_RANDOM_RANGE_DEG,
        CLOSE_LAPTOP_JOINT_INITIAL_RANDOM_RANGE_DEG,
    )
    return CLOSE_LAPTOP_JOINT_INITIAL_BASE_DEG + x


def sample_close_laptop_scene_root_translation() -> tuple[float, float, float]:
    """Each call: scene root translation = base + (dx, dy, 0)."""
    base_x, base_y, base_z = CLOSE_LAPTOP_SCENE_ROOT_BASE_TRANSLATION
    dx = random.uniform(
        -CLOSE_LAPTOP_SCENE_ROOT_RANDOM_X_RANGE_M,
        CLOSE_LAPTOP_SCENE_ROOT_RANDOM_X_RANGE_M,
    )
    dy = random.uniform(
        -CLOSE_LAPTOP_SCENE_ROOT_RANDOM_Y_RANGE_M,
        CLOSE_LAPTOP_SCENE_ROOT_RANDOM_Y_RANGE_M,
    )
    return (base_x + dx, base_y + dy, base_z)


def sample_close_laptop_scene_root_rotation_xyz() -> tuple[float, float, float]:
    """Each call: scene root rotation = base + (0, 0, yaw_delta)."""
    rx, ry, rz = CLOSE_LAPTOP_SCENE_ROOT_ROTATION_XYZ
    yaw_delta = random.uniform(
        -CLOSE_LAPTOP_SCENE_ROOT_RANDOM_YAW_RANGE_DEG,
        CLOSE_LAPTOP_SCENE_ROOT_RANDOM_YAW_RANGE_DEG,
    )
    return (rx, ry, rz + yaw_delta)


def sample_adjust_monitor_joint_initial_deg() -> float:
    """Each call: x ~ U[-range, +range], return base + x."""
    x = random.uniform(
        -ADJUST_MONITOR_JOINT_INITIAL_RANDOM_RANGE_DEG,
        ADJUST_MONITOR_JOINT_INITIAL_RANDOM_RANGE_DEG,
    )
    return ADJUST_MONITOR_JOINT_INITIAL_BASE_DEG + x


def sample_adjust_monitor_scene_root_translation() -> tuple[float, float, float]:
    """Each call: scene root translation = base + (dx, dy, 0)."""
    base_x, base_y, base_z = ADJUST_MONITOR_SCENE_ROOT_BASE_TRANSLATION
    dx = random.uniform(
        -ADJUST_MONITOR_SCENE_ROOT_RANDOM_X_RANGE_M,
        ADJUST_MONITOR_SCENE_ROOT_RANDOM_X_RANGE_M,
    )
    dy = random.uniform(
        -ADJUST_MONITOR_SCENE_ROOT_RANDOM_Y_RANGE_M,
        ADJUST_MONITOR_SCENE_ROOT_RANDOM_Y_RANGE_M,
    )
    return (base_x + dx, base_y + dy, base_z)


def sample_adjust_monitor_scene_root_rotation_xyz() -> tuple[float, float, float]:
    """Each call: scene root rotation = base + (0, 0, yaw_delta)."""
    rx, ry, rz = ADJUST_MONITOR_SCENE_ROOT_ROTATION_XYZ
    yaw_delta = random.uniform(
        -ADJUST_MONITOR_SCENE_ROOT_RANDOM_YAW_RANGE_DEG,
        ADJUST_MONITOR_SCENE_ROOT_RANDOM_YAW_RANGE_DEG,
    )
    return (rx, ry, rz + yaw_delta)


def get_task_preset(task_id: str) -> TaskPreset:
    if task_id not in TASK_PRESETS:
        known = ", ".join(sorted(TASK_PRESETS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {known}")
    preset = TASK_PRESETS[task_id]
    if task_id == CLOSE_LAPTOP_TASK_ID:
        angle_deg = sample_close_laptop_joint_initial_deg()
        joint_initial_specs = tuple(
            replace(spec, position=angle_deg) for spec in preset.joint_initial_specs
        )
        scene_translation = sample_close_laptop_scene_root_translation()
        scene_rotation_xyz = sample_close_laptop_scene_root_rotation_xyz()
        scene_root_specs = tuple(
            replace(spec, translation=scene_translation, rotation_xyz=scene_rotation_xyz)
            for spec in preset.scene_root_specs
        )
        return replace(
            preset,
            joint_initial_specs=joint_initial_specs,
            scene_root_specs=scene_root_specs,
        )
    elif task_id == ADJUST_MONITOR_TASK_ID:
        angle_deg = sample_adjust_monitor_joint_initial_deg()
        joint_initial_specs = tuple(
            replace(spec, position=angle_deg) for spec in preset.joint_initial_specs
        )
        scene_translation = sample_adjust_monitor_scene_root_translation()
        scene_rotation_xyz = sample_adjust_monitor_scene_root_rotation_xyz()
        scene_root_specs = tuple(
            replace(spec, translation=scene_translation, rotation_xyz=scene_rotation_xyz)
            for spec in preset.scene_root_specs
        )
        return replace(
            preset,
            joint_initial_specs=joint_initial_specs,
            scene_root_specs=scene_root_specs,
        )
    return preset


def list_task_presets() -> list[TaskPreset]:
    return [TASK_PRESETS[k] for k in sorted(TASK_PRESETS.keys())]


def rotation_xyz_deg_to_wxyz(
    rotation_xyz: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """与 isaaclab_env_module._editor_orient_xyz_to_quatd 相同：依次绕 X、Y、Z（度）。"""
    import math

    def _axis_quat(axis: str, degrees: float) -> tuple[float, float, float, float]:
        half_angle = math.radians(float(degrees)) * 0.5
        cos_v = math.cos(half_angle)
        sin_v = math.sin(half_angle)
        if axis == "x":
            return (cos_v, sin_v, 0.0, 0.0)
        if axis == "y":
            return (cos_v, 0.0, sin_v, 0.0)
        return (cos_v, 0.0, 0.0, sin_v)

    def _quat_mul(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        w0, x0, y0, z0 = left
        w1, x1, y1, z1 = right
        return (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        )

    quat = (1.0, 0.0, 0.0, 0.0)
    for axis, degrees in zip(("x", "y", "z"), rotation_xyz):
        quat = _quat_mul(quat, _axis_quat(axis, degrees))
    return quat


def orient_wxyz_to_rotation_xyz_deg(
    orient_wxyz: tuple[float, float, float, float],
    hint_rotation_xyz: tuple[float, float, float] = (0.0, -10.0, 180.0),
) -> tuple[float, float, float]:
    """从四元数反解欧拉 XYZ（度），与 rotation_xyz_deg_to_wxyz 同约定（离线读 USD 用）。"""
    hx, hy, hz = hint_rotation_xyz
    best = hint_rotation_xyz
    best_err = float("inf")
    for rx in (hx, 0.0, hx - 10.0, hx + 10.0):
        for ry in range(int(hy) - 30, int(hy) + 31):
            for rz in (hz, 0.0, 180.0, -180.0):
                cand = (float(rx), float(ry), float(rz))
                err = sum(
                    (a - b) ** 2
                    for a, b in zip(rotation_xyz_deg_to_wxyz(cand), orient_wxyz)
                )
                if err < best_err:
                    best_err, best = err, cand
    return best


def read_scene_root_from_usd(usd_path: str, prim_path: str = SCENE_ARTICULATION_PRIM_PATH) -> TaskSceneRootSpec:
    """离线读取 scene.usd 中场景根节点的 translate / orient→rotation_xyz / scale。"""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {usd_path}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim '{prim_path}' not found in {usd_path}")

    translate = (0.0, 0.0, 0.0)
    scale = (1.0, 1.0, 1.0)
    orient_wxyz = (1.0, 0.0, 0.0, 0.0)
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        value = op.Get()
        if value is None:
            continue
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            translate = tuple(float(v) for v in value)
        elif op_type == UsdGeom.XformOp.TypeScale:
            scale = tuple(float(v) for v in value)
        elif op_type == UsdGeom.XformOp.TypeOrient and hasattr(value, "GetImaginary"):
            im = value.GetImaginary()
            orient_wxyz = (
                float(im[2]),
                float(value.GetReal()),
                float(im[0]),
                float(im[1]),
            )
    rotation_xyz = orient_wxyz_to_rotation_xyz_deg(orient_wxyz)
    return TaskSceneRootSpec(
        prim_path=prim_path,
        translation=translate,
        rotation_xyz=rotation_xyz,
        scale=scale,
    )


def print_registered_scene_root_specs(task_id: str = CLOSE_LAPTOP_TASK_ID) -> None:
    preset = TASK_PRESETS[task_id]
    if not preset.scene_root_specs:
        print(f"[task_registry] task '{task_id}': no scene_root_specs registered.")
        return
    for spec in preset.scene_root_specs:
        print(f"[task_registry] task '{task_id}' scene_root registered:")
        print(f"  prim_path={spec.prim_path}")
        print(f"  translation={spec.translation}")
        print(f"  rotation_xyz_deg={spec.rotation_xyz}")
        print(f"  scale={spec.scale}")


if __name__ == "__main__":
    print_registered_scene_root_specs(CLOSE_LAPTOP_TASK_ID)
    preset = TASK_PRESETS[CLOSE_LAPTOP_TASK_ID]
    from_usd = read_scene_root_from_usd(preset.usd_path)
    registered = preset.scene_root_specs[0]
    print(
        f"[task_registry] scene.usd on disk: translation={from_usd.translation} "
        f"euler_hint≈{from_usd.rotation_xyz} scale={from_usd.scale}"
    )
    print(
        f"[task_registry] registry uses Isaac panel euler (deg): {registered.rotation_xyz}"
    )
