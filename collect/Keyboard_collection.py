import argparse
import os
import re
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import torch
import h5py
from isaaclab.app import AppLauncher

SERVER_DIR = Path(__file__).resolve().parents[1] / "Server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from isaaclab_env_module import (
    CameraPrimSpec,
    EnvironmentModuleConfig,
    IsaacLabEnvironmentModule,
    JointDrivePrimSpec,
    JointInitialPrimSpec,
    JointLimitPrimSpec,
    apply_camera_launch_workarounds,
)
from task_registry import DEFAULT_TASK_ID, get_task_preset, list_task_presets, PHYSVLA_ASSETS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=5)
parser.add_argument("--task_id", type=str, default=DEFAULT_TASK_ID)
parser.add_argument("--list_tasks", action="store_true", help="List all available task presets and exit.")
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="Override TaskPreset.usd_path if your scene is elsewhere (absolute path or cwd-relative).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.list_tasks:
    print("[INFO] Available task presets:")
    for preset in list_task_presets():
        print(f"  - {preset.task_id}: {preset.description}")
    raise SystemExit(0)

task_preset = get_task_preset(args_cli.task_id)
if args_cli.usd_path:
    p = Path(args_cli.usd_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    task_preset = replace(task_preset, usd_path=str(p))
    print(f"[INFO] usd_path override: {task_preset.usd_path}")

args_cli = apply_camera_launch_workarounds(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. 导入 Isaac Lab 模块 (必须在 App 启动后) ────────────────────
import omni.usd
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

# IsaacLab 2.3 使用通用数据集接口导出 HDF5
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

PIPER_CFG = ArticulationCfg(
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
            effort_limit=50.0,
            stiffness=400.0,
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


class OfficialEpisodeCollector:
    def __init__(self, dataset_file: str, env_name: str, num_demos: int = 0):
        self.dataset_file = self._make_session_dataset_file(dataset_file)
        self.env_name = env_name
        self.num_demos = num_demos

        output_dir = os.path.dirname(self.dataset_file) or "."
        output_name = os.path.splitext(os.path.basename(self.dataset_file))[0]
        os.makedirs(output_dir, exist_ok=True)

        self._dataset_handler = HDF5DatasetFileHandler()
        dataset_stem_path = os.path.join(output_dir, output_name)
        dataset_hdf5_path = f"{dataset_stem_path}.hdf5"
        if os.path.exists(dataset_hdf5_path):
            self._dataset_handler._hdf5_file_stream = h5py.File(dataset_hdf5_path, "a")
            self._dataset_handler._hdf5_data_group = self._dataset_handler._hdf5_file_stream.require_group("data")

            demo_ids: list[int] = []
            for name in self._dataset_handler._hdf5_data_group.keys():
                m = re.fullmatch(r"demo_(\d+)", name)
                if m is not None:
                    demo_ids.append(int(m.group(1)))

            next_demo_id = (max(demo_ids) + 1) if demo_ids else 0
            self._dataset_handler._demo_count = next_demo_id
            self.existing_episode_count = len(demo_ids)

            try:
                existing_env_name = self._dataset_handler.get_env_name()
            except Exception:
                existing_env_name = None

            if existing_env_name is None:
                self._dataset_handler.set_env_name(self.env_name)
                existing_env_name = self.env_name

            if existing_env_name != self.env_name:
                print(
                    f"[WARN] Existing dataset env_name='{existing_env_name}' != requested env_name='{self.env_name}'. "
                    "Appending anyway."
                )

            print(
                f"[INFO] Append mode enabled: {dataset_hdf5_path} "
                f"(existing episodes: {self.existing_episode_count}, next demo id: {next_demo_id})"
            )
        else:
            self._dataset_handler.create(dataset_stem_path, env_name=self.env_name)
            self.existing_episode_count = 0
            print(f"[INFO] Create mode enabled: {dataset_hdf5_path}")

        self._episode = EpisodeData()
        self.exported_successful_episode_count = 0
        self.exported_failed_episode_count = 0

    @staticmethod
    def _make_session_dataset_file(dataset_file: str) -> str:
        output_dir = os.path.dirname(dataset_file) or "."
        output_name = os.path.splitext(os.path.basename(dataset_file))[0]
        session_dir = os.path.join(output_dir, output_name)
        utc8 = timezone(timedelta(hours=8))
        timestamp = datetime.now(utc8).strftime("%Y%m%d_%H%M%S")
        return os.path.join(session_dir, f"{output_name}_{timestamp}.hdf5")

    def _prepare_episode_for_export(self):
        if hasattr(self._episode, "pre_export") and callable(self._episode.pre_export):
            self._episode.pre_export()
            return

        def _stack_leaf_lists(node):
            for key, value in node.items():
                if isinstance(value, list):
                    if len(value) > 0 and torch.is_tensor(value[0]):
                        node[key] = torch.stack(value)
                elif isinstance(value, dict):
                    _stack_leaf_lists(value)

        _stack_leaf_lists(self._episode.data)

    def reset_episode(self):
        self._episode = EpisodeData()

    def has_data(self) -> bool:
        return not self._episode.is_empty()

    def set_initial_state(self, initial_state: dict[str, torch.Tensor]):
        for key, value in initial_state.items():
            self._episode.add(f"initial_state/{key}", value.detach().clone())

    def add_step(
        self,
        obs_dict: dict[str, torch.Tensor],
        actions: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        state_dict: dict[str, torch.Tensor] | None = None,
    ):
        self._episode.add("actions", actions.detach().clone())

        for key, value in obs_dict.items():
            self._episode.add(f"obs/{key}", value.detach().clone())

        if state_dict is not None:
            for key, value in state_dict.items():
                self._episode.add(f"states/{key}", value.detach().clone())

        self._episode.add("rewards", reward.detach().clone())
        self._episode.add("dones", done.detach().clone())

    def export_episode(self, success: bool) -> bool:
        if self._episode.is_empty():
            return False

        self._episode.success = success
        self._prepare_episode_for_export()
        self._dataset_handler.write_episode(self._episode)
        self._dataset_handler.flush()

        if success:
            self.exported_successful_episode_count += 1
        else:
            self.exported_failed_episode_count += 1

        self.reset_episode()
        return True

    def close(self):
        self._dataset_handler.close()


# 配置参数
USD_PATH = task_preset.usd_path
_SENS = float(task_preset.sensitivity)
POS_SENSITIVITY = 0.002 * _SENS
ROT_SENSITIVITY = 0.01 * _SENS
CAMERA_WIDTH = max(32, int(task_preset.camera_width))
CAMERA_HEIGHT = max(32, int(task_preset.camera_height))
CAMERA_PATHS = {spec.name: spec.prim_path for spec in task_preset.camera_specs}

for required_camera_name in ("main", "wrist"):
    if required_camera_name not in CAMERA_PATHS:
        raise ValueError(
            f"Task preset '{task_preset.task_id}' misses required camera '{required_camera_name}'."
        )
MAIN_CAM_PATH = CAMERA_PATHS["main"]
WRIST_CAM_PATH = CAMERA_PATHS["wrist"]

print(f"[INFO] Selected task preset: {task_preset.task_id}")

_scene_usd = Path(task_preset.usd_path)
if not _scene_usd.is_file():
    raise FileNotFoundError(
        f"Scene USD missing: {_scene_usd}\n"
        "See collect/task_registry.py (TaskPreset.usd_path) for the expected path; "
        "or override with --usd_path /path/to/scene.usd"
    )

print(
    f"[INFO] Assets root {PHYSVLA_ASSETS_DIR.resolve()}: reference payloads relative to this tree in scene.usd."
)

# ── 3. 场景搭建 ───────────────────
env_cfg = EnvironmentModuleConfig(
    usd_path=USD_PATH,
    camera_width=CAMERA_WIDTH,
    camera_height=CAMERA_HEIGHT,
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
)

env_module = IsaacLabEnvironmentModule(env_cfg)
sim = env_module.create_simulation(dt=1 / 60.0, render_interval=4)

# ── 4. 资产接管 (Articulation) ───────────────────────────────────
robot_cfg = PIPER_CFG.replace(prim_path=task_preset.robot_prim_path)
robot_cfg = robot_cfg.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=task_preset.robot_init_root_pos,
        rot=task_preset.robot_init_root_rot,
        joint_pos=dict(task_preset.robot_init_joint_pos),
    )
)

robot = env_module.create_robot(robot_cfg)
env_module.initialize_robot_home_pose()

device = sim.device

# ── 5. 控制器设置 (DifferentialIKController) ───────────────────────
diff_ik_cfg = DifferentialIKControllerCfg(
    command_type="pose",
    ik_method="dls",
    ik_params={"lambda_val": 0.1},
)
diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)

# Piper 关节索引
# joint1~joint6：手臂 6 轴；joint7/joint8：夹爪
arm_joint_ids = robot.find_joints("joint[1-6]")[0]
gripper_joint_ids = robot.find_joints("joint[7-8]")[0]
# 末端执行器 body：link6（joint6 输出端）
ee_body_id = robot.find_bodies("link6")[0][0]

# ── 6. 键盘遥操设置 ────────────────────────────────────────────────
teleop_interface = Se3Keyboard(
    Se3KeyboardCfg(pos_sensitivity=POS_SENSITIVITY, rot_sensitivity=ROT_SENSITIVITY)
)

should_reset = False
should_skip = False
should_exit = False


def reset_recording_instance():
    global should_reset
    should_reset = True


def skip_recording_instance():
    global should_skip
    should_skip = True


def request_exit():
    global should_exit
    should_exit = True
    print("\n[INFO] Exit requested by keyboard.")


teleop_interface.add_callback("R", reset_recording_instance)
teleop_interface.add_callback("SPACE", skip_recording_instance)
teleop_interface.add_callback("P", request_exit)
teleop_interface.reset()
teleop_interface._close_gripper = True

# ── 7. 初始化目标位姿 ──────────────────────────────────────────────
target_pos = robot.data.body_pos_w[:, ee_body_id].clone()
target_quat = robot.data.body_quat_w[:, ee_body_id].clone()

# 夹爪开合目标位置（单位：m，对应 Piper joint7/joint8 平移关节）
gripper_open_target = torch.tensor([[0.035, -0.035]], dtype=torch.float32, device=device)
gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)

print("\n=== Isaac Lab Teleoperation Ready ===")
print('[INFO] Hotkeys: "R" save current demo and next, "SPACE" skip current demo and next, "P" exit.')

# ── 相机视口与传感器初始化 ────────────────────────────────────────
import omni.kit.viewport.utility as vp_utils
from pxr import Sdf
import omni.ui as ui
import omni.kit.app
import asyncio


def _set_viewport_resolution(viewport_window, width: int, height: int):
    if viewport_window is None:
        return
    vp_api = getattr(viewport_window, "viewport_api", None)
    if vp_api is None:
        return
    try:
        if hasattr(vp_api, "set_texture_resolution"):
            vp_api.set_texture_resolution((width, height))
        elif hasattr(vp_api, "resolution"):
            vp_api.resolution = (width, height)
    except Exception as e:
        win_title = getattr(viewport_window, "title", "unknown")
        print(f"[WARN] Failed to set viewport resolution for {win_title}: {e}")


# 1) 创建相机 prim（main 固定相机 + wrist 腕部相机）
env_module.define_camera_prims()

# 2) 创建视口窗口：左侧腕部，右侧主俯视
wrist_vp_window = vp_utils.create_viewport_window(
    "Wrist View",
    camera_path=WRIST_CAM_PATH,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
)

# 主视口切到主俯视相机
main_vp = vp_utils.get_viewport_from_window_name("Viewport")
if main_vp is not None:
    main_vp.camera_path = Sdf.Path(MAIN_CAM_PATH)

_set_viewport_resolution(wrist_vp_window, CAMERA_WIDTH, CAMERA_HEIGHT)

# 3) 初始化 IsaacLab 传感器相机（采集 RGB 图像）
sensor_cameras = {}
sensor_type = task_preset.camera_sensor_type.lower()
print(f"[INFO] RGB backend: isaaclab.{sensor_type}")
try:
    sensor_cameras = env_module.create_sensor_cameras()
except Exception as e:
    raise RuntimeError(
        f"Failed to initialize isaaclab {sensor_type} sensor pipeline: {e}. "
        "Please verify --enable_cameras/offscreen settings and camera prim path."
    ) from e

for cam_name in ("main", "wrist"):
    if cam_name in sensor_cameras:
        print(f"[INFO] Sensor camera ready: {cam_name} ({sensor_type})")

# 二次 reset 后刷新末端目标位姿
target_pos = robot.data.body_pos_w[:, ee_body_id].clone()
target_quat = robot.data.body_quat_w[:, ee_body_id].clone()

# 4) UI 布局：左侧腕部视图，右侧主俯视（Viewport）
async def dock_window():
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    main_win = ui.Workspace.get_window("Viewport")
    wrist_win = ui.Workspace.get_window("Wrist View")

    if main_win and wrist_win:
        wrist_win.dock_in(main_win, ui.DockPosition.LEFT, ratio=0.4)


dock_task = asyncio.ensure_future(dock_window())

# ── 数据收集器初始化 ──────────────────────────────────────────────
print("\n[INFO] Initializing Data Collector...")
collector = OfficialEpisodeCollector(
    dataset_file=task_preset.dataset_file,
    env_name=task_preset.env_name,
    num_demos=args_cli.num_demos,
)

instruction_text = (task_preset.language_instruction or "open laptop lid").strip()
if not instruction_text:
    instruction_text = "open laptop lid"
instruction_bytes = instruction_text.encode("utf-8")
language_instruction_tensor = torch.tensor(
    list(instruction_bytes), dtype=torch.uint8, device=device
).unsqueeze(0)
language_instruction_length = torch.tensor(
    [len(instruction_bytes)], dtype=torch.int32, device=device
)


def capture_initial_state_for_episode():
    initial_state = {
        "robot_root_state": robot.data.root_state_w[:, :13].clone(),
        "robot_joint_pos": robot.data.joint_pos.clone(),
        "robot_joint_vel": robot.data.joint_vel.clone(),
        "target_eef_pos": target_pos.clone(),
        "target_eef_quat": target_quat.clone(),
        "language_instruction_utf8": language_instruction_tensor.clone(),
        "language_instruction_length": language_instruction_length.clone(),
    }
    collector.set_initial_state(initial_state)


capture_initial_state_for_episode()

last_obs_dict = None
last_action = torch.zeros((1, 8), dtype=torch.float32, device=device)
last_reward = torch.zeros((1,), dtype=torch.float32, device=device)
last_state_dict = None
completed_demo_slots = 0

control_hz = max(1, int(task_preset.control_hz))
vision_hz_arg = max(1, int(task_preset.vision_hz))
vision_hz = min(vision_hz_arg, control_hz)
if vision_hz_arg > control_hz:
    print(
        f"[WARN] vision_hz ({vision_hz_arg}) > control_hz ({control_hz}). "
        f"Clamp vision_hz to {vision_hz}."
    )
sim_dt = float(sim.cfg.dt)
control_decimation = max(1, int(round((1.0 / control_hz) / sim_dt)))
vision_decimation = max(1, int(round(control_hz / vision_hz)))
sim_step_count = 0
control_step_count = 0
episode_start_wall_time = time.perf_counter()
last_rgb_wrist = None
last_rgb_main = None
last_vision_control_step = -1
vision_frame_counter = 0
print(f"[INFO] Language instruction: {instruction_text}")
print(
    f"[INFO] Loop decimation: sim_dt={sim_dt:.4f}s, "
    f"control_every={control_decimation} sim steps (~{control_hz}Hz), "
    f"vision_every={vision_decimation} control steps (~{vision_hz}Hz)."
)

# ── 8. 主循环 ──────────────────────────────────────────────────────
try:
    while simulation_app.is_running() and not should_exit:
        if should_reset or should_skip:
            finalize_mode = "save" if should_reset else "skip"

            if finalize_mode == "save":
                if last_obs_dict is not None and last_state_dict is not None:
                    done_flag = torch.ones((1,), device=device, dtype=torch.bool)
                    collector.add_step(last_obs_dict, last_action, last_reward, done_flag, last_state_dict)
                    if collector.export_episode(success=True):
                        print(
                            f"\n[INFO] Saved demo slot {completed_demo_slots + 1}"
                            f"/{collector.num_demos if collector.num_demos > 0 else '?'}"
                        )
                else:
                    print("\n[INFO] No trajectory data yet. This slot will be counted and skipped.")
            else:
                print(
                    f"\n[INFO] Skipped demo slot {completed_demo_slots + 1}"
                    f"/{collector.num_demos if collector.num_demos > 0 else '?'} (discard current trajectory)."
                )
                collector.reset_episode()
                last_obs_dict = None
                last_state_dict = None

            completed_demo_slots += 1

            if collector.num_demos > 0 and completed_demo_slots >= collector.num_demos:
                print("[INFO] Data Collection Completed by demo slots. Exiting...")
                break

            sim.reset()
            robot.reset()
            if sensor_cameras:
                for sensor in sensor_cameras.values():
                    sensor.reset()
            env_module.apply_joint_initial_overrides()
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.set_joint_position_target(gripper_close_target, joint_ids=gripper_joint_ids)
            robot.write_data_to_sim()

            sim.step()
            robot.update(sim.cfg.dt)

            env_module.define_camera_prims()

            target_pos = robot.data.body_pos_w[:, ee_body_id].clone()
            target_quat = robot.data.body_quat_w[:, ee_body_id].clone()
            teleop_interface.reset()
            teleop_interface._close_gripper = True
            diff_ik_controller.reset()

            collector.reset_episode()
            episode_start_wall_time = time.perf_counter()
            capture_initial_state_for_episode()
            last_obs_dict = None
            last_state_dict = None
            last_rgb_wrist = None
            last_rgb_main = None
            last_vision_control_step = -1
            vision_frame_counter = 0
            should_reset = False
            should_skip = False

        if sim_step_count % control_decimation == 0:
            control_step_count += 1

            with torch.inference_mode():
                keyboard_command = teleop_interface.advance()
                if not torch.is_tensor(keyboard_command):
                    keyboard_command = torch.tensor(keyboard_command, dtype=torch.float32, device=device)
                else:
                    keyboard_command = keyboard_command.to(device=device, dtype=torch.float32)

                delta_pos = keyboard_command[:3].unsqueeze(0)
                delta_rot = keyboard_command[3:6].unsqueeze(0)
                gripper_cmd = bool(keyboard_command[6] > 0.0)

                target_pos += delta_pos

                if torch.norm(delta_rot) > 1e-6:
                    angle = torch.norm(delta_rot, dim=-1)
                    axis = delta_rot / angle
                    delta_quat = math_utils.quat_from_angle_axis(angle, axis)
                    target_quat = math_utils.quat_mul(target_quat, delta_quat)
                    target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

                diff_ik_controller.set_command(torch.cat([target_pos, target_quat], dim=-1))

                jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_id, :, arm_joint_ids]

                arm_joint_targets = diff_ik_controller.compute(
                    ee_pos=robot.data.body_pos_w[:, ee_body_id],
                    ee_quat=robot.data.body_quat_w[:, ee_body_id],
                    jacobian=jacobian,
                    joint_pos=robot.data.joint_pos[:, arm_joint_ids],
                )

                gripper_targets = gripper_open_target if gripper_cmd else gripper_close_target
                gripper_action = torch.tensor(
                    [[1.0 if gripper_cmd else 0.0]], dtype=torch.float32, device=device
                )

                #  记录动作为 7-DoF：6 维手臂关节角 + 1 维夹爪开合(0=关/1=开)
                actual_action = torch.cat([arm_joint_targets, gripper_action], dim=-1)

                robot.set_joint_position_target(arm_joint_targets, joint_ids=arm_joint_ids)
                robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
                robot.write_data_to_sim()

                # ── 图像采集（main 主俯视 + wrist 腕部）────────────────
                should_capture_vision = ((control_step_count - 1) % vision_decimation == 0)
                if should_capture_vision:
                    new_rgb_wrist = env_module.capture_rgb("wrist", sim_dt)
                    new_rgb_main = env_module.capture_rgb("main", sim_dt)
                    if new_rgb_wrist is not None and new_rgb_main is not None:
                        last_rgb_wrist = new_rgb_wrist
                        last_rgb_main = new_rgb_main
                        last_vision_control_step = control_step_count
                        vision_frame_counter += 1

                rgb_wrist = last_rgb_wrist
                rgb_main = last_rgb_main

                has_vision = (rgb_wrist is not None) and (rgb_main is not None)
                vision_is_fresh = bool(last_vision_control_step == control_step_count)
                vision_age_steps = (
                    control_step_count - last_vision_control_step if last_vision_control_step >= 0 else -1
                )
                can_record_step = has_vision

                timestamp_sim_sec = torch.tensor(
                    [sim_step_count * sim_dt], dtype=torch.float32, device=device
                )
                timestamp_wall_sec = torch.tensor(
                    [time.perf_counter() - episode_start_wall_time],
                    dtype=torch.float32,
                    device=device,
                )

                # observation.state：7 维 = 6 维手臂关节角 + 1 维夹爪状态
                gripper_state = robot.data.joint_pos[:, gripper_joint_ids[0:1]].clone()
                obs_joint_pos = torch.cat([robot.data.joint_pos[:, arm_joint_ids].clone(), gripper_state], dim=-1)

                obs_dict = {
                    "robot_joint_pos": obs_joint_pos,
                    "robot_joint_vel": robot.data.joint_vel[:, arm_joint_ids].clone(),
                    "robot_eef_pos": robot.data.body_pos_w[:, ee_body_id].clone(),
                    "robot_eef_quat": robot.data.body_quat_w[:, ee_body_id].clone(),
                    "timestamp_sim_sec": timestamp_sim_sec.clone(),
                    "timestamp_wall_sec": timestamp_wall_sec.clone(),
                    # 图像以 CPU uint8 存储，对齐 π0.5 双相机输入（腕部 + 前向）
                    "rgb_wrist": rgb_wrist.detach().to(device="cpu", dtype=torch.uint8).clone() if rgb_wrist is not None else None,
                    "rgb_main": rgb_main.detach().to(device="cpu", dtype=torch.uint8).clone() if rgb_main is not None else None,
                    "vision_is_fresh": torch.tensor([vision_is_fresh], dtype=torch.bool, device=device),
                    "vision_age_steps": torch.tensor([vision_age_steps], dtype=torch.int32, device=device),
                    "vision_frame_counter": torch.tensor([vision_frame_counter], dtype=torch.int32, device=device),
                }

                reward = torch.zeros((1,), device=device, dtype=torch.float32)
                done = torch.zeros((1,), device=device, dtype=torch.bool)

                state_dict = {
                    "robot_root_state": robot.data.root_state_w[:, :13].clone(),
                    "robot_joint_pos": robot.data.joint_pos.clone(),
                    "robot_joint_vel": robot.data.joint_vel.clone(),
                }

                if not can_record_step:
                    print("[WARN] RGB is unavailable yet. Skip dataset write on this control step.")
                else:
                    collector.add_step(obs_dict, actual_action, reward, done, state_dict)

                    last_obs_dict = {k: v.clone() if torch.is_tensor(v) else v for k, v in obs_dict.items()}
                    last_action = actual_action.clone()
                    last_reward = reward.clone()
                    last_state_dict = {k: v.clone() for k, v in state_dict.items()}

        sim.step(render=True)
        robot.update(sim.cfg.dt)
        sim_step_count += 1

except KeyboardInterrupt:
    should_exit = True
    print("\n[INFO] KeyboardInterrupt received. Exiting gracefully...")

if should_exit:
    print("[INFO] Exiting by user request.")

if dock_task is not None and not dock_task.done():
    dock_task.cancel()

if collector.has_data() and last_obs_dict is not None and last_state_dict is not None:
    try:
        done_flag = torch.ones((1,), device=device, dtype=torch.bool)
        collector.add_step(last_obs_dict, last_action, last_reward, done_flag, last_state_dict)
        if collector.export_episode(success=False):
            print("[INFO] Exported in-progress episode before shutdown (marked as failed).")
    except Exception as e:
        print(f"[WARN] Failed to export in-progress episode on exit: {e}")

collector.close()
print("[INFO] Dataset saved and closed safely.")
print("[INFO] Closing simulation app...")
simulation_app.close()
