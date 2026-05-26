#!/usr/bin/env python3
"""Headless Pi05 policy rollout on Isaac Lab scenes (route B)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

# LeRobot (train/infer stack) — must be on PYTHONPATH before policy imports.
_PHYSVLA_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _PHYSVLA_ROOT.parent
_ROLLOUT_SIM_DIR = Path(__file__).resolve().parent
_COLLECT_DIR = _PHYSVLA_ROOT / "collect"
_LEROBOT_SRC = _WORKSPACE_ROOT / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

import torch
from isaaclab.app import AppLauncher

for _path in (_ROLLOUT_SIM_DIR, _COLLECT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from isaaclab_env_module import apply_camera_launch_workarounds
from policy_obs_utils import (
    build_policy_observation,
    build_state14,
    compose_video_frame,
    write_mp4,
    write_summary,
)
from success_utils import evaluate_rollout_success, update_peak_joint_degs
from task_registry import (
    get_task_preset,
    list_task_presets,
    PHYSVLA_ASSETS_DIR,
)

parser = argparse.ArgumentParser(description="Pi05 headless policy rollout (Isaac Lab + LeRobot).")
parser.add_argument("--task_id", type=str, default=None, help="Task preset from task_registry.")
parser.add_argument("--list_tasks", action="store_true", help="List task presets and exit.")
parser.add_argument("--usd_path", type=str, default=None, help="Override TaskPreset.usd_path.")
parser.add_argument(
    "--policy.path",
    dest="policy_path",
    type=str,
    default=None,
    help="Path to LeRobot pretrained_model directory (checkpoints/.../pretrained_model).",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps", type=int, default=600, help="Max control steps per episode.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for summary.json and episode_*.mp4 (default: rollout_sim/rollouts/...).",
)
parser.add_argument(
    "--video_layout",
    choices=["head", "head_wrist"],
    default="head_wrist",
    help="MP4 layout per episode.",
)
parser.add_argument("--video_fps", type=float, default=10.0, help="FPS for saved MP4 files.")
parser.add_argument(
    "--policy_device",
    type=str,
    default="cuda",
    help="Torch device for Pi05 inference (cuda/cpu). Not Isaac AppLauncher --device.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.list_tasks:
    print("[INFO] Available task presets:")
    for preset in list_task_presets():
        print(f"  - {preset.task_id}: {preset.description}")
    raise SystemExit(0)

if not args_cli.task_id:
    parser.error("--task_id is required (use --list_tasks).")
if not args_cli.policy_path:
    parser.error("--policy.path is required.")

task_preset = get_task_preset(args_cli.task_id)
rollout_success_specs = task_preset.rollout_success_specs
if args_cli.usd_path:
    p = Path(args_cli.usd_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    task_preset = replace(task_preset, usd_path=str(p))

if args_cli.output_dir:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
else:
    utc8 = timezone(timedelta(hours=8))
    stamp = datetime.now(utc8).strftime("%Y%m%d_%H%M%S")
    output_dir = (_ROLLOUT_SIM_DIR / "rollouts" / f"{task_preset.task_id}_{stamp}").resolve()

args_cli = apply_camera_launch_workarounds(args_cli)
if not getattr(args_cli, "headless", False):
    print("[WARN] Running without --headless opens GUI viewports; recommend --headless for batch eval.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports (after AppLauncher — env_setup pulls in isaaclab.actuators/pxr).
from env_setup import (
    build_environment_module_config,
    build_piper_robot_cfg,
    build_scene_hinge_cfg,
)
from isaaclab_env_module import IsaacLabEnvironmentModule

# ── LeRobot policy imports ─────────────────────────────────────────
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


CAMERA_WIDTH = max(32, int(task_preset.camera_width))
CAMERA_HEIGHT = max(32, int(task_preset.camera_height))
instruction_text = (task_preset.language_instruction or "").strip()
if not instruction_text:
    raise ValueError(f"Task '{task_preset.task_id}' has empty language_instruction.")

_scene_usd = Path(task_preset.usd_path)
if not _scene_usd.is_file():
    raise FileNotFoundError(f"Scene USD missing: {_scene_usd}")

print(f"[INFO] Task: {task_preset.task_id}")
print(f"[INFO] Instruction: {instruction_text}")
print(f"[INFO] Policy: {args_cli.policy_path}")
print(f"[INFO] Output: {output_dir}")
print(f"[INFO] Assets root: {PHYSVLA_ASSETS_DIR.resolve()}")
if rollout_success_specs:
    for spec in rollout_success_specs:
        print(f"[INFO] Success spec: {spec.joint_prim} > {spec.angle_gt_deg}°")
else:
    print("[INFO] Success criteria: none (task_registry.rollout_success_specs empty)")

SCENE_JOINT_PHYSICS_WARMUP_STEPS = 12
JOINT_LOG_INTERVAL = 30  # 实时铰链角度；不需要时注释掉 run_episode 里的 [JOINT] 打印即可

env_module = IsaacLabEnvironmentModule(build_environment_module_config(task_preset))
sim = env_module.create_simulation(dt=1 / 60.0, render_interval=4)

robot = env_module.create_robot(build_piper_robot_cfg(task_preset))
scene_hinge_cfg = build_scene_hinge_cfg(task_preset)
if scene_hinge_cfg is not None:
    env_module.create_scene_articulation(scene_hinge_cfg)
env_module.initialize_robot_home_pose()
device = sim.device
policy_device = torch.device(args_cli.policy_device if torch.cuda.is_available() else "cpu")

arm_joint_ids = robot.find_joints("joint[1-6]")[0]
gripper_joint_ids = robot.find_joints("joint[7-8]")[0]
gripper_open_target = torch.tensor([[0.035, -0.035]], dtype=torch.float32, device=device)
gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)

# Cameras: headless uses sensor pipeline only (no viewport windows).
env_module.define_camera_prims()
sensor_cameras = env_module.create_sensor_cameras()
for cam_name in ("main", "wrist"):
    if cam_name in sensor_cameras:
        print(f"[INFO] Sensor camera ready: {cam_name}")

print("[INFO] Loading Pi05 policy and processors...")
policy = PI05Policy.from_pretrained(args_cli.policy_path, local_files_only=True)
policy.eval()
policy.to(policy_device)
preprocessor = PolicyProcessorPipeline.from_pretrained(
    args_cli.policy_path,
    config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    local_files_only=True,
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    args_cli.policy_path,
    config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    local_files_only=True,
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

control_hz = max(1, int(task_preset.control_hz))
vision_hz = min(max(1, int(task_preset.vision_hz)), control_hz)
sim_dt = float(sim.cfg.dt)
control_decimation = max(1, int(round((1.0 / control_hz) / sim_dt)))
vision_decimation = max(1, int(round(control_hz / vision_hz)))
print(
    f"[INFO] control_hz={control_hz}, vision_hz={vision_hz}, "
    f"control_decimation={control_decimation}, max_steps={args_cli.max_steps}"
)


def reset_episode() -> None:
    sim.reset()
    robot.reset()
    if env_module.scene_articulation is not None:
        env_module.scene_articulation.reset()
    if sensor_cameras:
        for sensor in sensor_cameras.values():
            sensor.reset()
    env_module.apply_joint_initial_overrides()
    env_module.sync_scene_joint_initials_to_sim()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(gripper_close_target, joint_ids=gripper_joint_ids)
    robot.write_data_to_sim()
    env_module.define_camera_prims()
    for _ in range(SCENE_JOINT_PHYSICS_WARMUP_STEPS):
        sim.step(render=True)
    sim.step(render=True)
    robot.update(sim.cfg.dt)
    if env_module.scene_articulation is not None:
        env_module.scene_articulation.update(sim.cfg.dt)
    policy.reset()


def run_episode(episode_index: int) -> dict:
    reset_episode()

    last_rgb_main = None
    last_rgb_wrist = None
    video_frames: list = []
    sim_step_count = 0
    control_step_count = 0
    success = False
    success_step: int | None = None
    final_joint_degs: dict[str, float | None] = {}
    peak_joint_degs: dict[str, float | None] = {
        spec.joint_prim: None for spec in rollout_success_specs
    }
    gripper_open = False

    for _ in range(args_cli.max_steps):
        sim.step(render=True)
        robot.update(sim.cfg.dt)
        if env_module.scene_articulation is not None:
            env_module.scene_articulation.update(sim.cfg.dt)
        sim_step_count += 1

        if sim_step_count % control_decimation == 0:
            control_step_count += 1

            success_now, joint_degs = evaluate_rollout_success(env_module, rollout_success_specs)
            final_joint_degs = joint_degs
            update_peak_joint_degs(peak_joint_degs, joint_degs)

            if rollout_success_specs and (
                control_step_count == 1
                or control_step_count % JOINT_LOG_INTERVAL == 0
                or success_now
            ):
                deg_str = ", ".join(
                    f"{k.split('/')[-1]}={v:.2f}°" if v is not None else f"{k.split('/')[-1]}=None"
                    for k, v in joint_degs.items()
                )
                print(f"[JOINT] step={control_step_count} {deg_str}")

            if success_now:
                success = True
                success_step = control_step_count
                break

            with torch.inference_mode():
                if (control_step_count - 1) % vision_decimation == 0:
                    new_wrist = env_module.capture_rgb("wrist", sim_dt)
                    new_main = env_module.capture_rgb("main", sim_dt)
                    if new_wrist is not None and new_main is not None:
                        last_rgb_wrist = new_wrist
                        last_rgb_main = new_main
                        frame = compose_video_frame(
                            last_rgb_main, last_rgb_wrist, args_cli.video_layout
                        )
                        if frame is not None:
                            video_frames.append(frame)

                if last_rgb_main is None or last_rgb_wrist is None:
                    continue

                state14 = build_state14(
                    robot,
                    arm_joint_ids,
                    gripper_open01=1.0 if gripper_open else 0.0,
                    device=policy_device,
                )
                obs = build_policy_observation(
                    rgb_main=last_rgb_main,
                    rgb_wrist=last_rgb_wrist,
                    state14=state14,
                    task=instruction_text,
                    device=policy_device,
                )
                batch = preprocessor(obs)
                action = policy.select_action(batch)
                action = postprocessor(action)
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                left7 = action[0, :7].detach()

                arm_targets = left7[:6].unsqueeze(0).to(device=device, dtype=torch.float32)
                gripper_open = bool(float(left7[6]) > 0.5)
                gripper_targets = gripper_open_target if gripper_open else gripper_close_target

                robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
                robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
                robot.write_data_to_sim()

    mp4_path = output_dir / f"episode_{episode_index:04d}.mp4"
    write_mp4(video_frames, mp4_path, args_cli.video_fps)
    return {
        "episode_index": episode_index,
        "success": success,
        "success_step": success_step,
        "steps": control_step_count,
        "final_joint_degs": final_joint_degs,
        "peak_joint_degs": peak_joint_degs,
        "video": str(mp4_path) if video_frames else None,
    }


output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "run_meta.json").write_text(
    json.dumps(
        {
            "task_id": task_preset.task_id,
            "instruction": instruction_text,
            "policy_path": str(Path(args_cli.policy_path).resolve()),
            "num_episodes": args_cli.num_episodes,
            "max_steps": args_cli.max_steps,
            "rollout_success_specs": [asdict(spec) for spec in rollout_success_specs],
            "headless": bool(getattr(args_cli, "headless", False)),
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

episode_results = []
try:
    for ep in range(args_cli.num_episodes):
        t0 = time.perf_counter()
        print(f"\n[INFO] ===== Episode {ep + 1}/{args_cli.num_episodes} =====")
        result = run_episode(ep)
        result["wall_time_sec"] = time.perf_counter() - t0
        episode_results.append(result)
        print(
            f"[INFO] Episode {ep}: success={result['success']}, "
            f"steps={result['steps']}, success_step={result['success_step']}, "
            f"final_joint_degs={result['final_joint_degs']}, "
            f"peak_joint_degs={result['peak_joint_degs']}"
        )
    write_summary(output_dir, episode_results)
finally:
    print("[INFO] Closing simulation app...")
    simulation_app.close()
