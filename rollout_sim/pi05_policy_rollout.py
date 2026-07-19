#!/usr/bin/env python3
"""Headless Pi05 policy rollout on Isaac Lab scenes (route B)."""

from __future__ import annotations

import argparse
import json
import random
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
from domain_randomization_robotwin2 import (
    add_randomization_cli_args,
    apply_randomization_sample,
    format_randomization_sample,
    randomization_config_from_args,
    sample_randomization,
)
from policy_obs_utils import (
    build_policy_observation,
    build_state14,
    compose_video_frame,
    write_mp4,
    write_summary,
)
from realtime_action_controller import RealtimeActionController
from success_utils import evaluate_rollout_success, update_peak_joint_degs
from task_registry import (
    get_task_preset,
    list_task_presets,
    PHYSVLA_ASSETS_DIR,
    TASK_PRESETS,
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
    "--fixed_task_pose",
    action="store_true",
    help="Use the registry base object pose and joint initialization for every episode.",
)
parser.add_argument(
    "--realtime_chunking",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use overlapping action chunks, temporal ensembling, filtering, and rate limiting.",
)
parser.add_argument(
    "--replan_hz",
    type=float,
    default=3.0,
    help="Policy chunk generation frequency in simulated time when realtime chunking is enabled.",
)
parser.add_argument(
    "--ensemble_k",
    type=float,
    default=0.0625,
    help="Exponential temporal-ensemble decay; newer chunks receive greater weight.",
)
parser.add_argument(
    "--max_joint_speed_rad_s",
    type=float,
    default=0.9,
    help="Per-joint target slew-rate limit used to approximate Piper speed=30.",
)
parser.add_argument(
    "--model_debug_interval",
    type=int,
    default=10,
    help=(
        "Print PI05 input/output health checks for the first five replans and then "
        "every N replans; set 0 to disable."
    ),
)
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
add_randomization_cli_args(parser)
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


def load_episode_task_preset():
    if args_cli.fixed_task_pose:
        return TASK_PRESETS[args_cli.task_id]
    return get_task_preset(args_cli.task_id)


task_preset = load_episode_task_preset()
rand_config = randomization_config_from_args(args_cli, task_preset)
rng = random.Random(rand_config.seed)
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
policy_parameter_count = sum(parameter.numel() for parameter in policy.parameters())
print(
    f"[MODEL] Loaded PI05 on {policy_device}: "
    f"parameters={policy_parameter_count:,}, training={policy.training}",
    flush=True,
)
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


def reset_episode():
    # Re-sample object pose before reset, then visual/environment DR after reset.
    # This is the same Direct-GPU-safe order used by automatic collection.
    episode_preset = load_episode_task_preset()
    if args_cli.usd_path:
        episode_preset = replace(episode_preset, usd_path=task_preset.usd_path)
    env_module.apply_task_preset_scene_root(episode_preset)
    env_module.apply_task_preset_joint_initial(episode_preset)
    sim.reset()
    robot.reset()
    if env_module.scene_articulation is not None:
        env_module.scene_articulation.reset()
    if sensor_cameras:
        for sensor in sensor_cameras.values():
            sensor.reset()
    env_module.ensure_scene_root_baseline()
    env_module.define_camera_prims()
    sample = sample_randomization(rand_config, rng)
    apply_randomization_sample(env_module, rand_config, sample)
    env_module.sync_scene_joints_after_sim_reset(warmup_steps=SCENE_JOINT_PHYSICS_WARMUP_STEPS)
    env_module.reset_robot_pose_via_targets(
        gripper_targets=gripper_close_target,
        gripper_joint_ids=gripper_joint_ids,
    )
    policy.reset()
    scene_spec = episode_preset.scene_root_specs[0] if episode_preset.scene_root_specs else None
    joint_spec = episode_preset.joint_initial_specs[0] if episode_preset.joint_initial_specs else None
    print(
        "[INFO] Episode init: "
        f"scene_xyz={tuple(round(float(v), 4) for v in scene_spec.translation) if scene_spec else None} "
        f"scene_rot={tuple(round(float(v), 2) for v in scene_spec.rotation_xyz) if scene_spec else None} "
        f"joint={float(joint_spec.position):.2f}° "
        f"DR=({format_randomization_sample(sample)})",
        flush=True,
    )
    return episode_preset, sample


def run_episode(episode_index: int) -> dict:
    episode_preset, randomization_sample = reset_episode()

    def fmt_values(values) -> str:
        return "[" + ", ".join(f"{float(v):+.4f}" for v in values) + "]"

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
    realtime_controller = None
    inference_durations: list[float] = []
    last_normalized_left7 = torch.zeros(7, dtype=torch.float32, device=policy_device)
    replan_count = 0
    previous_normalized_chunk = None
    if args_cli.realtime_chunking:
        realtime_controller = RealtimeActionController(
            control_hz=control_hz,
            replan_hz=args_cli.replan_hz,
            ensemble_k=args_cli.ensemble_k,
            max_joint_speed_rad_s=args_cli.max_joint_speed_rad_s,
        )
        initial_state14 = build_state14(
            robot,
            arm_joint_ids,
            gripper_open01=0.0,
            device=policy_device,
        )
        realtime_controller.reset(initial_state14[0])
        print(
            "[INFO] Realtime chunking: "
            f"replan_hz={args_cli.replan_hz:.2f} "
            f"interval={realtime_controller.replan_interval} control steps "
            f"ensemble_k={args_cli.ensemble_k:.4f} "
            f"max_joint_speed={args_cli.max_joint_speed_rad_s:.3f}rad/s",
            flush=True,
        )

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
                if realtime_controller is not None:
                    if realtime_controller.should_replan(control_step_count):
                        batch = preprocessor(obs)
                        if policy_device.type == "cuda":
                            torch.cuda.synchronize(policy_device)
                        inference_start = time.perf_counter()
                        normalized_chunk = policy.predict_action_chunk(batch)
                        if policy_device.type == "cuda":
                            torch.cuda.synchronize(policy_device)
                        inference_sec = time.perf_counter() - inference_start
                        inference_durations.append(inference_sec)

                        action_chunk = postprocessor(normalized_chunk)
                        if action_chunk.dim() == 2:
                            action_chunk = action_chunk.unsqueeze(0)
                        replan_count += 1
                        should_log_model = args_cli.model_debug_interval > 0 and (
                            replan_count <= 5
                            or replan_count % args_cli.model_debug_interval == 0
                        )
                        if should_log_model:
                            normalized_finite = bool(torch.isfinite(normalized_chunk).all())
                            action_finite = bool(torch.isfinite(action_chunk).all())
                            normalized_first = normalized_chunk[0, 0]
                            action_first = action_chunk[0, 0]
                            previous_delta = (
                                float(
                                    (
                                        normalized_chunk[0, 0]
                                        - previous_normalized_chunk[0, 0]
                                    )
                                    .abs()
                                    .max()
                                    .item()
                                )
                                if previous_normalized_chunk is not None
                                else None
                            )
                            previous_delta_text = (
                                f"{previous_delta:.5f}"
                                if previous_delta is not None
                                else "initial"
                            )
                            print(
                                f"[MODEL] replan={replan_count} step={control_step_count} "
                                f"input_state={fmt_values(obs['observation.state'][0])} "
                                f"head_mean={float(obs['observation.images.head'].mean()):.4f} "
                                f"wrist_mean={float(obs['observation.images.left_wrist'].mean()):.4f} "
                                f"normalized_shape={tuple(normalized_chunk.shape)} "
                                f"normalized_finite={normalized_finite} "
                                f"normalized_first={fmt_values(normalized_first[:7])} "
                                f"action_finite={action_finite} "
                                f"action_first={fmt_values(action_first[:7])} "
                                f"prev_first_max_delta={previous_delta_text}",
                                flush=True,
                            )
                        previous_normalized_chunk = normalized_chunk.detach().clone()
                        realtime_controller.add_chunk(
                            control_step_count,
                            action_chunk[0],
                        )
                        last_normalized_left7 = normalized_chunk[0, 0, :7].detach().clone()
                        achieved_hz = 1.0 / inference_sec if inference_sec > 0 else float("inf")
                        print(
                            f"[INFERENCE] step={control_step_count} "
                            f"latency={inference_sec * 1000:.1f}ms "
                            f"max_sync_hz={achieved_hz:.2f}",
                            flush=True,
                        )

                    selected_action = realtime_controller.action_for_step(
                        control_step_count,
                        state14[0],
                    )
                    if selected_action is None:
                        continue
                    action = selected_action.unsqueeze(0)
                    normalized_left7 = last_normalized_left7
                else:
                    batch = preprocessor(obs)
                    normalized_action = policy.select_action(batch)
                    normalized_batch = (
                        normalized_action.unsqueeze(0)
                        if normalized_action.dim() == 1
                        else normalized_action
                    )
                    normalized_left7 = normalized_batch[0, :7].detach().clone()
                    action = postprocessor(normalized_action)
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                left7 = action[0, :7].detach()

                arm_targets = left7[:6].unsqueeze(0).to(device=device, dtype=torch.float32)
                gripper_open = bool(float(left7[6]) > 0.5)
                gripper_targets = gripper_open_target if gripper_open else gripper_close_target

                if control_step_count <= 10 or control_step_count % JOINT_LOG_INTERVAL == 0:
                    current_arm = robot.data.joint_pos[0, arm_joint_ids].detach().to(
                        device=arm_targets.device, dtype=torch.float32
                    )
                    target_delta = arm_targets[0] - current_arm
                    print(
                        f"[ACTION] step={control_step_count} "
                        f"normalized_left7={fmt_values(normalized_left7)} "
                        f"current_arm={fmt_values(current_arm)} "
                        f"target_arm={fmt_values(arm_targets[0])} "
                        f"delta={fmt_values(target_delta)} "
                        f"gripper_raw={float(left7[6]):+.4f} open={gripper_open}",
                        flush=True,
                    )

                robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
                robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
                robot.write_data_to_sim()

    mp4_path = output_dir / f"episode_{episode_index:04d}.mp4"
    write_mp4(video_frames, mp4_path, args_cli.video_fps)
    mean_inference_ms = (
        1000.0 * sum(inference_durations) / len(inference_durations)
        if inference_durations
        else None
    )
    return {
        "episode_index": episode_index,
        "success": success,
        "success_step": success_step,
        "steps": control_step_count,
        "final_joint_degs": final_joint_degs,
        "peak_joint_degs": peak_joint_degs,
        "video": str(mp4_path) if video_frames else None,
        "scene_root_specs": [asdict(spec) for spec in episode_preset.scene_root_specs],
        "joint_initial_specs": [asdict(spec) for spec in episode_preset.joint_initial_specs],
        "randomization": asdict(randomization_sample),
        "realtime_chunking": bool(args_cli.realtime_chunking),
        "replan_hz": args_cli.replan_hz if args_cli.realtime_chunking else None,
        "mean_inference_ms": mean_inference_ms,
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
            "randomization_config": asdict(rand_config),
            "realtime_chunking": bool(args_cli.realtime_chunking),
            "replan_hz": args_cli.replan_hz,
            "ensemble_k": args_cli.ensemble_k,
            "max_joint_speed_rad_s": args_cli.max_joint_speed_rad_s,
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
