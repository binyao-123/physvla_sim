#!/usr/bin/env python3
"""Headless PRTS policy rollout on Isaac Lab scenes.

Control path: obs -> PRTSRuntime.select_action -> slew-limit -> joints.
Scene / DR / success / video match pi05_policy_rollout.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PHYSVLA_ROOT = Path(__file__).resolve().parents[1]
_ROLLOUT_SIM_DIR = Path(__file__).resolve().parent
_COLLECT_DIR = _PHYSVLA_ROOT / "collect"

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
from episode_config import (
    DR_STREAM,
    EPISODE_CONFIG_DIRNAME,
    POLICY_STREAM,
    POSE_STREAM,
    build_episode_config,
    derive_episode_seed,
    episode_preset_from_config,
    load_episode_configs,
    randomization_config_from_config,
    randomization_sample_from_config,
    write_episode_config,
)
from policy_obs_utils import (
    build_policy_observation,
    build_state7,
    compose_video_frame,
    write_mp4,
    write_summary,
)
from prts_runtime import PRTSRuntime
from success_utils import evaluate_rollout_success, update_peak_joint_degs
from task_registry import (
    PHYSVLA_ASSETS_DIR,
    TASK_PRESETS,
    get_task_preset,
    list_task_presets,
)

parser = argparse.ArgumentParser(description="PRTS headless policy rollout (Isaac Lab).")
parser.add_argument("--task_id", type=str, default=None, help="Task preset from task_registry.")
parser.add_argument("--list_tasks", action="store_true", help="List task presets and exit.")
parser.add_argument("--usd_path", type=str, default=None, help="Override TaskPreset.usd_path.")
parser.add_argument(
    "--policy.path",
    dest="policy_path",
    type=str,
    default=None,
    help="Path to PRTS checkpoint directory (contains config.json + statistics.json).",
)
parser.add_argument(
    "--policy_server",
    type=str,
    default=None,
    help="Remote PRTS websocket server as host:port. Overrides --policy.path.",
)
parser.add_argument(
    "--prts_root",
    type=str,
    default=str(Path("/home/ubuntu/workspace/PRTS")),
    help="Cloned TeleHuman/PRTS repository root.",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=600,
    help="Max control steps per episode (not sim substeps).",
)
parser.add_argument(
    "--fixed_task_pose",
    action="store_true",
    help="Use the registry base object pose and joint initialization for every episode.",
)
parser.add_argument(
    "--max_joint_speed_rad_s",
    type=float,
    default=0.9,
    help="Per-control-step arm joint slew limit (rad/s). <=0 disables.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for summary.json and episode_*.mp4.",
)
parser.add_argument(
    "--save_video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save episode preview MP4s (use --no-save_video to skip).",
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
    help="Torch device for PRTS inference (cuda/cpu). Not Isaac AppLauncher --device.",
)
parser.add_argument("--action_dim", type=int, default=7, help="PRTS action dimensionality (left6 ckpt = 7).")
parser.add_argument(
    "--state_relative_action",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Add current state back on piper_left6 masked dims (joints relative, gripper absolute).",
)
parser.add_argument(
    "--action_exec_horizon",
    type=int,
    default=1,
    help="How many steps of each predicted chunk to execute before replanning.",
)
parser.add_argument(
    "--seed_policy_noise",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Send a per-(episode, inference) seed to the remote policy so its flow-matching "
        "noise is reproducible. Disable to get fresh noise on every call."
    ),
)
parser.add_argument(
    "--replay_episode_config",
    type=str,
    default=None,
    help=(
        "Replay saved episode initial configs instead of sampling. Accepts one "
        "episode_XXXX.json, a directory of them, or a previous rollout output dir."
    ),
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
if not args_cli.policy_path and not args_cli.policy_server:
    parser.error("--policy.path or --policy_server is required.")

server_host = server_port = None
if args_cli.policy_server:
    _spec = args_cli.policy_server.strip()
    if _spec.startswith("ws://"):
        _spec = _spec[len("ws://") :]
    if ":" not in _spec:
        parser.error(f"--policy_server must be host:port, got: {args_cli.policy_server}")
    server_host, _, _port_str = _spec.rpartition(":")
    if not _port_str.isdigit():
        parser.error(f"--policy_server port is not a number: {args_cli.policy_server}")
    server_port = int(_port_str)


def load_episode_task_preset():
    if args_cli.fixed_task_pose:
        return TASK_PRESETS[args_cli.task_id]
    return get_task_preset(args_cli.task_id)


replay_configs = load_episode_configs(args_cli.replay_episode_config) if args_cli.replay_episode_config else None
if replay_configs:
    replay_single = Path(args_cli.replay_episode_config).expanduser().resolve().is_file()
    if not replay_single:
        args_cli.num_episodes = len(replay_configs)
    print(
        f"[INFO] Replaying {len(replay_configs)} saved episode config(s) from "
        f"{args_cli.replay_episode_config}; num_episodes={args_cli.num_episodes}"
    )

task_preset = load_episode_task_preset()
rand_config = randomization_config_from_args(args_cli, task_preset)
# Always materialize a run seed, so even an unseeded run stays reproducible after the fact.
run_seed = int(rand_config.seed) if rand_config.seed is not None else random.SystemRandom().randrange(2**31)
rand_config = replace(rand_config, seed=run_seed)
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
    output_dir = (_ROLLOUT_SIM_DIR / "rollouts" / "PRTS" / f"{task_preset.task_id}_{stamp}").resolve()

args_cli = apply_camera_launch_workarounds(args_cli)
if not getattr(args_cli, "headless", False):
    print("[WARN] Running without --headless opens GUI viewports; recommend --headless for batch eval.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from env_setup import (
    build_environment_module_config,
    build_piper_robot_cfg,
    build_scene_hinge_cfg,
)
from isaaclab_env_module import IsaacLabEnvironmentModule

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
if args_cli.policy_server:
    print(f"[INFO] Policy: remote ws://{server_host}:{server_port}")
else:
    print(f"[INFO] Policy: {args_cli.policy_path}")
print(f"[INFO] PRTS root: {args_cli.prts_root}")
print(f"[INFO] Output: {output_dir}")
print(f"[INFO] Assets root: {PHYSVLA_ASSETS_DIR.resolve()}")
if rollout_success_specs:
    for spec in rollout_success_specs:
        if spec.angle_gt_deg is not None:
            print(f"[INFO] Success spec: {spec.joint_prim} > {spec.angle_gt_deg}")
        if spec.angle_lt_deg is not None:
            print(f"[INFO] Success spec: {spec.joint_prim} < {spec.angle_lt_deg}")
else:
    print("[INFO] Success criteria: none (task_registry.rollout_success_specs empty)")

SCENE_JOINT_PHYSICS_WARMUP_STEPS = 12

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
# left6 ckpts never open the gripper; keep both fingers at 0.
gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)

env_module.define_camera_prims()
sensor_cameras = env_module.create_sensor_cameras()
for cam_name in ("main", "wrist"):
    if cam_name in sensor_cameras:
        print(f"[INFO] Sensor camera ready: {cam_name}")

if args_cli.policy_server:
    # Server does unnormalize + state-relative inverse; client gets absolute joints.
    print("[INFO] Connecting to remote PRTS policy...")
    from prts_remote_runtime import PRTSRemoteRuntime

    policy = PRTSRemoteRuntime(
        server_host,
        server_port,
        prts_root=args_cli.prts_root,
        action_dim=args_cli.action_dim,
        action_exec_horizon=args_cli.action_exec_horizon,
    )
else:
    print("[INFO] Loading PRTS policy...")
    policy = PRTSRuntime(
        args_cli.policy_path,
        prts_root=args_cli.prts_root,
        device=policy_device,
        action_dim=args_cli.action_dim,
        state_relative_action=args_cli.state_relative_action,
        action_exec_horizon=args_cli.action_exec_horizon,
    )

control_hz = max(1, int(task_preset.control_hz))
vision_hz = min(max(1, int(task_preset.vision_hz)), control_hz)
sim_dt = float(sim.cfg.dt)
control_decimation = max(1, int(round((1.0 / control_hz) / sim_dt)))
vision_decimation = max(1, int(round(control_hz / vision_hz)))
max_joint_delta = (
    float(args_cli.max_joint_speed_rad_s) / float(control_hz)
    if args_cli.max_joint_speed_rad_s > 0
    else None
)
print(
    f"[INFO] control_hz={control_hz}, vision_hz={vision_hz}, "
    f"control_decimation={control_decimation}, max_steps={args_cli.max_steps}, "
    f"max_joint_speed_rad_s={args_cli.max_joint_speed_rad_s}"
)


def sample_episode_init(episode_index: int):
    """Return (episode_preset, DR sample, DR config, pose_seed, dr_seed) for one episode.

    Seeding both streams per episode from (run_seed, episode_index) keeps episode N
    independent of how many draws episodes 0..N-1 consumed.
    """

    if replay_configs is not None:
        saved = replay_configs[episode_index % len(replay_configs)]
        episode_preset = episode_preset_from_config(TASK_PRESETS[args_cli.task_id], saved)
        return (
            episode_preset,
            randomization_sample_from_config(saved),
            randomization_config_from_config(saved),
            (saved.run_seed, saved.pose_seed, saved.dr_seed, saved.policy_seed),
        )

    pose_seed = derive_episode_seed(run_seed, episode_index, POSE_STREAM)
    dr_seed = derive_episode_seed(run_seed, episode_index, DR_STREAM)
    policy_seed = derive_episode_seed(run_seed, episode_index, POLICY_STREAM)
    # task_registry samples object pose / joint initial off the global random module.
    random.seed(pose_seed)
    episode_preset = load_episode_task_preset()
    if args_cli.usd_path:
        episode_preset = replace(episode_preset, usd_path=task_preset.usd_path)
    sample = sample_randomization(rand_config, random.Random(dr_seed))
    return episode_preset, sample, rand_config, (run_seed, pose_seed, dr_seed, policy_seed)


def reset_episode(episode_index: int):
    episode_preset, sample, episode_rand_config, seeds = sample_episode_init(episode_index)
    episode_run_seed, pose_seed, dr_seed, policy_seed = seeds
    write_episode_config(
        output_dir / EPISODE_CONFIG_DIRNAME,
        build_episode_config(
            task_id=args_cli.task_id,
            episode_index=episode_index,
            run_seed=episode_run_seed,
            pose_seed=pose_seed,
            dr_seed=dr_seed,
            policy_seed=policy_seed,
            fixed_task_pose=bool(args_cli.fixed_task_pose),
            episode_preset=episode_preset,
            randomization_sample=sample,
            randomization_config=episode_rand_config,
            replayed_from=args_cli.replay_episode_config,
        ),
    )
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
    apply_randomization_sample(env_module, episode_rand_config, sample)
    env_module.sync_scene_joints_after_sim_reset(warmup_steps=SCENE_JOINT_PHYSICS_WARMUP_STEPS)
    env_module.reset_robot_pose_via_targets(
        gripper_targets=gripper_close_target,
        gripper_joint_ids=gripper_joint_ids,
    )
    for cam_name in ("main", "wrist"):
        ok, reason = env_module.validate_camera_mount(cam_name)
        if not ok:
            print(f"[WARN] Camera mount failed ({cam_name}): {reason}", flush=True)
    policy.reset()
    # Remote runtime only; the local PRTSRuntime has no server-side noise seed.
    if hasattr(policy, "set_episode_seed"):
        policy.set_episode_seed(policy_seed if args_cli.seed_policy_noise else None)
    scene_spec = episode_preset.scene_root_specs[0] if episode_preset.scene_root_specs else None
    joint_spec = episode_preset.joint_initial_specs[0] if episode_preset.joint_initial_specs else None
    print(
        "[INFO] init "
        f"{'replay ' if replay_configs is not None else ''}"
        f"pose_seed={pose_seed} dr_seed={dr_seed} "
        f"xyz={tuple(round(float(v), 3) for v in scene_spec.translation) if scene_spec else None} "
        f"rot={tuple(round(float(v), 1) for v in scene_spec.rotation_xyz) if scene_spec else None} "
        f"joint0={float(joint_spec.position):.4f} "
        f"DR={format_randomization_sample(sample)}",
        flush=True,
    )
    return episode_preset, sample


def run_episode(episode_index: int) -> dict:
    episode_preset, randomization_sample = reset_episode(episode_index)

    last_rgb_main = None
    last_rgb_wrist = None
    video_frames: list = []
    control_step_count = 0
    success = False
    success_step: int | None = None
    final_joint_degs: dict[str, float | None] = {}
    peak_joint_degs: dict[str, float | None] = {
        spec.joint_prim: None for spec in rollout_success_specs
    }
    last_arm_cmd: torch.Tensor | None = None

    while control_step_count < args_cli.max_steps:
        for _ in range(control_decimation):
            sim.step(render=True)
            robot.update(sim.cfg.dt)
            if env_module.scene_articulation is not None:
                env_module.scene_articulation.update(sim.cfg.dt)

        control_step_count += 1

        success_now, joint_degs = evaluate_rollout_success(env_module, rollout_success_specs)
        final_joint_degs = joint_degs
        update_peak_joint_degs(peak_joint_degs, joint_degs, specs=rollout_success_specs)

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
                    if args_cli.save_video:
                        frame = compose_video_frame(
                            last_rgb_main, last_rgb_wrist, args_cli.video_layout
                        )
                        if frame is not None:
                            video_frames.append(frame)

            if last_rgb_main is None or last_rgb_wrist is None:
                continue

            state7 = build_state7(
                robot,
                arm_joint_ids,
                gripper_open01=0.0,  # left6: gripper always closed
                device=policy_device,
            )
            obs = build_policy_observation(
                rgb_main=last_rgb_main,
                rgb_wrist=last_rgb_wrist,
                state=state7,
                task=instruction_text,
                device=policy_device,
                include_right_wrist=False,
            )

            action = policy.select_action(obs)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            # Only left 7-D is valid; joints 1–6 move, gripper stays 0.
            left7 = action[0, :7].detach()

            arm_targets = left7[:6].unsqueeze(0).to(device=device, dtype=torch.float32)
            if max_joint_delta is not None:
                if last_arm_cmd is None:
                    last_arm_cmd = robot.data.joint_pos[0, arm_joint_ids].detach().clone()
                delta = arm_targets[0] - last_arm_cmd
                arm_targets = (last_arm_cmd + delta.clamp(-max_joint_delta, max_joint_delta)).unsqueeze(0)
                last_arm_cmd = arm_targets[0].detach().clone()
            gripper_targets = gripper_close_target

            robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
            robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
            robot.write_data_to_sim()

    mp4_path = None
    if args_cli.save_video and video_frames:
        mp4_path = output_dir / f"episode_{episode_index:04d}.mp4"
        write_mp4(video_frames, mp4_path, args_cli.video_fps)
    return {
        "episode_index": episode_index,
        "success": success,
        "success_step": success_step,
        "steps": control_step_count,
        "final_joint_degs": final_joint_degs,
        "peak_joint_degs": peak_joint_degs,
        "video": str(mp4_path) if mp4_path is not None else None,
        "scene_root_specs": [asdict(spec) for spec in episode_preset.scene_root_specs],
        "joint_initial_specs": [asdict(spec) for spec in episode_preset.joint_initial_specs],
        "randomization": asdict(randomization_sample),
    }


output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "run_meta.json").write_text(
    json.dumps(
        {
            "task_id": task_preset.task_id,
            "instruction": instruction_text,
            "usd_path": str(Path(task_preset.usd_path).resolve()),
            "policy_path": str(Path(args_cli.policy_path).resolve()) if args_cli.policy_path else None,
            "policy_server": f"ws://{server_host}:{server_port}" if args_cli.policy_server else None,
            "prts_root": str(Path(args_cli.prts_root).resolve()),
            "run_seed": run_seed,
            "fixed_task_pose": bool(args_cli.fixed_task_pose),
            "replay_episode_config": args_cli.replay_episode_config,
            "episode_config_dir": EPISODE_CONFIG_DIRNAME,
            "num_episodes": args_cli.num_episodes,
            "max_steps": args_cli.max_steps,
            "control_path": "prts_sample_actions",
            "action_dim": args_cli.action_dim,
            "state_relative_action": args_cli.state_relative_action,
            "action_exec_horizon": args_cli.action_exec_horizon,
            "cameras": ["head", "left_wrist"],
            "rollout_success_specs": [asdict(spec) for spec in rollout_success_specs],
            "randomization_config": asdict(rand_config),
            "control_hz": control_hz,
            "vision_hz": vision_hz,
            "headless": bool(getattr(args_cli, "headless", False)),
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

episode_results = []
rollout_failed = False
try:
    for ep in range(args_cli.num_episodes):
        t0 = time.perf_counter()
        print(f"\n[INFO] ===== Episode {ep + 1}/{args_cli.num_episodes} =====", flush=True)
        result = run_episode(ep)
        result["wall_time_sec"] = time.perf_counter() - t0
        episode_results.append(result)
        joint_peak = result["peak_joint_degs"]
        joint_final = result["final_joint_degs"]
        if rollout_success_specs:
            key = rollout_success_specs[0].joint_prim
            peak_v = joint_peak.get(key)
            final_v = joint_final.get(key)
            joint_text = (
                f"joint={peak_v:.4f}/{final_v:.4f}"
                if peak_v is not None and final_v is not None
                else "joint=?"
            )
        else:
            joint_text = "joint=n/a"
        print(
            f"[INFO] Episode {ep}: success={result['success']} steps={result['steps']} {joint_text}",
            flush=True,
        )
    write_summary(output_dir, episode_results)
except Exception:
    import traceback

    rollout_failed = True
    print("[ERROR] rollout failed; traceback follows", flush=True)
    traceback.print_exc()
    raise
finally:
    print("[INFO] Closing simulation app...", flush=True)

    def _force_exit(_signum, _frame):
        # Isaac teardown can hang forever; every artifact is already on disk by now.
        print("[WARN] simulation_app.close() hung; forcing exit.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if rollout_failed else 0)

    signal.signal(signal.SIGALRM, _force_exit)
    signal.alarm(20)
    try:
        simulation_app.close()
    except Exception as exc:
        print(f"[WARN] simulation_app.close() raised: {exc}", flush=True)
    finally:
        signal.alarm(0)
