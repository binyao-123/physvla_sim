#!/usr/bin/env python3
"""PLD stage-2 data collection: PRTS base policy probes, motion planning finishes.

One episode is a single continuous trajectory made of two segments:

  * probing  -- the deployed base policy drives the arm for T_probe control steps,
    sampled per episode from U[0, alpha*T]. This is what puts the arm in the
    off-distribution states the base policy actually reaches at deployment, and it
    is recorded, not discarded: the recovery it teaches is only well defined
    relative to the drift that preceded it.
  * takeover -- the scripted expert re-solves the task from wherever the probing
    left the arm and object. It re-reads the handle pose and the live joint state,
    so no reset happens at the seam and the step budget keeps counting.

Only successful episodes are worth distilling, but failures are kept on disk with
``success=false`` so the probing-drift distribution can be inspected.

Scene init / DR / seeding / success criteria are identical to prts_policy_rollout.py,
so a PLD episode and a plain BC eval episode with the same seed start from the same
scene.
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
    PROBE_STREAM,
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
    rgb_tensor_to_uint8_hwc,
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
parser.add_argument(
    "--probe_alpha",
    type=float,
    default=0.2,
    help="Probing length is drawn from U[probe_min_steps, probe_alpha*max_steps] control steps.",
)
parser.add_argument(
    "--probe_min_steps",
    type=int,
    default=10,
    help="Lower bound on the probing prefix; 0 would make the episode a pure MP demo.",
)
parser.add_argument(
    "--target_successes",
    type=int,
    default=0,
    help=(
        "Keep attempting episodes until this many succeed, instead of running a fixed "
        "num_episodes. 0 keeps the fixed-count behaviour."
    ),
)
parser.add_argument(
    "--episode_index_offset",
    type=int,
    default=0,
    help=(
        "Start episode indices here instead of 0. A top-up run must set this past the "
        "previous run's last index: scenes are derived from (run_seed, episode_index), so "
        "restarting at 0 with the same seed would re-collect the same scenes."
    ),
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=0,
    help="Attempt cap when --target_successes is set. 0 means no cap (run until the success target).",
)
parser.add_argument(
    "--expert_verbose",
    action="store_true",
    help="Print the expert's per-phase planning detail (waypoint counts, step budgets).",
)
parser.add_argument(
    "--takeover_ee_abort_mm",
    type=float,
    default=0.0,
    help=(
        "EE-to-handle drift that aborts the takeover; 0 disables it. The 30mm collection "
        "default assumes a clean approach from home. During a takeover the gripper commonly "
        "slides along the lid while still driving it, so this guard fires on episodes that "
        "are in fact progressing; the step budget and the joint-angle spec decide instead."
    ),
)
parser.add_argument(
    "--task_config",
    type=str,
    default=None,
    help="Interaction config yaml for the expert (defaults to the task's registered one).",
)
parser.add_argument(
    "--save_failed",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep artifacts for episodes the expert could not finish.",
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
    output_dir = (_ROLLOUT_SIM_DIR / "rollouts" / "PRTS_PLD" / f"{task_preset.task_id}_{stamp}").resolve()

args_cli = apply_camera_launch_workarounds(args_cli)
if not getattr(args_cli, "headless", False):
    print("[WARN] Running without --headless opens GUI viewports; recommend --headless for batch eval.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from collection_health import HealthLimits
from env_setup import (
    build_environment_module_config,
    build_piper_robot_cfg,
    build_scene_hinge_cfg,
    compute_control_loop_timing,
    resolve_robot_handles,
)
from interaction_executor import PushInteractionExecutor
from isaaclab_env_module import IsaacLabEnvironmentModule
from recording_utils import RecordingContext, build_step_tensors, capture_rgb_if_due
from task_config import load_task_interaction_config

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

task_interaction = load_task_interaction_config(
    task_preset.task_id,
    Path(args_cli.task_config).expanduser().resolve() if args_cli.task_config else None,
)
if task_interaction.interaction_mode != "push":
    raise NotImplementedError(
        f"PLD takeover needs a push-mode expert; task '{task_preset.task_id}' is "
        f"'{task_interaction.interaction_mode}'."
    )
if not task_interaction.sampling or not task_interaction.push:
    raise ValueError(f"Task config for '{task_preset.task_id}' must define 'sampling' and 'push'.")

handles = resolve_robot_handles(robot)
timing = compute_control_loop_timing(task_preset, float(sim.cfg.dt))
gripper_open_target = torch.tensor([[0.035, -0.035]], dtype=torch.float32, device=device)

joint_upper_limit_deg = 104.0
for _spec in task_preset.joint_limit_specs:
    if _spec.prim_path == task_interaction.joint_prim and _spec.upper_limit is not None:
        joint_upper_limit_deg = float(_spec.upper_limit)

_diff_ik_cfg = DifferentialIKControllerCfg(
    command_type="pose", ik_method="dls", ik_params={"lambda_val": 0.1}
)
_diff_ik_pos_cfg = DifferentialIKControllerCfg(
    command_type="position", ik_method="dls", ik_params={"lambda_val": 0.1}
)
expert = PushInteractionExecutor(
    robot=robot,
    env_module=env_module,
    handles=handles,
    timing=timing,
    task_config=task_interaction,
    sampling_config=task_interaction.sampling,
    push_config=task_interaction.push,
    success_specs=rollout_success_specs,
    device=device,
    diff_ik_controller=DifferentialIKController(_diff_ik_cfg, num_envs=1, device=device),
    diff_ik_pos_controller=DifferentialIKController(_diff_ik_pos_cfg, num_envs=1, device=device),
    gripper_open_target=gripper_open_target,
    gripper_close_target=gripper_close_target,
    joint_upper_limit_deg=joint_upper_limit_deg,
    rng=np.random.default_rng(run_seed),
    verbose=bool(args_cli.expert_verbose),
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
    # Probing length belongs to the episode, not to this run: on replay it has to come off
    # the saved run seed and the saved index, or the replayed scene gets a different drift.
    source_index = (
        replay_configs[episode_index % len(replay_configs)].episode_index
        if replay_configs is not None
        else episode_index
    )
    probe_seed = derive_episode_seed(episode_run_seed, source_index, PROBE_STREAM)
    return episode_preset, sample, probe_seed


class PLDTrajectoryRecorder:
    """Collector shim holding one hybrid episode in memory.

    Exposes the OfficialEpisodeCollector call surface so PushInteractionExecutor can
    write the takeover segment into it, but buffers instead of appending to the shared
    HDF5: a PLD episode is only worth keeping once the expert has actually finished it.
    The probing loop writes through the same ``add_step``, so both segments land in one
    trajectory with one format.
    """

    def __init__(self, *, video_layout: str, save_video: bool):
        self.video_layout = video_layout
        self.save_video = save_video
        self.segment = "probe"
        self.step_hook = None
        self.reset_episode()

    def reset_episode(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.segments: list[str] = []
        self.head_frames: list[np.ndarray] = []
        self.wrist_frames: list[np.ndarray] = []
        self.preview_frames: list[np.ndarray] = []

    def has_data(self) -> bool:
        return bool(self.states)

    def set_initial_state(self, initial_state) -> None:
        del initial_state

    def add_step(self, obs_dict, actions, reward, done, state_dict=None) -> None:
        del reward, done, state_dict
        rgb_main = obs_dict.get("rgb_main")
        rgb_wrist = obs_dict.get("rgb_wrist")
        self.states.append(obs_dict["robot_joint_pos"][0].detach().float().cpu().numpy().copy())
        self.actions.append(actions[0].detach().float().cpu().numpy().copy())
        self.segments.append(self.segment)
        if rgb_main is not None:
            self.head_frames.append(rgb_tensor_to_uint8_hwc(rgb_main))
        if rgb_wrist is not None:
            self.wrist_frames.append(rgb_tensor_to_uint8_hwc(rgb_wrist))
        if self.save_video:
            frame = compose_video_frame(rgb_main, rgb_wrist, self.video_layout)
            if frame is not None:
                self.preview_frames.append(frame)
        if self.step_hook is not None:
            self.step_hook()

    def export_episode(self, success: bool):
        del success
        return False, None, len(self.states)

    def close(self) -> None:
        pass

    def segment_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.segments:
            counts[name] = counts.get(name, 0) + 1
        return counts

    def save(self, raw_dir: Path, episode_index: int) -> dict[str, str | None]:
        """Dump the trajectory in the layout the LeRobot v2.1 conversion reads.

        States/actions go to one npz; the two camera streams go to per-episode mp4s,
        which is already how LeRobot v2.1 stores image features.
        """

        raw_dir.mkdir(parents=True, exist_ok=True)
        npz_path = raw_dir / f"episode_{episode_index:04d}.npz"
        np.savez(
            npz_path,
            states=np.asarray(self.states, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            segments=np.asarray(self.segments),
        )
        paths: dict[str, str | None] = {"npz": str(npz_path), "head": None, "wrist": None}
        for key, frames in (("head", self.head_frames), ("wrist", self.wrist_frames)):
            if not frames:
                continue
            path = raw_dir / f"episode_{episode_index:04d}_{key}.mp4"
            write_mp4(frames, path, float(control_hz))
            paths[key] = str(path)
        return paths


def run_episode(episode_index: int) -> dict:
    episode_preset, randomization_sample, probe_seed = reset_episode(episode_index)

    recorder = PLDTrajectoryRecorder(
        video_layout=args_cli.video_layout, save_video=args_cli.save_video
    )
    control_step_count = 0
    sim_step_count = 0
    success = False
    success_step: int | None = None
    final_joint_degs: dict[str, float | None] = {}
    peak_joint_degs: dict[str, float | None] = {
        spec.joint_prim: None for spec in rollout_success_specs
    }
    last_arm_cmd: torch.Tensor | None = None

    probe_cap = max(0, int(round(args_cli.probe_alpha * args_cli.max_steps)))
    probe_lo = min(max(0, int(args_cli.probe_min_steps)), probe_cap)
    t_probe = random.Random(probe_seed).randint(probe_lo, probe_cap) if probe_cap > 0 else 0

    episode_start_wall = time.perf_counter()
    ctx = RecordingContext(
        device=device,
        sim_dt=sim_dt,
        sim_step_count=0,
        control_step_count=0,
        vision_decimation=vision_decimation,
        episode_start_wall_time=episode_start_wall,
        arm_joint_ids=arm_joint_ids,
        ee_body_id=handles.ee_body_id,
    )

    def track_joints() -> None:
        nonlocal final_joint_degs
        _, joint_degs = evaluate_rollout_success(env_module, rollout_success_specs)
        final_joint_degs = joint_degs
        update_peak_joint_degs(peak_joint_degs, joint_degs, specs=rollout_success_specs)

    recorder.step_hook = track_joints
    print(f"[PLD] episode {episode_index}: probing for {t_probe} control steps", flush=True)

    recorder.segment = "probe"
    while control_step_count < t_probe:
        for _ in range(control_decimation):
            sim.step(render=True)
            robot.update(sim.cfg.dt)
            if env_module.scene_articulation is not None:
                env_module.scene_articulation.update(sim.cfg.dt)
            sim_step_count += 1

        control_step_count += 1
        ctx.control_step_count = control_step_count
        ctx.sim_step_count = sim_step_count

        success_now, joint_degs = evaluate_rollout_success(env_module, rollout_success_specs)
        final_joint_degs = joint_degs
        update_peak_joint_degs(peak_joint_degs, joint_degs, specs=rollout_success_specs)
        if success_now:
            success = True
            success_step = control_step_count
            break

        # no_grad, not inference_mode: the expert later writes in place into the same robot
        # buffers from outside, which torch forbids once they are inference tensors.
        with torch.no_grad():
            rgb_main, rgb_wrist = capture_rgb_if_due(env_module, ctx)
            if rgb_main is None or rgb_wrist is None:
                continue

            state7 = build_state7(
                robot,
                arm_joint_ids,
                gripper_open01=0.0,  # left6: gripper always closed
                device=policy_device,
            )
            obs = build_policy_observation(
                rgb_main=rgb_main,
                rgb_wrist=rgb_wrist,
                state=state7,
                task=instruction_text,
                device=policy_device,
                include_right_wrist=False,
            )

            action = policy.select_action(obs)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            left7 = action[0, :7].detach()

            arm_targets = left7[:6].unsqueeze(0).to(device=device, dtype=torch.float32)
            if max_joint_delta is not None:
                if last_arm_cmd is None:
                    last_arm_cmd = robot.data.joint_pos[0, arm_joint_ids].detach().clone()
                delta = arm_targets[0] - last_arm_cmd
                arm_targets = (last_arm_cmd + delta.clamp(-max_joint_delta, max_joint_delta)).unsqueeze(0)
                last_arm_cmd = arm_targets[0].detach().clone()

            obs_dict, action_t, reward_t, done_t, state_dict = build_step_tensors(
                robot, ctx, arm_targets, False, rgb_main, rgb_wrist
            )
            recorder.add_step(obs_dict, action_t, reward_t, done_t, state_dict)

            robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
            robot.set_joint_position_target(gripper_close_target, joint_ids=gripper_joint_ids)
            robot.write_data_to_sim()

    probe_steps = control_step_count
    takeover_used = False
    takeover_error = None
    if not success:
        # Hand the expert the running counters so max_steps stays an episode budget.
        takeover_used = True
        recorder.segment = "takeover"
        expert.sim_step_count = sim_step_count
        expert.control_step_count = control_step_count
        print(f"[PLD] episode {episode_index}: expert takeover at step {control_step_count}", flush=True)
        try:
            success, final_joint_degs = expert.takeover(
                recorder,
                episode_start_wall,
                episode_step_limit=args_cli.max_steps,
                ee_handle_abort_dist_m=args_cli.takeover_ee_abort_mm / 1000.0,
            )
        except Exception as exc:  # expert aborts must not kill the whole run
            takeover_error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] episode {episode_index}: takeover raised {takeover_error}", flush=True)
        control_step_count = int(expert.control_step_count)
        update_peak_joint_degs(peak_joint_degs, final_joint_degs, specs=rollout_success_specs)
        if success:
            success_step = control_step_count

    mp4_path = None
    if args_cli.save_video and recorder.preview_frames:
        mp4_path = output_dir / f"episode_{episode_index:04d}.mp4"
        write_mp4(recorder.preview_frames, mp4_path, args_cli.video_fps)

    raw_paths: dict[str, str | None] = {}
    if recorder.has_data() and (success or args_cli.save_failed):
        raw_paths = recorder.save(output_dir / "raw", episode_index)

    return {
        "episode_index": episode_index,
        "success": success,
        "success_step": success_step,
        "steps": control_step_count,
        "probe_steps": probe_steps,
        "probe_steps_planned": t_probe,
        "takeover_used": takeover_used,
        "takeover_error": takeover_error,
        "recorded_steps": len(recorder.states),
        "segment_counts": recorder.segment_counts(),
        "final_joint_degs": final_joint_degs,
        "peak_joint_degs": peak_joint_degs,
        "video": str(mp4_path) if mp4_path is not None else None,
        "raw": raw_paths,
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
            "control_path": "pld_probe_then_mp_takeover",
            "probe_alpha": args_cli.probe_alpha,
            "probe_min_steps": args_cli.probe_min_steps,
            "expert": "PushInteractionExecutor.takeover",
            "takeover_ee_abort_mm": args_cli.takeover_ee_abort_mm,
            "task_config": str(Path(args_cli.task_config).resolve()) if args_cli.task_config else None,
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
# Only successful episodes get distilled, so the useful target is a success count.
# max_attempts<=0 means no cap: keep going until target_successes is reached.
target_successes = int(args_cli.target_successes)
attempt_cap = int(args_cli.max_attempts) if args_cli.max_attempts > 0 else None
successes = 0
try:
    ep = 0
    while True:
        if target_successes > 0:
            if successes >= target_successes:
                break
            if attempt_cap is not None and ep >= attempt_cap:
                break
            label = f"attempt {ep + 1} (success {successes}/{target_successes})"
        else:
            if ep >= args_cli.num_episodes:
                break
            label = f"Episode {ep + 1}/{args_cli.num_episodes}"
        episode_index = ep + int(args_cli.episode_index_offset)
        t0 = time.perf_counter()
        print(f"\n[INFO] ===== {label} =====", flush=True)
        result = run_episode(episode_index)
        successes += bool(result["success"])
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
            f"[INFO] Episode {episode_index}: success={result['success']} steps={result['steps']} {joint_text} "
            f"| successes={successes}/{ep + 1} ({successes / (ep + 1) * 100:.0f}%)",
            flush=True,
        )
        ep += 1
        write_summary(output_dir, episode_results)
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
