#!/usr/bin/env python3
"""Automatic trajectory collection for pi0.5 training (ArticuBot-style MVP).

Runtime policy: auto collection must NOT read reference HDF5 for control.
Handle/contact come only from task yaml (link-local offsets calibrated offline
via scripts/inspect_touch_hdf5.py). Output demos are still written as HDF5.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

_COLLECT_DIR = Path(__file__).resolve().parent
if str(_COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(_COLLECT_DIR))

import torch
from isaaclab.app import AppLauncher

from domain_randomization_robotwin2 import (
    add_randomization_cli_args,
    apply_randomization_sample,
    format_randomization_sample,
    randomization_config_from_args,
    sample_randomization,
)
from isaaclab_env_module import apply_camera_launch_workarounds
from task_config import load_task_interaction_config
from task_registry import get_task_preset, list_task_presets

parser = argparse.ArgumentParser(description="Auto trajectory collection (Isaac Lab + Piper).")
parser.add_argument("--task_id", type=str, default=None, help="Task preset from task_registry.")
parser.add_argument("--list_tasks", action="store_true", help="List task presets and exit.")
parser.add_argument("--num_demos", type=int, default=5, help="Target number of successful saved demos.")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=None,
    help="Max total attempts (including failures). Default: num_demos + 200.",
)
parser.add_argument(
    "--task_config",
    type=str,
    default=None,
    help="Override path to task_configs/<task_id>.yaml.",
)
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="Override TaskPreset.usd_path.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help="Override TaskPreset.dataset_file for isolated worker output.",
)
parser.add_argument(
    "--flat_dataset_session_dir",
    action="store_true",
    help="Write timestamped HDF5 directly under dataset_file's parent directory instead of a stem subdirectory.",
)
parser.add_argument(
    "--save_failed",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Write failed attempts to HDF5 (success=False) for offline video review. Default: discard.",
)
parser.add_argument(
    "--home_reset_steps",
    type=int,
    default=40,
    help="Control steps for arm home reset segment after success.",
)
parser.add_argument(
    "--episode_step_limit",
    type=int,
    default=600,
    help="Hard per-attempt control-step limit. Reaching it marks the attempt failed.",
)
parser.add_argument(
    "--debug-logs",
    action="store_true",
    dest="debug_logs",
    help="Print per-step planning/close debug logs (default: quiet).",
)
parser.add_argument(
    "--no-health-checks",
    action="store_true",
    help="Disable post-reset / per-step / pre-export sanity checks.",
)
parser.add_argument(
    "--reset_health_retries",
    type=int,
    default=3,
    help="Re-run reset_episode_scene when post-reset health check fails.",
)
parser.add_argument(
    "--post_reset_warmup_sec",
    type=float,
    default=1.0,
    help="Extra wall-clock/render warmup after each reset before recording starts.",
)
parser.add_argument(
    "--fatal_on_wrist_mount_error",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit the process if wrist camera mount validation fails; wrapper should restart.",
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

max_attempts = args_cli.max_attempts
if max_attempts is None:
    max_attempts = int(args_cli.num_demos) + 200

task_preset = get_task_preset(args_cli.task_id)
if args_cli.usd_path:
    p = Path(args_cli.usd_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    task_preset = replace(task_preset, usd_path=str(p))
if args_cli.dataset_file:
    p = Path(args_cli.dataset_file).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    task_preset = replace(task_preset, dataset_file=str(p))

task_config_path = Path(args_cli.task_config).expanduser().resolve() if args_cli.task_config else None
task_interaction = load_task_interaction_config(task_preset.task_id, task_config_path)
rollout_success_specs = task_preset.rollout_success_specs

push_strategy = (
    task_interaction.push.push_strategy
    if task_interaction.push
    else "yaml_handle"
)

# Auto collection: no runtime reference HDF5 (no replay, no live touch-HDF5 load).
_AUTO_COLLECT_PUSH_STRATEGIES = frozenset({"yaml_handle"})
if push_strategy not in _AUTO_COLLECT_PUSH_STRATEGIES:
    raise ValueError(
        f"auto_trajectory_collection only supports push_strategy in "
        f"{sorted(_AUTO_COLLECT_PUSH_STRATEGIES)}; got '{push_strategy}'. "
        "Use debug_link_contact_probe for articulation_calibrated / mesh sampling."
    )
if task_interaction.push and task_interaction.push.keyboard_reference_hdf5:
    raise ValueError(
        "auto_trajectory_collection: remove push.keyboard_reference_hdf5 from task yaml. "
        "Touch handle is calibrated offline only; runtime uses "
        "push_contact_offset_link / contact_quat_link."
    )

rand_config = randomization_config_from_args(args_cli, task_preset)
rng = random.Random(rand_config.seed)

_scene_usd = Path(task_preset.usd_path)
if not _scene_usd.is_file():
    raise FileNotFoundError(f"Scene USD missing: {_scene_usd}")

print(
    f"[INFO] Auto collect: task={task_preset.task_id} "
    f"target={args_cli.num_demos} max_attempts={max_attempts} "
    f"save_failed={args_cli.save_failed}"
)
print(
    f"[INFO] Episode step limit: control_steps<={int(args_cli.episode_step_limit)} "
    "(timeout => failed attempt)"
)
print(
    f"[INFO] Health checks: enabled={not bool(args_cli.no_health_checks)} "
    f"reset_retries={int(args_cli.reset_health_retries)}"
)

args_cli = apply_camera_launch_workarounds(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab / Omniverse imports must come after AppLauncher.
import numpy as np
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from env_setup import (
    build_environment_module_config,
    build_piper_robot_cfg,
    build_scene_hinge_cfg,
    compute_control_loop_timing,
    resolve_robot_handles,
)
from episode_collector import OfficialEpisodeCollector
from interaction_executor import PushInteractionExecutor
from collection_health import (
    HealthLimits,
    check_robot_state,
    check_scene_joint_angle_deg,
)
from isaaclab_env_module import IsaacLabEnvironmentModule, SCENE_JOINT_PHYSICS_WARMUP_STEPS

health_limits = HealthLimits.from_task_preset(task_preset)
health_checks_enabled = not bool(args_cli.no_health_checks)
fatal_exit_requested = False
FATAL_RESTART_EXIT_CODE = 75
WRIST_CAMERA_PARENT_PATH = "/World/piper_description/gripper_base"

env_module = IsaacLabEnvironmentModule(
    build_environment_module_config(task_preset, quiet_logging=not args_cli.debug_logs)
)
sim = env_module.create_simulation(dt=1 / 60.0, render_interval=4)

robot = env_module.create_robot(build_piper_robot_cfg(task_preset))
scene_hinge_cfg = build_scene_hinge_cfg(task_preset)
if scene_hinge_cfg is not None:
    env_module.create_scene_articulation(scene_hinge_cfg)
env_module.initialize_robot_home_pose()

device = sim.device
handles = resolve_robot_handles(robot)
timing = compute_control_loop_timing(task_preset, float(sim.cfg.dt))

gripper_open_target = torch.tensor([[0.035, -0.035]], dtype=torch.float32, device=device)
gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)

env_module.define_camera_prims()
sensor_cameras = env_module.create_sensor_cameras()

instruction_text = (task_preset.language_instruction or "").strip()
if not instruction_text:
    raise ValueError(f"TaskPreset '{task_preset.task_id}' has empty language_instruction.")
instruction_bytes = instruction_text.encode("utf-8")
language_instruction_tensor = torch.tensor(
    list(instruction_bytes), dtype=torch.uint8, device=device
).unsqueeze(0)
language_instruction_length = torch.tensor([len(instruction_bytes)], dtype=torch.int32, device=device)

collector = OfficialEpisodeCollector(
    dataset_file=task_preset.dataset_file,
    env_name=task_preset.env_name,
    num_demos=args_cli.num_demos,
    health_limits=health_limits,
    health_checks_enabled=health_checks_enabled,
    session_subdir=not bool(args_cli.flat_dataset_session_dir),
)

if not task_interaction.sampling or not task_interaction.push:
    raise ValueError(f"Task config for '{task_preset.task_id}' must define 'sampling' and 'push' sections.")

joint_upper_limit_deg = 104.0
for spec in task_preset.joint_limit_specs:
    if spec.prim_path == task_interaction.joint_prim and spec.upper_limit is not None:
        joint_upper_limit_deg = float(spec.upper_limit)

np_rng = np.random.default_rng(rand_config.seed)

diff_ik_cfg = DifferentialIKControllerCfg(
    command_type="pose",
    ik_method="dls",
    ik_params={"lambda_val": 0.1},
)
diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)
diff_ik_pos_cfg = DifferentialIKControllerCfg(
    command_type="position",
    ik_method="dls",
    ik_params={"lambda_val": 0.1},
)
diff_ik_pos_controller = DifferentialIKController(diff_ik_pos_cfg, num_envs=1, device=device)

if task_interaction.interaction_mode == "push":
    executor = PushInteractionExecutor(
        robot=robot,
        env_module=env_module,
        handles=handles,
        timing=timing,
        task_config=task_interaction,
        sampling_config=task_interaction.sampling,
        push_config=task_interaction.push,
        success_specs=rollout_success_specs,
        device=device,
        diff_ik_controller=diff_ik_controller,
        diff_ik_pos_controller=diff_ik_pos_controller,
        gripper_open_target=gripper_open_target,
        gripper_close_target=gripper_close_target,
        joint_upper_limit_deg=joint_upper_limit_deg,
        rng=np_rng,
        home_steps=args_cli.home_reset_steps,
        verbose=bool(args_cli.debug_logs),
        health_limits=health_limits,
        health_checks_enabled=health_checks_enabled,
    )
else:
    raise NotImplementedError(
        f"interaction_mode={task_interaction.interaction_mode} not implemented in MVP."
    )

def capture_initial_state() -> None:
    initial_state = {
        "robot_root_state": robot.data.root_state_w[:, :13].clone(),
        "robot_joint_pos": robot.data.joint_pos.clone(),
        "robot_joint_vel": robot.data.joint_vel.clone(),
        "language_instruction_utf8": language_instruction_tensor.clone(),
        "language_instruction_length": language_instruction_length.clone(),
    }
    collector.set_initial_state(initial_state)


def reset_episode_scene() -> None:
    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: begin", flush=True)
    episode_preset = get_task_preset(args_cli.task_id)
    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: apply scene root", flush=True)
    env_module.apply_task_preset_scene_root(episode_preset)
    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: apply joint initial USD", flush=True)
    env_module.apply_task_preset_joint_initial(episode_preset)

    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: before sim.reset", flush=True)
    sim.reset()
    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: after sim.reset; before robot.reset", flush=True)
    robot.reset()
    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: after robot.reset", flush=True)
    if sensor_cameras:
        if args_cli.debug_logs:
            print("[TRACE] reset_episode_scene: reset sensor cameras", flush=True)
        for sensor in sensor_cameras.values():
            sensor.reset()

    env_module.ensure_scene_root_baseline()

    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: define default camera prims", flush=True)
    env_module.define_camera_prims()

    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: sample/apply RobotWin2 DR", flush=True)
    sample = sample_randomization(rand_config, rng)
    apply_randomization_sample(
        env_module,
        rand_config,
        sample,
        verbose=bool(args_cli.debug_logs),
    )

    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: sync scene joints after reset", flush=True)
    env_module.sync_scene_joints_after_sim_reset(warmup_steps=SCENE_JOINT_PHYSICS_WARMUP_STEPS)
    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: reset robot pose via targets", flush=True)
    env_module.reset_robot_pose_via_targets(
        gripper_targets=gripper_close_target,
        gripper_joint_ids=handles.gripper_joint_ids,
    )

    warmup_sec = max(0.0, float(args_cli.post_reset_warmup_sec))
    if warmup_sec > 0.0:
        deadline = time.perf_counter() + warmup_sec
        while time.perf_counter() < deadline:
            sim.render()
            robot.update(sim.cfg.dt)
            if env_module.scene_articulation is not None:
                env_module.scene_articulation.update(sim.cfg.dt)

    scene_spec = episode_preset.scene_root_specs[0] if episode_preset.scene_root_specs else None
    joint_spec = episode_preset.joint_initial_specs[0] if episode_preset.joint_initial_specs else None
    scene_xyz = tuple(round(float(v), 4) for v in scene_spec.translation) if scene_spec else None
    joint_target = float(joint_spec.position) if joint_spec else float("nan")
    joint_sim = (
        env_module.read_scene_joint_angle_deg(joint_spec.prim_path)
        if joint_spec is not None
        else None
    )
    joint_sim_s = f"{float(joint_sim):.2f}°" if joint_sim is not None else "n/a"
    print(
        "[INFO] Attempt init: "
        f"episode_step_limit={int(args_cli.episode_step_limit)} "
        f"scene_xyz={scene_xyz} joint_target={joint_target:.2f}° "
        f"joint_sim={joint_sim_s} DR=({format_randomization_sample(sample)})",
        flush=True,
    )
    if args_cli.debug_logs:
        print("[TRACE] reset_episode_scene: end", flush=True)


def validate_episode_scene_health(
    *,
    joint_target_deg: float | None,
) -> tuple[bool, str]:
    if not health_checks_enabled:
        return True, ""

    robot_health = check_robot_state(robot, handles, health_limits)
    if not robot_health.ok:
        return False, robot_health.reason

    joint_spec = get_task_preset(args_cli.task_id).joint_initial_specs
    joint_prim = joint_spec[0].prim_path if joint_spec else None
    joint_sim = (
        env_module.read_scene_joint_angle_deg(joint_prim)
        if joint_prim is not None
        else None
    )
    joint_health = check_scene_joint_angle_deg(
        joint_sim,
        target_deg=joint_target_deg,
        limits=health_limits,
    )
    if not joint_health.ok:
        return False, joint_health.reason

    return True, ""


def reset_episode_scene_with_health() -> tuple[bool, str]:
    global fatal_exit_requested

    max_retries = max(1, int(args_cli.reset_health_retries))
    last_reason = ""
    for retry_idx in range(max_retries):
        reset_episode_scene()
        joint_target_deg = None
        if env_module.cfg.joint_initial_specs:
            joint_target_deg = float(env_module.cfg.joint_initial_specs[0].position)
        ok, reason = validate_episode_scene_health(joint_target_deg=joint_target_deg)
        if ok:
            return True, ""
        last_reason = reason
        print(
            f"[WARN] Post-reset health check failed "
            f"({retry_idx + 1}/{max_retries}): {reason}",
            flush=True,
        )
        if reason.startswith("FATAL_WRIST_CAMERA_MOUNT") and bool(args_cli.fatal_on_wrist_mount_error):
            fatal_exit_requested = True
            return False, last_reason
    return False, last_reason


def export_episode_final_step(success: bool) -> tuple[bool, str | None, int]:
    if executor.last_record is None:
        return False, None, 0
    done_flag = torch.ones((1,), device=device, dtype=torch.bool)
    rec = executor.last_record
    collector.add_step(rec.obs_dict, rec.action, rec.reward, done_flag, rec.state_dict)
    return collector.export_episode(success=success)


successful_demos = 0
failed_exports = 0
total_attempts = 0

try:
    while simulation_app.is_running() and successful_demos < args_cli.num_demos:
        if total_attempts >= max_attempts:
            print(f"[INFO] Reached max_attempts={max_attempts}. Stopping.")
            break

        total_attempts += 1
        print(f"\n[INFO] ===== Attempt {total_attempts} (saved {successful_demos}/{args_cli.num_demos}) =====")

        try:
            collector.reset_episode()
            executor.reset_recording_health()
            scene_ok, scene_reason = reset_episode_scene_with_health()
            if not scene_ok:
                print(
                    f"[WARN] Attempt {total_attempts}: skipping after reset health failures: "
                    f"{scene_reason}",
                    flush=True,
                )
                collector.reset_episode()
                if fatal_exit_requested:
                    print("[FATAL] Restart requested after unrecoverable wrist camera mount error.", flush=True)
                    break
                continue
            capture_initial_state()

            t0 = time.perf_counter()
            success, final_joint_degs = executor.run_yaml_handle_push(
                collector,
                episode_start_wall_time=t0,
                episode_step_limit=int(args_cli.episode_step_limit),
            )

            if success and executor.recording_health_failed:
                print(
                    f"[WARN] Attempt {total_attempts}: task succeeded but recording failed health "
                    f"check ({executor._recording_health_reason}); discarding episode.",
                    flush=True,
                )
                success = False

            if success:
                saved, demo_key, num_steps = export_episode_final_step(success=True)
                if saved:
                    successful_demos += 1
                    print(
                        f"[INFO] Attempt {total_attempts}: saved {demo_key} "
                        f"T={num_steps} control_steps={executor.control_step_count}/"
                        f"{int(args_cli.episode_step_limit)} joints={final_joint_degs} "
                        f"({successful_demos}/{args_cli.num_demos})",
                        flush=True,
                    )
                    if successful_demos >= args_cli.num_demos:
                        print(
                            f"[INFO] Target reached ({successful_demos}/{args_cli.num_demos}); "
                            "leaving collection loop.",
                            flush=True,
                        )
                        break
                else:
                    print("[WARN] Success but no recorded steps; skipping export.")
                    collector.reset_episode()
            elif args_cli.save_failed:
                saved, demo_key, num_steps = export_episode_final_step(success=False)
                if saved:
                    failed_exports += 1
                    print(
                        f"[INFO] Attempt {total_attempts}: saved {demo_key} "
                        f"T={num_steps} control_steps={executor.control_step_count}/"
                        f"{int(args_cli.episode_step_limit)} timeout={executor.episode_step_limit_hit} "
                        f"success=False joints={final_joint_degs}"
                    )
                else:
                    print(
                        f"[WARN] Failed with no recorded steps: "
                        f"control_steps={executor.control_step_count}/"
                        f"{int(args_cli.episode_step_limit)} "
                        f"timeout={executor.episode_step_limit_hit} joints={final_joint_degs}"
                    )
                    collector.reset_episode()
            else:
                if args_cli.debug_logs:
                    print(f"[INFO] Attempt {total_attempts} failed: {final_joint_degs}")
                collector.reset_episode()
        except Exception as exc:
            import traceback

            print(f"[ERROR] Attempt {total_attempts} failed: {exc}")
            traceback.print_exc()
            collector.reset_episode()
            if not simulation_app.is_running():
                print("[WARN] Simulation app stopped after error.")
                break

finally:
    if not simulation_app.is_running():
        print("[WARN] Exited collection loop: simulation_app.is_running() is False.")
    collector.close()
    print(
        f"[INFO] Auto collection finished: {successful_demos} success, "
        f"{failed_exports} failed saved, {total_attempts} attempts"
    )
    print(f"[INFO] Dataset: {collector.dataset_file}", flush=True)
    print("[INFO] Closing simulation app...", flush=True)
    simulation_app.close()
    if fatal_exit_requested:
        raise SystemExit(FATAL_RESTART_EXIT_CODE)
