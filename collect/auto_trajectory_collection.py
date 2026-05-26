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

from domain_randomization import (
    add_randomization_cli_args,
    apply_randomization_sample,
    attach_joint_initial_baseline,
    format_randomization_sample,
    randomization_config_from_args,
    sample_randomization,
)
from isaaclab_env_module import apply_camera_launch_workarounds
from task_config import load_task_interaction_config
from task_registry import get_task_preset, list_task_presets, PHYSVLA_ASSETS_DIR

parser = argparse.ArgumentParser(description="Auto trajectory collection (Isaac Lab + Piper).")
parser.add_argument("--task_id", type=str, default=None, help="Task preset from task_registry.")
parser.add_argument("--list_tasks", action="store_true", help="List task presets and exit.")
parser.add_argument("--num_demos", type=int, default=5, help="Target number of successful saved demos.")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=None,
    help="Max total attempts (including failures). Default: max(50, num_demos * 10).",
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
    "--save_failed",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Write failed attempts to HDF5 (success=False) for offline video review.",
)
parser.add_argument(
    "--home_reset_steps",
    type=int,
    default=40,
    help="Control steps for arm home reset segment after success.",
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
    max_attempts = max(50, int(args_cli.num_demos) * 10)

task_preset = get_task_preset(args_cli.task_id)
if args_cli.usd_path:
    p = Path(args_cli.usd_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    task_preset = replace(task_preset, usd_path=str(p))

task_config_path = Path(args_cli.task_config).expanduser().resolve() if args_cli.task_config else None
task_interaction = load_task_interaction_config(task_preset.task_id, task_config_path)
rollout_success_specs = task_preset.rollout_success_specs

push_strategy = (
    task_interaction.push.push_strategy
    if task_interaction.push
    else "articubot"
)

# Auto collection: no runtime reference HDF5 (no replay, no live touch-HDF5 load).
_AUTO_COLLECT_PUSH_STRATEGIES = frozenset({"yaml_handle"})
if task_interaction.push and task_interaction.push.debug_hardcoded_push:
    raise ValueError(
        "auto_trajectory_collection forbids push.debug_hardcoded_push "
        "(reference HDF5 replay). Calibrate handle offline via "
        "scripts/inspect_touch_hdf5.py → task yaml."
    )
if push_strategy not in _AUTO_COLLECT_PUSH_STRATEGIES:
    raise ValueError(
        f"auto_trajectory_collection only supports push_strategy in "
        f"{sorted(_AUTO_COLLECT_PUSH_STRATEGIES)}; got '{push_strategy}'. "
        "keyboard_aligned / articulation_calibrated / articubot replay reference HDF5 "
        "at runtime — use debug_link_contact_probe for those legacy paths."
    )
if task_interaction.push and (
    task_interaction.push.keyboard_reference_hdf5
    or task_interaction.push.debug_reference_hdf5
):
    raise ValueError(
        "auto_trajectory_collection: remove push.keyboard_reference_hdf5 and "
        "push.debug_reference_hdf5 from task yaml. Touch handle is calibrated offline "
        "only; runtime uses push_contact_offset_link / contact_quat_link."
    )

rand_config = attach_joint_initial_baseline(
    randomization_config_from_args(args_cli, task_preset),
    task_preset,
    joint_prim=task_interaction.joint_prim or None,
)
rng = random.Random(rand_config.seed)

_scene_usd = Path(task_preset.usd_path)
if not _scene_usd.is_file():
    raise FileNotFoundError(f"Scene USD missing: {_scene_usd}")

print(f"[INFO] Task: {task_preset.task_id}")
print(f"[INFO] Interaction mode: {task_interaction.interaction_mode}")
print(f"[INFO] Target successful demos: {args_cli.num_demos}, max attempts: {max_attempts}")
print(f"[INFO] Save failed attempts: {args_cli.save_failed}")
print(
    "[INFO] Push mode: yaml_handle (link-local yaml only; no reference HDF5 at runtime)"
)
print(f"[INFO] Assets root: {PHYSVLA_ASSETS_DIR.resolve()}")

args_cli = apply_camera_launch_workarounds(args_cli)
if not getattr(args_cli, "headless", False):
    print("[WARN] Recommend --headless for batch auto collection.")

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
from isaaclab_env_module import IsaacLabEnvironmentModule, SCENE_JOINT_PHYSICS_WARMUP_STEPS

env_module = IsaacLabEnvironmentModule(build_environment_module_config(task_preset))
sim = env_module.create_simulation(dt=1 / 60.0, render_interval=4)

robot = env_module.create_robot(build_piper_robot_cfg(task_preset))
scene_hinge_cfg = build_scene_hinge_cfg(task_preset)
if scene_hinge_cfg is not None:
    env_module.create_scene_articulation(scene_hinge_cfg)
    print("[INFO] Scene articulation enabled for live hinge readback (same as pi05 rollout).")
env_module.initialize_robot_home_pose()

device = sim.device
handles = resolve_robot_handles(robot)
timing = compute_control_loop_timing(task_preset, float(sim.cfg.dt))

gripper_open_target = torch.tensor([[0.035, -0.035]], dtype=torch.float32, device=device)
gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)

env_module.define_camera_prims()
sensor_cameras = env_module.create_sensor_cameras()
for cam_name in ("main", "wrist"):
    if cam_name in sensor_cameras:
        print(f"[INFO] Sensor camera ready: {cam_name}")

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
    sim.reset()
    robot.reset()
    if env_module.scene_articulation is not None:
        env_module.scene_articulation.reset()
    if sensor_cameras:
        for sensor in sensor_cameras.values():
            sensor.reset()

    env_module.capture_scene_root_baseline()
    sample = sample_randomization(rand_config, rng)
    apply_randomization_sample(
        env_module,
        rand_config,
        sample,
        joint_prim=task_interaction.joint_prim or None,
    )
    print(f"[INFO] Randomization: {format_randomization_sample(sample)}")

    env_module.sync_scene_joints_after_sim_reset(warmup_steps=SCENE_JOINT_PHYSICS_WARMUP_STEPS)
    env_module.reset_robot_pose_via_targets(
        gripper_targets=gripper_close_target,
        gripper_joint_ids=handles.gripper_joint_ids,
    )

    env_module.define_camera_prims()


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
            reset_episode_scene()
            capture_initial_state()

            t0 = time.perf_counter()
            success, final_joint_degs = executor.run_yaml_handle_push(
                collector,
                episode_start_wall_time=t0,
            )

            if success:
                print(f"[INFO] Push success: {final_joint_degs}")
                executor.run_home_reset(collector)
                saved, demo_key, num_steps = export_episode_final_step(success=True)
                if saved:
                    successful_demos += 1
                    print(
                        f"[INFO] Saved {demo_key}: T={num_steps}, success=True "
                        f"({successful_demos}/{args_cli.num_demos})"
                    )
                else:
                    print("[WARN] Success but no recorded steps; skipping export.")
                    collector.reset_episode()
            elif args_cli.save_failed:
                saved, demo_key, num_steps = export_episode_final_step(success=False)
                if saved:
                    failed_exports += 1
                    print(
                        f"[INFO] Saved {demo_key}: T={num_steps}, success=False "
                        f"(joint={final_joint_degs}, failed_exports={failed_exports})"
                    )
                else:
                    print(f"[WARN] Push failed with no recorded steps: {final_joint_degs}")
                    collector.reset_episode()
            else:
                print(f"[INFO] Push failed: {final_joint_degs} (discard, --no-save_failed).")
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
    print(f"[INFO] Dataset: {collector.dataset_file}")
    print("[INFO] Closing simulation app...")
    simulation_app.close()
