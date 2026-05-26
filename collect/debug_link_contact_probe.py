#!/usr/bin/env python3
"""Debug link_1 contact sampling (Phase C).

Samples mesh contacts, filters by scene.usd workspace, moves arm to top contact approach.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_COLLECT_DIR = Path(__file__).resolve().parent
if str(_COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(_COLLECT_DIR))

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
from task_registry import get_task_preset, list_task_presets

parser = argparse.ArgumentParser(description="Debug link_1 contact sampling and top-contact probe.")
parser.add_argument("--task_id", type=str, default="close_laptop_lid")
parser.add_argument("--list_tasks", action="store_true")
parser.add_argument("--task_config", type=str, default=None)
parser.add_argument("--probe_steps", type=int, default=400)
parser.add_argument("--hold_steps", type=int, default=120)
parser.add_argument("--max_draw_contacts", type=int, default=24)
parser.add_argument("--no_filter", action="store_true", help="Skip workspace filter (rank raw only).")
parser.add_argument("--no_move", action="store_true", help="Only print/draw; skip arm motion.")
parser.add_argument(
    "--mode",
    choices=("top_contact", "link_axis", "yaml_handle_contact", "yaml_handle_push", "articulation_push"),
    default="yaml_handle_contact",
    help=(
        "yaml_handle_contact: reach yaml handle, print report, then wait for Ctrl+C; "
        "yaml_handle_push: approach + contact + hinge close; "
        "top_contact: move to best mesh approach; "
        "link_axis: legacy +Z offset probe; "
        "articulation_push: legacy HDF5 touch reference."
    ),
)
parser.add_argument("--local_offset", type=float, nargs=3, default=(0.0, 0.0, 0.2))
add_randomization_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.list_tasks:
    for preset in list_task_presets():
        print(f"  - {preset.task_id}: {preset.description}")
    raise SystemExit(0)

task_preset = get_task_preset(args_cli.task_id)
task_config_path = Path(args_cli.task_config).expanduser().resolve() if args_cli.task_config else None
task_interaction = load_task_interaction_config(task_preset.task_id, task_config_path)
if not task_interaction.sampling or not task_interaction.push:
    raise ValueError("Task yaml must define sampling and push sections.")

rand_config = attach_joint_initial_baseline(
    randomization_config_from_args(args_cli, task_preset),
    task_preset,
    joint_prim=task_interaction.joint_prim or None,
)

args_cli = apply_camera_launch_workarounds(args_cli)
if getattr(args_cli, "headless", False):
    print("[WARN] Headless: debug lines will not be visible. Prefer --livestream 2 for GUI.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from env_setup import (
    build_environment_module_config,
    build_piper_robot_cfg,
    build_scene_hinge_cfg,
    compute_control_loop_timing,
    resolve_robot_handles,
)
from episode_collector import OfficialEpisodeCollector
from grasp_sampler import (
    candidate_world_geometry,
    compute_link_local_probe_point,
    link_local_axes_world,
    link_to_world_candidate,
)
from reference.contact_reference import approach_from_contact
from interaction_executor import PushInteractionExecutor
from isaaclab_env_module import IsaacLabEnvironmentModule

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

gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)
gripper_open_target = torch.tensor([[0.035, -0.035]], dtype=torch.float32, device=device)

env_module.define_camera_prims()
env_module.create_sensor_cameras()

collector = OfficialEpisodeCollector(
    dataset_file=task_preset.dataset_file,
    env_name=task_preset.env_name,
    num_demos=1,
)

diff_ik_controller = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="pose", ik_method="dls", ik_params={"lambda_val": 0.1}),
    num_envs=1,
    device=device,
)
diff_ik_pos_controller = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="position", ik_method="dls", ik_params={"lambda_val": 0.1}),
    num_envs=1,
    device=device,
)

executor = PushInteractionExecutor(
    robot=robot,
    env_module=env_module,
    handles=handles,
    timing=timing,
    task_config=task_interaction,
    sampling_config=task_interaction.sampling,
    push_config=task_interaction.push,
    success_specs=task_preset.rollout_success_specs,
    device=device,
    diff_ik_controller=diff_ik_controller,
    diff_ik_pos_controller=diff_ik_pos_controller,
    gripper_open_target=gripper_open_target,
    gripper_close_target=gripper_close_target,
)

draw_interface = None
if not getattr(args_cli, "headless", False):
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        print("[INFO] Debug draw enabled.")
    except Exception as exc:
        print(f"[WARN] Debug draw unavailable: {exc}")


def reset_scene() -> None:
    sim.reset()
    robot.reset()
    if env_module.scene_articulation is not None:
        env_module.scene_articulation.reset()
    env_module.capture_scene_root_baseline()
    sample = sample_randomization(rand_config, np.random.default_rng(rand_config.seed))
    apply_randomization_sample(
        env_module,
        rand_config,
        sample,
        joint_prim=task_interaction.joint_prim or None,
    )
    print(f"[INFO] Randomization: {format_randomization_sample(sample)}")
    env_module.sync_scene_joints_after_sim_reset()
    env_module.reset_robot_pose_via_targets(
        gripper_targets=gripper_close_target,
        gripper_joint_ids=handles.gripper_joint_ids,
    )


def _draw_segment(start: np.ndarray, end: np.ndarray, color: list[float], thickness: float = 4.0) -> None:
    if draw_interface is None:
        return
    draw_interface.draw_lines([start.tolist()], [end.tolist()], [color], [thickness])


def draw_link_debug(
    link_pos_np: np.ndarray,
    link_quat: tuple[float, float, float, float],
    probe_w: np.ndarray,
    candidates,
    top_candidate=None,
) -> None:
    if draw_interface is None:
        return
    draw_interface.clear_lines()
    axes = link_local_axes_world(link_quat)
    axis_len = 0.12
    colors = {
        "x": [1.0, 0.2, 0.2, 1.0],
        "y": [0.2, 1.0, 0.2, 1.0],
        "z": [0.2, 0.4, 1.0, 1.0],
        "probe": [1.0, 0.0, 1.0, 1.0],
        "contact": [1.0, 1.0, 0.0, 1.0],
        "approach": [0.2, 1.0, 1.0, 1.0],
        "top": [1.0, 0.2, 1.0, 1.0],
    }
    for name, axis in axes.items():
        end = link_pos_np + axis * axis_len
        _draw_segment(link_pos_np, end, colors[name], thickness=5.0)
    _draw_segment(link_pos_np, probe_w, colors["probe"], thickness=8.0)

    for candidate in candidates[: args_cli.max_draw_contacts]:
        approach_w, contact_w, _ = link_to_world_candidate(candidate, link_pos_np, link_quat)
        _draw_segment(link_pos_np, contact_w, colors["contact"], thickness=2.0)
        _draw_segment(contact_w, approach_w, colors["approach"], thickness=2.0)

    if top_candidate is not None:
        geom = candidate_world_geometry(top_candidate, link_pos_np, link_quat)
        contact_w = np.asarray(geom["contact_w"])
        approach_w = np.asarray(geom["approach_w"])
        _draw_segment(link_pos_np, contact_w, colors["top"], thickness=6.0)
        _draw_segment(contact_w, approach_w, colors["top"], thickness=5.0)


def draw_articulation_debug(probe_result: dict[str, object]) -> None:
    if draw_interface is None:
        return
    draw_interface.clear_lines()

    planned_contact = np.asarray(probe_result["planned_contact_w"], dtype=np.float64)
    actual_contact = np.asarray(probe_result["actual_contact_w"], dtype=np.float64)
    approach_w = np.asarray(probe_result["approach_w"], dtype=np.float64)
    hinge_origin = np.asarray(probe_result["hinge_origin_w"], dtype=np.float64)
    hinge_axis = np.asarray(probe_result["hinge_axis_w"], dtype=np.float64)
    close_poses = probe_result.get("close_poses") or []

    axis = hinge_axis / max(float(np.linalg.norm(hinge_axis)), 1e-9)
    hinge_len = 0.18
    _draw_segment(hinge_origin - axis * hinge_len, hinge_origin + axis * hinge_len, [1.0, 0.5, 0.0, 1.0], 6.0)
    _draw_segment(hinge_origin, planned_contact, [1.0, 1.0, 0.0, 1.0], 4.0)
    _draw_segment(planned_contact, approach_w, [0.2, 1.0, 1.0, 1.0], 5.0)
    _draw_segment(planned_contact, actual_contact, [1.0, 0.2, 0.2, 1.0], 6.0)

    prev = actual_contact
    for pos, _quat in close_poses[:: max(1, len(close_poses) // 40)]:
        pos_np = np.asarray(pos, dtype=np.float64)
        _draw_segment(prev, pos_np, [0.2, 1.0, 0.4, 1.0], 3.0)
        prev = pos_np


LAST_PROBE_RESULT: dict[str, object] | None = None
LAST_PROBE_MODE: str | None = None


def main_loop() -> bool:
    """Run probe. Returns True to skip the post-probe hold loop (unused for yaml_handle_contact)."""
    global LAST_HANDLE_CONTACT_REPORT, LAST_PROBE_RESULT, LAST_PROBE_MODE
    reset_scene()
    collector.reset_episode()

    link_prim = task_interaction.link_prim
    link_pos, link_quat = env_module.get_prim_world_pose_wxyz(link_prim)
    link_pos_np = np.asarray(link_pos, dtype=np.float64)
    ee_pos = robot.data.body_pos_w[0, handles.ee_body_id].detach().cpu().numpy()

    if args_cli.mode == "yaml_handle_contact":
        if args_cli.no_move:
            contact_w, quat_w, hinge_lever_m = executor._resolve_yaml_handle_world()
            approach_w = approach_from_contact(
                contact_w,
                quat_w,
                float(task_interaction.push.approach_backoff_m),
            )
            report = executor._summarize_handle_contact(
                contact_w, quat_w, approach_w, hinge_lever_m=hinge_lever_m
            )
            executor.print_handle_contact_report(report, label="yaml_handle (no_move)")
            LAST_HANDLE_CONTACT_REPORT = report
            draw_articulation_debug(
                {
                    "planned_contact_w": contact_w,
                    "actual_contact_w": report["actual_ee_pos_w"],
                    "approach_w": approach_w,
                    "close_poses": [],
                    "hinge_origin_w": env_module.get_hinge_world_frame(
                        link_prim,
                        task_interaction.sampling.hinge.origin,
                        task_interaction.sampling.hinge.axis,
                    )[0],
                    "hinge_axis_w": env_module.get_hinge_world_frame(
                        link_prim,
                        task_interaction.sampling.hinge.origin,
                        task_interaction.sampling.hinge.axis,
                    )[1],
                }
            )
        else:
            probe_result = executor.run_yaml_handle_contact_only_probe(
                collector,
                max_servo_steps=int(args_cli.probe_steps),
            )
            LAST_HANDLE_CONTACT_REPORT = probe_result
            draw_articulation_debug(probe_result)
        print("[INFO] Yellow: planned contact. Red: actual EE. Cyan: approach retreat.")
        print("[INFO] Handle reach report printed. Press Ctrl+C to exit.", flush=True)
        return False

    if args_cli.mode in ("yaml_handle_push", "articulation_push"):
        if args_cli.mode == "articulation_push":
            print("[WARN] articulation_push is legacy; prefer --mode yaml_handle_push.")
            print("[INFO] Loading touch contact reference...", flush=True)
            executor.preload_articulation_contact_reference()
            contact_w, quat_w, ref = executor._resolve_touch_contact_world()
            hinge_lever_m = ref.hinge_lever_m
        else:
            contact_w, quat_w, hinge_lever_m = executor._resolve_yaml_handle_world()
        approach_w = approach_from_contact(
            contact_w,
            quat_w,
            float(task_interaction.push.approach_backoff_m),
        )
        print(f"[INFO] Handle reference: hinge_lever={hinge_lever_m:.4f}m")

        if not args_cli.no_move:
            if args_cli.mode == "articulation_push":
                probe_result = executor.run_articulation_calibrated_probe(
                    collector,
                    max_servo_steps=int(args_cli.probe_steps),
                    hold_control_steps=int(args_cli.hold_steps),
                )
            else:
                probe_result = executor.run_yaml_handle_probe(
                    collector,
                    max_servo_steps=int(args_cli.probe_steps),
                    hold_control_steps=int(args_cli.hold_steps),
                )
            LAST_PROBE_RESULT = probe_result
            LAST_PROBE_MODE = args_cli.mode
            draw_articulation_debug(probe_result)
            joint_deg = (probe_result.get("joint_degs") or {}).get(task_interaction.joint_prim)
            drift = probe_result.get("contact_drift_m")
            drift_s = f"{float(drift):.4f}m" if drift is not None else "n/a"
            print(
                f"[INFO] Push+close result: success={probe_result.get('success')} "
                f"contact_drift={drift_s} "
                f"joint_1={joint_deg}deg close_wps={len(probe_result.get('close_poses') or [])}",
                flush=True,
            )
        else:
            hinge_origin_w, hinge_axis_w = env_module.get_hinge_world_frame(
                link_prim,
                task_interaction.sampling.hinge.origin,
                task_interaction.sampling.hinge.axis,
            )
            draw_articulation_debug(
                {
                    "planned_contact_w": contact_w,
                    "actual_contact_w": contact_w,
                    "approach_w": approach_w,
                    "close_poses": executor._build_hinge_close_poses(contact_w, quat_w),
                    "hinge_origin_w": hinge_origin_w,
                    "hinge_axis_w": hinge_axis_w,
                }
            )

        print("[INFO] Yellow: planned contact. Red: actual contact drift. Green: close trajectory.")
        print(
            "[INFO] Motion phases finished (approach + contact + close). "
            "Process stays alive until Ctrl+C (close ended ≠ program exit).",
            flush=True,
        )
        return False

    else:
        candidates = executor.log_contact_candidate_preview(
            max_rows=16, apply_filters=not args_cli.no_filter
        )

        top = candidates[0] if candidates else None
        if top is not None:
            geom = candidate_world_geometry(top, link_pos_np, link_quat)
            probe_w = np.asarray(geom["approach_w"], dtype=np.float64)
        else:
            probe_w = compute_link_local_probe_point(
                link_pos_np, link_quat, tuple(float(v) for v in args_cli.local_offset)
            )

        draw_link_debug(link_pos_np, link_quat, probe_w, candidates, top_candidate=top)

        if not args_cli.no_move:
            if args_cli.mode == "link_axis":
                executor.run_link_local_axis_probe(
                    collector,
                    local_offset=tuple(float(v) for v in args_cli.local_offset),
                    max_servo_steps=int(args_cli.probe_steps),
                    hold_control_steps=int(args_cli.hold_steps),
                )
            else:
                executor.run_top_contact_probe(
                    collector,
                    max_servo_steps=int(args_cli.probe_steps),
                    hold_control_steps=int(args_cli.hold_steps),
                )
            draw_link_debug(link_pos_np, link_quat, probe_w, candidates, top_candidate=top)

        print("[INFO] Magenta: top contact -> approach. Yellow/cyan: other ranked contacts.")
        print("[INFO] Hold scene open for visual check.")

    return False


LAST_HANDLE_CONTACT_REPORT: dict[str, object] | None = None


try:
    main_loop()
    print(
        "[INFO] Motion complete. Simulation idle — press Ctrl+C to exit (no auto-quit).",
        flush=True,
    )
    while True:
        render = bool(simulation_app.is_running())
        sim.step(render=render)
        robot.update(sim.cfg.dt)
        if env_module.scene_articulation is not None:
            env_module.scene_articulation.update(sim.cfg.dt)
except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C received.", flush=True)
except Exception as exc:
    print(f"\n[ERROR] Idle loop exited: {exc}", flush=True)
    raise
finally:
    if LAST_PROBE_RESULT is not None and LAST_PROBE_MODE in ("yaml_handle_push", "articulation_push"):
        executor.print_push_probe_exit_summary(LAST_PROBE_RESULT)
    elif LAST_HANDLE_CONTACT_REPORT is not None:
        executor.print_handle_contact_exit_summary(LAST_HANDLE_CONTACT_REPORT)
    simulation_app.close()
