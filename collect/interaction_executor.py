"""Push interaction via ArticuBot-style sampling + differential IK."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from grasp_sampler import (
    ContactCandidate,
    candidate_world_geometry,
    compute_link_local_probe_point,
    contact_passes_sanity,
    filter_candidates_by_approach_direction,
    link_local_axes_world,
    link_to_world_candidate,
    prepare_ranked_contact_candidates,
    rank_contact_candidates,
    sample_contact_candidates,
    scene_approach_direction,
    select_top_contact_candidate,
    summarize_candidates_world,
)
from reference.contact_reference import (
    TouchContactReference,
    approach_from_contact,
    hinge_lever_arm,
    load_touch_contact_from_hdf5,
    resolve_contact_pose_world,
    summarize_touch_reference,
)
from reference.handle_reference import derive_contact_quat_link, resolve_yaml_handle_world
from reference.opening_kinematics import (
    compute_articulation_ee_trajectory,
    compose_pose,
    invert_pose,
)
from recording_utils import RecordingContext, build_step_tensors, capture_rgb_if_due
from success_utils import evaluate_rollout_success, update_peak_joint_degs
from task_registry import SHARED_TELEOP_PIPER

if TYPE_CHECKING:
    from episode_collector import OfficialEpisodeCollector
    from env_setup import ControlLoopTiming, RobotHandles
    from grasp_sampler import SamplingConfig
    from isaaclab_env_module import IsaacLabEnvironmentModule
    from task_config import PushConfig, TaskInteractionConfig
    from task_registry import TaskRolloutSuccessSpec


@dataclass
class StepRecord:
    obs_dict: dict
    action: torch.Tensor
    reward: torch.Tensor
    state_dict: dict


# Keyboard teleop matched defaults (150830 demo: EE max step ~0.0067 m).
KEYBOARD_EE_POS_STEP_M = 0.005
KEYBOARD_JOINT_STEP_RAD = 0.02
POSITION_REACH_TOL_M = 0.012
POSE_REACH_TOL_M = 0.015
POSE_REACH_ROT_RAD = 0.15
# T_rel close waypoints are ~3–4 mm apart; use a tighter tol than approach/contact.
CLOSE_POSE_REACH_TOL_M = 0.005
CLOSE_POSE_REACH_ROT_RAD = 0.12
# Piper URDF: joint7 origin is 0.1358 m along gripper_base +Z (closed-gripper pad estimate).
GRIPPER_FINGER_ORIGIN_OFFSET_M = 0.1358
# Scene joint_1 USD scale (task_registry close_laptop): 15°≈laptop open 90°, 104°≈closed 0°.
# Control/gating/success all use USD readback; mapping below is log-only.
USD_JOINT_LID_OPEN_DEG = 15.0
USD_JOINT_LID_CLOSED_DEG = 104.0
REAL_LID_OPEN_DEG = 90.0
REAL_LID_CLOSED_DEG = 0.0
def usd_joint_to_real_lid_deg(usd_joint_deg: float) -> float:
    """Linear map for human-readable logs only (not used in IK/T_rel)."""
    span = USD_JOINT_LID_CLOSED_DEG - USD_JOINT_LID_OPEN_DEG
    if abs(span) < 1e-6:
        return usd_joint_deg
    t = (float(usd_joint_deg) - USD_JOINT_LID_OPEN_DEG) / span
    return REAL_LID_OPEN_DEG + t * (REAL_LID_CLOSED_DEG - REAL_LID_OPEN_DEG)


def _wxyz_to_rot(quat_wxyz: tuple[float, float, float, float]) -> R:
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w])


def _rot_to_wxyz(rot: R) -> tuple[float, float, float, float]:
    x, y, z, w = rot.as_quat()
    return (float(w), float(x), float(y), float(z))


def _slerp_wxyz(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return q0
    if t >= 1.0:
        return q1
    rot = _wxyz_to_rot(q0).inv() * _wxyz_to_rot(q1)
    step = R.from_rotvec(rot.as_rotvec() * t)
    return _rot_to_wxyz(_wxyz_to_rot(q0) * step)


def _interp_positions(start_pos: np.ndarray, end_pos: np.ndarray, num_steps: int) -> list[np.ndarray]:
    if num_steps <= 1:
        return [end_pos.copy()]
    ts = np.linspace(0.0, 1.0, num_steps)
    return [start_pos + float(t) * (end_pos - start_pos) for t in ts[1:]]


def _build_safe_approach_path(
    ee_pos: np.ndarray,
    approach_w: np.ndarray,
    *,
    clearance_z_m: float = 0.14,
) -> list[np.ndarray]:
    """Insert a via-point above the laptop base to avoid diving under the keyboard."""
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    approach_w = np.asarray(approach_w, dtype=np.float64)
    cruise_z = max(float(ee_pos[2]) - 0.02, clearance_z_m, float(approach_w[2]) + 0.05)
    via = np.array([float(approach_w[0]), float(approach_w[1]), cruise_z], dtype=np.float64)
    if float(np.linalg.norm(via - ee_pos)) < 0.02:
        return [approach_w]
    return [via, approach_w]


class PushInteractionExecutor:
    """Sample 15×8 contact candidates on link mesh, IK approach, hinge-relative push."""

    def __init__(
        self,
        *,
        robot,
        env_module: IsaacLabEnvironmentModule,
        handles: RobotHandles,
        timing: ControlLoopTiming,
        task_config: TaskInteractionConfig,
        sampling_config: SamplingConfig,
        push_config: PushConfig,
        success_specs: tuple[TaskRolloutSuccessSpec, ...],
        device: torch.device,
        diff_ik_controller,
        diff_ik_pos_controller,
        gripper_open_target: torch.Tensor,
        gripper_close_target: torch.Tensor,
        joint_upper_limit_deg: float = 104.0,
        rng: np.random.Generator | None = None,
        home_steps: int = 40,
        candidate_home_steps: int = 25,
    ):
        if task_config.interaction_mode != "push":
            raise ValueError(f"Expected interaction_mode=push, got {task_config.interaction_mode}")

        self.robot = robot
        self.env_module = env_module
        self.handles = handles
        self.timing = timing
        self.task_config = task_config
        self.sampling_config = sampling_config
        self.push_cfg = push_config
        self.success_specs = success_specs
        self.device = device
        self.diff_ik_controller = diff_ik_controller
        self.diff_ik_pos_controller = diff_ik_pos_controller
        self.gripper_open_target = gripper_open_target
        self.gripper_close_target = gripper_close_target
        self.joint_upper_limit_deg = joint_upper_limit_deg
        self.rng = rng or np.random.default_rng()
        self.home_steps = home_steps
        self.candidate_home_steps = candidate_home_steps
        self.max_ee_pos_step_m = float(getattr(push_config, "max_ee_pos_step_m", KEYBOARD_EE_POS_STEP_M))
        self.max_joint_step_rad = float(getattr(push_config, "max_joint_step_rad", KEYBOARD_JOINT_STEP_RAD))
        self._keyboard_waypoints: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._touch_contact_ref = None
        self._close_anchor: dict[str, object] | None = None

        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = 0.0
        self.last_record: StepRecord | None = None

        home_joints = SHARED_TELEOP_PIPER.joint_pos_dict()
        self.home_arm_rad = tuple(float(home_joints[f"joint{i}"]) for i in range(1, 7))
        self._all_candidates: list[ContactCandidate] = []
        self._tracking_pos = np.zeros(3, dtype=np.float64)
        self._tracking_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self._debug_trajectory_cache: dict[int, tuple[tuple[float, ...], ...]] = {}
        self._skip_recording = False

    def preload_debug_reference_trajectories(self) -> None:
        """Load reference joint trajectories once (before sim loop)."""
        if self._resolve_reference_hdf5() is None:
            return

        demo_ids = self.push_cfg.debug_reference_demos or (self.push_cfg.debug_reference_demo,)
        for demo_idx in demo_ids:
            trajectory = self._load_debug_reference_arm_trajectory(int(demo_idx))
            self._debug_trajectory_cache[int(demo_idx)] = tuple(trajectory)
            print(
                f"[INFO] Preloaded reference demo_{demo_idx}: "
                f"{len(trajectory)} frames."
            )

    def preload_articulation_contact_reference(self) -> None:
        """Load touch HDF5 once; store contact pose in link_1 frame + hinge lever length."""
        path = self._resolve_articulation_reference_hdf5()

        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_movable_link_pose()
        hinge_origin_w, hinge_axis_w = self.env_module.get_hinge_world_frame(
            link_prim,
            self.sampling_config.hinge.origin,
            self.sampling_config.hinge.axis,
        )

        demo_idx = int(self.push_cfg.keyboard_reference_demo)
        self._touch_contact_ref = load_touch_contact_from_hdf5(
            path,
            demo_idx,
            link_pos_world=link_pos_np,
            link_quat_wxyz=link_quat,
            hinge_origin_world=np.asarray(hinge_origin_w, dtype=np.float64),
            hinge_axis_world=np.asarray(hinge_axis_w, dtype=np.float64),
        )
        print(
            f"[INFO] Touch contact calibration ({path.name}): "
            f"{summarize_touch_reference(self._touch_contact_ref)}",
            flush=True,
        )

    def _resolve_articulation_reference_hdf5(self) -> Path:
        path = self._resolve_reference_hdf5()
        if path is None:
            raise ValueError(
                "articulation_calibrated requires push.keyboard_reference_hdf5 in task yaml."
            )
        if path.is_file():
            return path

        collect_dir = Path(__file__).resolve().parent
        fallback = collect_dir / "datasets/close_laptop_lid/close_laptop_lid_20260524_150830.hdf5"
        if fallback.is_file():
            print(
                f"[WARN] keyboard_reference_hdf5 not found: {path}\n"
                f"       Falling back to {fallback.name}. "
                "Re-record touch HDF5 or update yaml for calibrated contact.",
                flush=True,
            )
            return fallback

        raise FileNotFoundError(
            f"keyboard_reference_hdf5 not found: {path}\n"
            f"Expected touch demo at that path, or fallback {fallback}."
        )

    def _resolve_touch_contact_world(
        self,
    ) -> tuple[np.ndarray, tuple[float, float, float, float], TouchContactReference]:
        if self._touch_contact_ref is None:
            raise RuntimeError(
                "Touch contact not loaded; call preload_articulation_contact_reference() first."
            )
        ref = self._touch_contact_ref
        link_pos_np, link_quat = self._read_movable_link_pose()
        contact_w, quat_w = resolve_contact_pose_world(
            link_pos_np,
            link_quat,
            ref.contact_pos_link,
            ref.contact_quat_wxyz_link,
        )
        return contact_w, quat_w, ref

    def _resolve_yaml_handle_world(
        self,
    ) -> tuple[np.ndarray, tuple[float, float, float, float], float | None]:
        cfg = self.sampling_config
        if cfg.push_contact_offset_link is None:
            raise ValueError("yaml_handle requires push_contact_offset_link in task yaml.")

        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_link_pose()

        quat_link = cfg.contact_quat_link
        if quat_link is None:
            approach = cfg.approach_direction_world
            if cfg.use_scene_approach_direction:
                ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
                approach = scene_approach_direction(link_pos_np, link_quat, ee_pos)
            if approach is None:
                raise ValueError(
                    "yaml_handle requires contact_quat_link or approach_direction_world in task yaml."
                )
            quat_link = derive_contact_quat_link(
                link_quat,
                tuple(float(v) for v in approach),
                horizontal=cfg.horizontal_grasp,
            )

        contact_w, quat_w = resolve_yaml_handle_world(
            link_pos_np,
            link_quat,
            cfg.push_contact_offset_link,
            quat_link,
        )

        hinge_origin_w, hinge_axis_w = self.env_module.get_hinge_world_frame(
            link_prim,
            cfg.hinge.origin,
            cfg.hinge.axis,
        )
        lever_m, _ = hinge_lever_arm(contact_w, hinge_origin_w, hinge_axis_w)
        return contact_w, quat_w, float(lever_m)

    def _read_ee_pose(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        ee_quat = tuple(
            float(v) for v in self.robot.data.body_quat_w[0, self.handles.ee_body_id].tolist()
        )
        return np.asarray(ee_pos, dtype=np.float64), ee_quat

    def _read_body_pose_by_name(
        self, body_name: str
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        try:
            body_ids = self.robot.find_bodies(body_name)[0]
            if not body_ids:
                return None
            body_id = int(body_ids[0])
            pos = self.robot.data.body_pos_w[0, body_id].detach().cpu().numpy()
            quat = tuple(float(v) for v in self.robot.data.body_quat_w[0, body_id].tolist())
            return np.asarray(pos, dtype=np.float64), quat
        except Exception:
            return None

    @staticmethod
    def _gripper_push_axis_world(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
        """Unit vector: gripper +Z (into push surface) in world frame."""
        axis = _wxyz_to_rot(quat_wxyz).apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return axis / norm

    def _read_link_pose(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """link_1 world pose from USD stage (matches keyboard calibration at USD joint=15°).

        task_registry joint position uses USD asset units (15° ≈ lid open 90° real,
        104° = closed), not URDF radians. After sync_scene_joints_after_sim_reset,
        get_prim_world_pose reflects the authored/sim hinge state — do not re-rotate.
        """
        pos, quat = self.env_module.get_prim_world_pose_wxyz(self.task_config.link_prim)
        return np.asarray(pos, dtype=np.float64), quat

    def _read_movable_link_pose(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        return self._read_link_pose()

    def _reset_close_phase_tracking(self) -> None:
        """Re-sync incremental IK tracking before hinge-relative close."""
        self._close_anchor = None
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

    def _resolve_close_target_deg(self, joint_deg: float) -> float:
        """ArticuBot T_rel target: close_ratio, bumped to rollout success threshold if needed."""
        ratio_target = joint_deg + self.push_cfg.close_ratio * (
            self.joint_upper_limit_deg - joint_deg
        )
        success_floor = self.joint_upper_limit_deg
        for spec in self.success_specs:
            if spec.joint_prim == self.task_config.joint_prim:
                success_floor = float(spec.angle_gt_deg) + 1.0
                break
        target = max(ratio_target, success_floor)
        return min(target, self.joint_upper_limit_deg)

    def _resolve_num_close_waypoints(self, joint_init_deg: float, joint_target_deg: float) -> int:
        """ArticuBot Sec IV-A: one waypoint per close_step_deg_usd (paper uses ~1°)."""
        step_deg = float(getattr(self.push_cfg, "close_step_deg_usd", 0.0))
        if step_deg > 1e-6:
            span = float(joint_target_deg) - float(joint_init_deg)
            return max(2, int(math.ceil(abs(span) / step_deg)) + 1)
        return max(2, int(self.push_cfg.num_close_steps))

    def _build_hinge_close_poses(
        self,
        eef_pos_world: np.ndarray,
        eef_quat_wxyz: tuple[float, float, float, float],
    ) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        link_prim = self.task_config.link_prim
        joint_deg = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        if joint_deg is None:
            return []

        target_deg = self._resolve_close_target_deg(float(joint_deg))
        n_wp = self._resolve_num_close_waypoints(float(joint_deg), target_deg)
        theta_init_rad = math.radians(joint_deg)
        theta_targets = np.linspace(theta_init_rad, math.radians(target_deg), n_wp)

        hinge_origin_w, hinge_axis_w = self.env_module.get_hinge_world_frame(
            link_prim,
            self.sampling_config.hinge.origin,
            self.sampling_config.hinge.axis,
        )
        link_pos_np, link_quat = self._read_link_pose()

        ee_start = np.asarray(eef_pos_world, dtype=np.float64)
        poses = compute_articulation_ee_trajectory(
            eef_pos_world=eef_pos_world,
            eef_quat_wxyz=eef_quat_wxyz,
            link_pos_world=link_pos_np,
            link_quat_wxyz=link_quat,
            hinge_origin_world=hinge_origin_w,
            hinge_axis_world=hinge_axis_w,
            theta_init_rad=theta_init_rad,
            theta_targets_rad=tuple(float(t) for t in theta_targets),
        )
        if poses:
            ee_end = np.asarray(poses[-1][0], dtype=np.float64)
            arc_m = float(sum(
                np.linalg.norm(np.asarray(poses[i][0]) - np.asarray(poses[i - 1][0]))
                for i in range(1, len(poses))
            ))
            print(
                f"[INFO] T_rel close plan: joint {joint_deg:.2f}° -> {target_deg:.2f}° "
                f"({len(poses)} waypoints, EE arc {arc_m:.3f}m, "
                f"ΔEE {float(np.linalg.norm(ee_end - ee_start)):.3f}m)",
                flush=True,
            )
        return poses

    def _init_close_anchor(
        self,
        eef_pos_world: np.ndarray,
        eef_quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        """Capture ArticuBot T_rel at contact: constant eef_in_link while lid rotates."""
        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_link_pose()
        link_pos_inv, link_quat_inv = invert_pose(link_pos_np, link_quat)
        eef_in_link_pos, eef_in_link_quat = compose_pose(
            link_pos_inv,
            link_quat_inv,
            np.asarray(eef_pos_world, dtype=np.float64),
            eef_quat_wxyz,
        )
        hinge_origin_w, hinge_axis_w = self.env_module.get_hinge_world_frame(
            link_prim,
            self.sampling_config.hinge.origin,
            self.sampling_config.hinge.axis,
        )
        joint_init = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        if joint_init is None:
            joint_init = 15.0
        target_joint = self._resolve_close_target_deg(float(joint_init))
        self._close_anchor = {
            "link_pos_init": link_pos_np.copy(),
            "link_quat_init": link_quat,
            "contact_pos_w": np.asarray(eef_pos_world, dtype=np.float64),
            "contact_quat_w": eef_quat_wxyz,
            "eef_in_link_pos": np.asarray(eef_in_link_pos, dtype=np.float64),
            "eef_in_link_quat": eef_in_link_quat,
            "hinge_origin_w": np.asarray(hinge_origin_w, dtype=np.float64),
            "hinge_axis_w": np.asarray(hinge_axis_w, dtype=np.float64),
            "joint_init_deg": float(joint_init),
            "joint_target_deg": float(target_joint),
        }
        n_wp = self._resolve_num_close_waypoints(joint_init, target_joint)
        print(
            f"[INFO] Close anchor T_rel (USD joint_1: 15°≈real90°open, 104°≈real0°closed): "
            f"{joint_init:.2f}° -> {target_joint:.2f}° ({n_wp} waypoints) "
            f"eef_in_link={np.round(eef_in_link_pos, 4).tolist()}",
            flush=True,
        )
        diag0 = self._close_handle_diagnostics(joint_init, joint_deg=float(joint_init))
        if diag0:
            print(
                f"[INFO] Close anchor handle diag @joint={joint_init:.2f}°USD: "
                f"contact={np.round(diag0['handle_contact'], 4).tolist()} "
                f"handle_usd={np.round(diag0['handle_usd'], 4).tolist()} "
                f"handle_trl_joint={np.round(diag0['handle_trl_joint'], 4).tolist()} "
                f"trl_joint↔handle_usd={diag0['trl_joint_handle_usd_m']:.4f}m "
                f"handle_usd_physx_gap={diag0['handle_usd_physx_gap_m']:.4f}m",
                flush=True,
            )

    def _build_close_trajectory_from_anchor(
        self,
    ) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        """ArticuBot Eq.: T_eef(θ)=T_link(θ)T_link^{-1}(θ_init)T_eef_init; fixed eef_in_link from contact."""
        if self._close_anchor is None:
            return []
        anchor = self._close_anchor
        joint_init = float(anchor["joint_init_deg"])
        joint_target = float(anchor["joint_target_deg"])
        n_wp = self._resolve_num_close_waypoints(joint_init, joint_target)
        theta_init_rad = math.radians(joint_init)
        theta_targets = np.linspace(theta_init_rad, math.radians(joint_target), n_wp)
        contact_pos = np.asarray(anchor["contact_pos_w"], dtype=np.float64)
        poses = compute_articulation_ee_trajectory(
            eef_pos_world=contact_pos,
            eef_quat_wxyz=anchor["contact_quat_w"],
            link_pos_world=anchor["link_pos_init"],
            link_quat_wxyz=anchor["link_quat_init"],
            hinge_origin_world=anchor["hinge_origin_w"],
            hinge_axis_world=anchor["hinge_axis_w"],
            theta_init_rad=theta_init_rad,
            theta_targets_rad=tuple(float(t) for t in theta_targets),
        )
        if poses:
            ee_end = np.asarray(poses[-1][0], dtype=np.float64)
            arc_m = float(sum(
                np.linalg.norm(np.asarray(poses[i][0]) - np.asarray(poses[i - 1][0]))
                for i in range(1, len(poses))
            ))
            print(
                f"[INFO] T_rel close plan: joint {joint_init:.2f}° -> {joint_target:.2f}° "
                f"({len(poses)} waypoints @ {getattr(self.push_cfg, 'close_step_deg_usd', 1.0):.1f}°USD, "
                f"EE arc {arc_m:.3f}m, ΔEE {float(np.linalg.norm(ee_end - contact_pos)):.3f}m)",
                flush=True,
            )
        return poses

    def _sample_anchored_close_poses(
        self, num_samples: int | None = None
    ) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        if self._close_anchor is None:
            return []
        traj = self._build_close_trajectory_from_anchor()
        if num_samples is not None and len(traj) > num_samples:
            idx = np.linspace(0, len(traj) - 1, num_samples, dtype=int)
            return [traj[i] for i in idx]
        return traj

    def _live_handle_pose(
        self,
        joint_deg: float | None = None,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """T_rel handle at current (or given) joint — avoids stale USD link xform."""
        if self._close_anchor is None:
            raise RuntimeError("Call _init_close_anchor before close phase.")
        if joint_deg is None:
            joint_now = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
            if joint_now is None:
                joint_deg = float(self._close_anchor["joint_init_deg"])
            else:
                joint_deg = float(joint_now)
        return self._compute_close_pose_at_joint_deg(joint_deg)

    def _compute_close_pose_at_joint_deg(
        self, theta_deg: float
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Single T_rel EE pose for θ (USD), fixed eef_in_link from contact anchor."""
        if self._close_anchor is None:
            raise RuntimeError("Call _init_close_anchor before close phase.")
        anchor = self._close_anchor
        joint_init = float(anchor["joint_init_deg"])
        theta_init_rad = math.radians(joint_init)
        theta_rad = math.radians(float(theta_deg))
        poses = compute_articulation_ee_trajectory(
            eef_pos_world=np.asarray(anchor["contact_pos_w"], dtype=np.float64),
            eef_quat_wxyz=anchor["contact_quat_w"],
            link_pos_world=anchor["link_pos_init"],
            link_quat_wxyz=anchor["link_quat_init"],
            hinge_origin_world=anchor["hinge_origin_w"],
            hinge_axis_world=anchor["hinge_axis_w"],
            theta_init_rad=theta_init_rad,
            theta_targets_rad=(theta_rad,),
        )
        if not poses:
            raise RuntimeError(f"T_rel close pose failed at θ={theta_deg:.2f}°USD")
        return poses[-1]

    def _close_live_anchor_err_m(self) -> float:
        """Slip metric: EE vs T_rel at live joint (diagnostics / optional recover only)."""
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        live_pos, _ = self._live_handle_pose()
        return float(np.linalg.norm(ee_pos - np.asarray(live_pos)))

    def _ee_to_pose_err_m(
        self,
        target_pos: np.ndarray,
    ) -> float:
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        return float(np.linalg.norm(ee_pos - np.asarray(target_pos, dtype=np.float64)))

    def _joint_bind_deg_for_diag(self) -> float:
        joint_prim = self.task_config.joint_prim
        for spec in self.env_module.cfg.joint_initial_specs:
            if spec.prim_path == joint_prim:
                return float(spec.position)
        return 15.0

    def _compose_handle_from_link(
        self,
        link_pos: np.ndarray,
        link_quat: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if self._close_anchor is None:
            raise RuntimeError("Call _init_close_anchor before handle diagnostics.")
        return compose_pose(
            np.asarray(link_pos, dtype=np.float64),
            link_quat,
            self._close_anchor["eef_in_link_pos"],
            self._close_anchor["eef_in_link_quat"],
        )

    def _read_handle_trl_physx(
        self,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Handle contact point in world frame from PhysX link pose (not stale USD xform)."""
        cfg = self.sampling_config
        link_pos, link_quat = self.env_module.get_movable_link_world_pose_wxyz(
            self.task_config.link_prim,
            self.task_config.joint_prim,
            hinge_origin_link=cfg.hinge.origin,
            hinge_axis_link=cfg.hinge.axis,
            bind_joint_deg=self._joint_bind_deg_for_diag(),
        )
        return self._compose_handle_from_link(
            np.asarray(link_pos, dtype=np.float64), link_quat
        )

    def _close_handle_diagnostics(
        self,
        theta_cmd: float,
        joint_deg: float | None = None,
    ) -> dict[str, object]:
        """Read-only: compare stale USD handle vs T_rel references vs push lead target."""
        if self._close_anchor is None:
            return {}

        anchor = self._close_anchor
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        cfg = self.sampling_config
        if joint_deg is None:
            joint_log = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
            joint_deg = float(joint_log) if joint_log is not None else float(anchor["joint_init_deg"])

        link_usd, quat_usd = self._read_link_pose()
        handle_usd_pos, _ = self._compose_handle_from_link(link_usd, quat_usd)

        link_physx_pos, link_physx_quat = self.env_module.get_movable_link_world_pose_wxyz(
            self.task_config.link_prim,
            self.task_config.joint_prim,
            hinge_origin_link=cfg.hinge.origin,
            hinge_axis_link=cfg.hinge.axis,
            bind_joint_deg=self._joint_bind_deg_for_diag(),
        )
        link_physx = np.asarray(link_physx_pos, dtype=np.float64)
        handle_physx_pos, _ = self._compose_handle_from_link(link_physx, link_physx_quat)

        handle_trl_joint_pos, _ = self._compute_close_pose_at_joint_deg(float(joint_deg))
        handle_trl_pos, _ = self._compute_close_pose_at_joint_deg(theta_cmd)
        contact_pos = np.asarray(anchor["contact_pos_w"], dtype=np.float64)

        return {
            "ee_pos": ee_pos,
            "joint_deg": float(joint_deg),
            "handle_usd": handle_usd_pos,
            "handle_physx": handle_physx_pos,
            "handle_trl_joint": handle_trl_joint_pos,
            "handle_trl": handle_trl_pos,
            "handle_contact": contact_pos,
            "link_usd": link_usd,
            "link_physx": link_physx,
            "link_usd_physx_gap_m": float(np.linalg.norm(link_usd - link_physx)),
            "handle_usd_physx_gap_m": float(np.linalg.norm(handle_usd_pos - handle_physx_pos)),
            "ee_handle_usd_m": float(np.linalg.norm(ee_pos - handle_usd_pos)),
            "ee_handle_trl_joint_m": float(np.linalg.norm(ee_pos - handle_trl_joint_pos)),
            "ee_handle_physx_m": float(np.linalg.norm(ee_pos - handle_physx_pos)),
            "trl_joint_handle_usd_m": float(np.linalg.norm(handle_trl_joint_pos - handle_usd_pos)),
            "trl_handle_usd_m": float(np.linalg.norm(handle_trl_pos - handle_usd_pos)),
            "trl_handle_trl_joint_m": float(np.linalg.norm(handle_trl_pos - handle_trl_joint_pos)),
        }

    def _servo_toward_close_pose(
        self,
        collector: OfficialEpisodeCollector,
        target_pos: np.ndarray,
        target_quat: tuple[float, float, float, float],
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        *,
        substeps: int | None = None,
        max_pos_step_m: float | None = None,
        clamp_joints: bool | None = None,
    ) -> bool:
        """Incremental pose IK toward one T_rel waypoint (ArticuBot Sec IV-A)."""
        inner = max(1, int(substeps or getattr(self.push_cfg, "close_ik_substeps", 4)))
        pos_step = self.max_ee_pos_step_m if max_pos_step_m is None else max_pos_step_m
        if clamp_joints is None:
            clamp_joints = bool(getattr(self.push_cfg, "close_clamp_joints", False))
        for _ in range(inner):
            self._advance_tracking_pose(target_pos, target_quat, max_pos_step_m=pos_step)
            arm_targets = self._ik_targets_for_pose(
                self._tracking_pos, self._tracking_quat_wxyz
            )
            if self._control_step_ik(
                collector,
                arm_targets,
                gripper_open=False,
                on_control_step=on_control_step,
                clamp_joints=clamp_joints,
            ):
                return True
        return False

    def _execute_articubot_close_trajectory(
        self,
        collector: OfficialEpisodeCollector,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        trajectory: list[tuple[np.ndarray, tuple[float, float, float, float]]] | None = None,
    ) -> bool:
        """ArticuBot Sec IV-A: march precomputed T_rel arc once; advance wp when ee↔wp within tol."""
        if self._close_anchor is None:
            return False

        if trajectory is None:
            trajectory = self._build_close_trajectory_from_anchor()
        if len(trajectory) < 2:
            return False

        anchor = self._close_anchor
        joint_init = float(anchor["joint_init_deg"])
        joint_target = float(anchor["joint_target_deg"])
        close_poses = list(trajectory[1:])
        pos_tol_m = float(
            getattr(self.push_cfg, "close_pose_reach_tol_m", CLOSE_POSE_REACH_TOL_M)
        )
        max_steps_per_wp = max(
            1, int(getattr(self.push_cfg, "close_max_steps_per_waypoint", 500))
        )
        close_step_m = float(getattr(self.push_cfg, "close_push_ee_step_m", 0.003))
        close_clamp = bool(getattr(self.push_cfg, "close_clamp_joints", False))
        log_stride = max(1, len(close_poses) // 8)

        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        wp0 = np.asarray(trajectory[0][0], dtype=np.float64)
        wp_last = np.asarray(trajectory[-1][0], dtype=np.float64)
        print(
            f"[INFO] Close phase (open-loop T_rel): {len(close_poses)} waypoints, "
            f"{joint_init:.1f}°->{joint_target:.1f}°USD, "
            f"ee_step={close_step_m * 1000:.1f}mm/substep, pos_tol={pos_tol_m * 1000:.1f}mm, "
            f"max_substeps/wp={max_steps_per_wp}, close_clamp_joints={close_clamp}, "
            f"planned EE arc {float(np.linalg.norm(wp_last - wp0)):.3f}m",
            flush=True,
        )

        for wp_i, (target_pos, target_quat) in enumerate(close_poses, start=1):
            target_pos_np = np.asarray(target_pos, dtype=np.float64)
            substeps_used = 0
            wp_err = self._ee_to_pose_err_m(target_pos_np)

            while wp_err > pos_tol_m and substeps_used < max_steps_per_wp:
                if self._servo_toward_close_pose(
                    collector,
                    target_pos_np,
                    target_quat,
                    on_control_step,
                    substeps=1,
                    max_pos_step_m=close_step_m,
                    clamp_joints=close_clamp,
                ):
                    return True

                substeps_used += 1
                wp_err = self._ee_to_pose_err_m(target_pos_np)

                if on_control_step is not None:
                    success_now, joint_degs = evaluate_rollout_success(
                        self.env_module, self.success_specs
                    )
                    if on_control_step(success_now, joint_degs):
                        return True
                else:
                    success_now, _ = evaluate_rollout_success(
                        self.env_module, self.success_specs
                    )
                    if success_now:
                        return True

            if wp_i == 1 or wp_i % log_stride == 0 or wp_i == len(close_poses):
                status = "reached" if wp_err <= pos_tol_m else "timeout"
                print(
                    f"[INFO] Close wp {wp_i}/{len(close_poses)} ({status}): "
                    f"ee↔wp={wp_err:.4f}m, substeps={substeps_used}",
                    flush=True,
                )
            if wp_err > pos_tol_m:
                print(
                    f"[WARN] Close wp {wp_i}/{len(close_poses)}: ee↔wp={wp_err:.4f}m "
                    f"> tol {pos_tol_m:.4f}m after {substeps_used} substeps (cap); advancing",
                    flush=True,
                )

        return False

    def _execute_close_phase(
        self,
        collector: OfficialEpisodeCollector,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
    ) -> bool:
        trajectory = self._build_close_trajectory_from_anchor()
        if len(trajectory) < 2:
            return False

        stopped = self._execute_articubot_close_trajectory(
            collector, on_control_step, trajectory=trajectory
        )

        joint_prim = self.task_config.joint_prim
        joint_init = float(self._close_anchor["joint_init_deg"]) if self._close_anchor else 15.0
        joint_fin = self.env_module.read_scene_joint_angle_deg(joint_prim)
        fin = float(joint_fin or joint_init)
        print(
            f"[INFO] Close finished: {joint_prim} {joint_init:.2f}°USD(real≈"
            f"{usd_joint_to_real_lid_deg(joint_init):.0f}°) -> {fin:.2f}°USD(real≈"
            f"{usd_joint_to_real_lid_deg(fin):.0f}°). "
            f"Motion ended; process stays alive until Ctrl+C.",
            flush=True,
        )
        return stopped

    def _execute_hinge_close_anchored(
        self,
        collector: OfficialEpisodeCollector,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
    ) -> bool:
        """Close after contact: ArticuBot T_rel from contact anchor (fixed link_init + eef_in_link)."""
        if self._close_anchor is None:
            return False
        return self._execute_close_phase(collector, on_control_step)

    def _approach_steps_per_waypoint(self, num_waypoints: int) -> int:
        budget = int(getattr(self.push_cfg, "approach_steps", 30))
        return max(8, budget // max(1, num_waypoints))

    def _plan_handle_reach_poses(
        self,
        ee_pos: np.ndarray,
        approach_w: np.ndarray,
        contact_w: np.ndarray,
        quat_w: tuple[float, float, float, float],
        start_quat_wxyz: tuple[float, float, float, float],
    ) -> tuple[list[np.ndarray], list[tuple[np.ndarray, tuple[float, float, float, float]]]]:
        """Pose waypoints: cruise with slerp to contact orientation, then contact segment."""
        clearance_z = float(getattr(self.push_cfg, "approach_clearance_z_m", 0.14))
        approach_path = _build_safe_approach_path(ee_pos, approach_w, clearance_z_m=clearance_z)
        n_contact = max(2, int(self.push_cfg.contact_hold_steps))
        contact_path = _interp_positions(approach_w, contact_w, n_contact)

        poses: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
        n_approach = len(approach_path)
        for i, wp in enumerate(approach_path):
            t = float(i + 1) / float(n_approach)
            quat = _slerp_wxyz(start_quat_wxyz, quat_w, t)
            poses.append((wp, quat))
        for p in contact_path[1:]:
            poses.append((p, quat_w))
        return approach_path, poses

    def _execute_handle_reach(
        self,
        collector: OfficialEpisodeCollector,
        approach_w: np.ndarray,
        contact_w: np.ndarray,
        quat_w: tuple[float, float, float, float],
        *,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        approach_step_budget: int | None = None,
        contact_steps_per_waypoint: int | None = None,
    ) -> bool:
        """Pose IK approach + contact with joint clamping. Returns True if on_control_step signals stop."""
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        start_quat = tuple(
            float(v) for v in self.robot.data.body_quat_w[0, self.handles.ee_body_id].tolist()
        )
        approach_path, reach_poses = self._plan_handle_reach_poses(
            ee_pos, approach_w, contact_w, quat_w, start_quat
        )
        n_approach = len(approach_path)
        approach_poses = reach_poses[:n_approach]
        contact_poses = reach_poses[n_approach:]

        if approach_step_budget is None:
            approach_steps_per = self._approach_steps_per_waypoint(n_approach)
        else:
            approach_steps_per = max(8, int(approach_step_budget) // max(1, n_approach))
        contact_steps_per = max(8, int(contact_steps_per_waypoint or self.push_cfg.contact_hold_steps))

        print(
            f"[INFO] Handle reach: {n_approach} approach + {len(contact_poses)} contact waypoints "
            f"({approach_steps_per} steps/wp approach, {contact_steps_per} steps/wp contact, pose IK)",
            flush=True,
        )
        print(
            f"[INFO] Safe approach path ({len(approach_path)} via-points): "
            + " -> ".join(str(np.round(wp, 3).tolist()) for wp in approach_path),
            flush=True,
        )

        if self._follow_pose_path(
            collector,
            approach_poses,
            approach_steps_per,
            on_control_step,
            clamp_joints=True,
        ):
            return True
        if self._follow_pose_path(
            collector,
            contact_poses,
            contact_steps_per,
            on_control_step,
            clamp_joints=True,
        ):
            return True
        return False

    def _summarize_handle_contact(
        self,
        contact_w: np.ndarray,
        quat_w: tuple[float, float, float, float],
        approach_w: np.ndarray,
        *,
        hinge_lever_m: float | None = None,
    ) -> dict[str, object]:
        """Diagnostic snapshot after handle reach (planned vs actual EE / link frame)."""
        actual_pos, actual_quat = self._read_ee_pose()
        contact_w = np.asarray(contact_w, dtype=np.float64)
        approach_w = np.asarray(approach_w, dtype=np.float64)

        pos_err_m = float(np.linalg.norm(actual_pos - contact_w))
        rot_err_rad = (_wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_quat).inv()).magnitude()
        rot_err_deg = math.degrees(rot_err_rad)
        handle_reached = pos_err_m <= POSE_REACH_TOL_M and rot_err_rad <= POSE_REACH_ROT_RAD

        link_pos_np, link_quat = self._read_movable_link_pose()
        link_pos_inv, link_quat_inv = invert_pose(link_pos_np, link_quat)
        ee_pos_link, ee_quat_link = compose_pose(
            link_pos_inv, link_quat_inv, actual_pos, actual_quat
        )

        joint_prim = self.task_config.joint_prim
        joint_deg = self.env_module.read_scene_joint_angle_deg(joint_prim)
        arm_joint_rad = [
            float(v)
            for v in self.robot.data.joint_pos[0, self.handles.arm_joint_ids].detach().cpu().tolist()
        ]

        cfg = self.sampling_config
        offset_link = cfg.push_contact_offset_link if cfg else None
        offset_err_m = None
        if offset_link is not None:
            offset_err_m = float(np.linalg.norm(np.asarray(ee_pos_link) - np.asarray(offset_link)))

        ee_to_contact = contact_w - actual_pos
        push_axis_planned = self._gripper_push_axis_world(quat_w)
        push_axis_actual = self._gripper_push_axis_world(actual_quat)
        signed_push_gap_m = float(np.dot(ee_to_contact, push_axis_actual))
        lateral_err_m = float(
            np.linalg.norm(ee_to_contact - signed_push_gap_m * push_axis_actual)
        )

        gripper_base_pose = self._read_body_pose_by_name("gripper_base")
        gripper_base_pos_w = None
        gripper_base_dist_m = None
        link6_to_gripper_base_m = None
        est_finger_pad_pos_w = None
        est_finger_pad_dist_m = None
        if gripper_base_pose is not None:
            gripper_base_pos_w, gripper_base_quat = gripper_base_pose
            gripper_base_dist_m = float(np.linalg.norm(gripper_base_pos_w - contact_w))
            link6_to_gripper_base_m = float(np.linalg.norm(gripper_base_pos_w - actual_pos))
            pad_offset = _wxyz_to_rot(gripper_base_quat).apply(
                np.array([0.0, 0.0, GRIPPER_FINGER_ORIGIN_OFFSET_M], dtype=np.float64)
            )
            est_finger_pad_pos_w = gripper_base_pos_w + pad_offset
            est_finger_pad_dist_m = float(np.linalg.norm(est_finger_pad_pos_w - contact_w))

        ref_contact_w = None
        ref_contact_err_m = None
        if cfg is not None and cfg.reference_contact_world is not None:
            ref_contact_w = np.asarray(cfg.reference_contact_world, dtype=np.float64)
            ref_contact_err_m = float(np.linalg.norm(contact_w - ref_contact_w))

        handle_mesh_local = None
        if cfg is not None and offset_link is not None:
            mesh_origin = np.asarray(cfg.mesh_origin, dtype=np.float64)
            handle_mesh_local = np.asarray(offset_link, dtype=np.float64) - mesh_origin

        handle_reached_strict = (
            pos_err_m <= 0.005 and rot_err_rad <= POSE_REACH_ROT_RAD
        )
        visual_touch_likely = (
            handle_reached_strict
            and signed_push_gap_m <= 0.005
            and lateral_err_m <= 0.008
        )

        return {
            "handle_reached": handle_reached,
            "handle_reached_strict": handle_reached_strict,
            "visual_touch_likely": visual_touch_likely,
            "ee_body_name": "link6",
            "handle_reached_note": (
                "REACHED = link6 within pos/rot tol; does not guarantee visible finger contact."
            ),
            "planned_contact_w": contact_w.copy(),
            "planned_quat_w": quat_w,
            "planned_approach_w": approach_w.copy(),
            "actual_ee_pos_w": actual_pos.copy(),
            "actual_ee_quat_w": actual_quat,
            "actual_ee_pos_link": np.asarray(ee_pos_link, dtype=np.float64),
            "actual_ee_quat_link": ee_quat_link,
            "yaml_push_contact_offset_link": offset_link,
            "offset_err_m": offset_err_m,
            "pos_err_m": pos_err_m,
            "rot_err_deg": rot_err_deg,
            "pos_tol_m": POSE_REACH_TOL_M,
            "rot_tol_deg": math.degrees(POSE_REACH_ROT_RAD),
            "ee_to_contact_w": ee_to_contact.copy(),
            "push_axis_planned_w": push_axis_planned.copy(),
            "push_axis_actual_w": push_axis_actual.copy(),
            "signed_push_gap_m": signed_push_gap_m,
            "lateral_err_m": lateral_err_m,
            "gripper_base_pos_w": gripper_base_pos_w.copy() if gripper_base_pos_w is not None else None,
            "gripper_base_dist_m": gripper_base_dist_m,
            "link6_to_gripper_base_m": link6_to_gripper_base_m,
            "est_finger_pad_pos_w": (
                est_finger_pad_pos_w.copy() if est_finger_pad_pos_w is not None else None
            ),
            "est_finger_pad_dist_m": est_finger_pad_dist_m,
            "reference_contact_w": ref_contact_w.copy() if ref_contact_w is not None else None,
            "reference_contact_err_m": ref_contact_err_m,
            "handle_mesh_local": (
                handle_mesh_local.copy() if handle_mesh_local is not None else None
            ),
            "mesh_origin_link": (
                np.asarray(cfg.mesh_origin, dtype=np.float64).copy() if cfg is not None else None
            ),
            "hinge_lever_m": hinge_lever_m,
            "link_pos_w": link_pos_np.copy(),
            "link_quat_w": link_quat,
            "joint_prim": joint_prim,
            "joint_deg": joint_deg,
            "arm_joint_rad": arm_joint_rad,
            "control_steps": self.control_step_count,
        }

    def print_handle_contact_report(self, report: dict[str, object], *, label: str = "yaml_handle") -> None:
        reached = bool(report["handle_reached"])
        status = "REACHED" if reached else "NOT REACHED"
        print(f"\n{'=' * 72}", flush=True)
        print(f"[HANDLE CONTACT] {label} — {status}", flush=True)
        print(f"{'=' * 72}", flush=True)
        print(
            f"  EE body (IK target): {report.get('ee_body_name', 'link6')} "
            f"(keyboard/HDF5 robot_eef_pos uses same frame)",
            flush=True,
        )
        if report.get("handle_reached_note"):
            print(f"  note              : {report['handle_reached_note']}", flush=True)
        print(
            f"  planned contact_w : {np.round(report['planned_contact_w'], 4).tolist()}",
            flush=True,
        )
        print(
            f"  planned approach_w: {np.round(report['planned_approach_w'], 4).tolist()}",
            flush=True,
        )
        print(
            f"  actual EE pos_w   : {np.round(report['actual_ee_pos_w'], 4).tolist()}",
            flush=True,
        )
        print(f"  actual EE quat_w  : {np.round(report['actual_ee_quat_w'], 4).tolist()}", flush=True)
        print(
            f"  pos err           : {report['pos_err_m']:.4f} m "
            f"(tol {report['pos_tol_m']:.3f} m) "
            f"{'OK' if float(report['pos_err_m']) <= float(report['pos_tol_m']) else 'FAIL'}",
            flush=True,
        )
        print(
            f"  rot err           : {report['rot_err_deg']:.1f} deg "
            f"(tol {report['rot_tol_deg']:.1f} deg) "
            f"{'OK' if float(report['rot_err_deg']) <= float(report['rot_tol_deg']) else 'FAIL'}",
            flush=True,
        )
        if report.get("signed_push_gap_m") is not None:
            gap = float(report["signed_push_gap_m"])
            print(
                f"  push-axis gap     : {gap:.4f} m (+ = EE short of contact along gripper +Z)",
                flush=True,
            )
        if report.get("lateral_err_m") is not None:
            print(f"  lateral err       : {float(report['lateral_err_m']):.4f} m", flush=True)
        if report.get("yaml_push_contact_offset_link") is not None:
            print(
                f"  yaml offset_link  : {np.round(report['yaml_push_contact_offset_link'], 4).tolist()}",
                flush=True,
            )
            print(
                f"  actual EE pos_link: {np.round(report['actual_ee_pos_link'], 4).tolist()}",
                flush=True,
            )
            if report.get("offset_err_m") is not None:
                print(f"  offset err (link) : {float(report['offset_err_m']):.4f} m", flush=True)
        if report.get("mesh_origin_link") is not None and report.get("handle_mesh_local") is not None:
            print(
                f"  mesh origin (link): {np.round(report['mesh_origin_link'], 4).tolist()} "
                f"(URDF link_1 visual/collision offset)",
                flush=True,
            )
            print(
                f"  handle mesh-local : {np.round(report['handle_mesh_local'], 4).tolist()} "
                f"(= offset_link − mesh_origin)",
                flush=True,
            )
        if report.get("gripper_base_pos_w") is not None:
            print(
                f"  gripper_base pos_w: {np.round(report['gripper_base_pos_w'], 4).tolist()}",
                flush=True,
            )
            if report.get("gripper_base_dist_m") is not None:
                print(
                    f"  gripper_base dist : {float(report['gripper_base_dist_m']):.4f} m to planned contact",
                    flush=True,
                )
            if report.get("link6_to_gripper_base_m") is not None:
                print(
                    f"  link6↔gripper_base: {float(report['link6_to_gripper_base_m']):.4f} m",
                    flush=True,
                )
        if report.get("est_finger_pad_pos_w") is not None:
            print(
                f"  est finger pad_w  : {np.round(report['est_finger_pad_pos_w'], 4).tolist()} "
                f"(URDF +{GRIPPER_FINGER_ORIGIN_OFFSET_M:.4f} m along gripper +Z)",
                flush=True,
            )
            if report.get("est_finger_pad_dist_m") is not None:
                print(
                    f"  finger pad dist   : {float(report['est_finger_pad_dist_m']):.4f} m to planned contact",
                    flush=True,
                )
        if report.get("reference_contact_w") is not None:
            print(
                f"  ref contact_w     : {np.round(report['reference_contact_w'], 4).tolist()}",
                flush=True,
            )
            if report.get("reference_contact_err_m") is not None:
                print(
                    f"  ref contact err   : {float(report['reference_contact_err_m']):.4f} m "
                    f"(planned vs yaml tie-break)",
                    flush=True,
                )
        if report.get("hinge_lever_m") is not None:
            print(f"  hinge lever       : {float(report['hinge_lever_m']):.4f} m", flush=True)
        print(f"  link_1 pos_w      : {np.round(report['link_pos_w'], 4).tolist()}", flush=True)
        print(f"  {report['joint_prim']} : {report['joint_deg']} deg", flush=True)
        print(f"  arm joints (rad)  : {np.round(report['arm_joint_rad'], 3).tolist()}", flush=True)
        print(f"  control steps     : {report['control_steps']}", flush=True)
        if reached and not bool(report.get("visual_touch_likely", False)):
            print(
                "  [WARN] Numerical REACHED but visual touch unlikely: "
                "check push-axis gap / lateral err / yaml handle offset.",
                flush=True,
            )
        elif bool(report.get("visual_touch_likely", False)):
            print("  [OK] Strict reach + small push-axis/lateral err — visual touch likely.", flush=True)
        print(f"{'=' * 72}\n", flush=True)

    def print_handle_contact_exit_summary(self, report: dict[str, object]) -> None:
        """Compact shutdown line for contact-only probe (survives sim teardown)."""
        reached = bool(report.get("handle_reached", False))
        status = "REACHED" if reached else "NOT_REACHED"
        visual = "likely" if bool(report.get("visual_touch_likely", False)) else "unlikely"
        parts = [
            f"[HANDLE CONTACT EXIT] {status}",
            f"pos_err={float(report.get('pos_err_m', float('nan'))):.4f}m",
            f"offset_err={float(report.get('offset_err_m', float('nan'))):.4f}m",
            f"push_gap={float(report.get('signed_push_gap_m', float('nan'))):+.4f}m",
            f"lateral={float(report.get('lateral_err_m', float('nan'))):.4f}m",
            f"visual={visual}",
            f"steps={report.get('control_steps', '?')}",
        ]
        if report.get("joint_deg") is not None:
            parts.append(f"joint_1={report['joint_deg']}deg")
        print("\n" + " | ".join(parts), flush=True)
        if reached and visual == "unlikely":
            print(
                "[HANDLE CONTACT EXIT] IK tracks link6; REACHED ≠ finger on lid. "
                "Retune push_contact_offset_link or tighten approach if gap/lateral too large.",
                flush=True,
            )

    def print_push_probe_exit_summary(self, result: dict[str, object]) -> None:
        """Compact shutdown line for yaml_handle_push probe."""
        success = bool(result.get("success", False))
        joint_prim = self.task_config.joint_prim
        joint_degs = result.get("joint_degs") or {}
        joint_deg = joint_degs.get(joint_prim) if isinstance(joint_degs, dict) else None
        planned = np.asarray(result.get("planned_contact_w"), dtype=np.float64)
        actual = np.asarray(result.get("actual_contact_w"), dtype=np.float64)
        drift = float(np.linalg.norm(actual - planned)) if planned.size and actual.size else float("nan")
        n_close = len(result.get("close_poses") or [])
        parts = [
            f"[PUSH+CLOSE EXIT] success={success}",
            f"contact_drift={drift:.4f}m",
            f"close_wps={n_close}",
            f"steps={self.control_step_count}",
        ]
        if joint_deg is not None:
            parts.append(f"{joint_prim}={float(joint_deg):.2f}deg")
        print("\n" + " | ".join(parts), flush=True)
        if not success and joint_deg is not None:
            print(
                "[PUSH+CLOSE EXIT] Success needs joint_1 > 98°USD (task_registry). "
                "Check Close push logs if lid did not move.",
                flush=True,
            )

    def run_yaml_handle_contact_only_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        max_servo_steps: int = 400,
    ) -> dict[str, object]:
        """Move to yaml handle contact only; print report and stop (no close phase)."""
        prev_skip = self._skip_recording
        self._skip_recording = True
        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = time.perf_counter()
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        try:
            contact_w, quat_w, lever_m = self._resolve_yaml_handle_world()
            approach_w = approach_from_contact(
                contact_w,
                quat_w,
                float(self.push_cfg.approach_backoff_m),
            )
            print(
                f"[INFO] yaml_handle contact-only: hinge_lever={lever_m:.4f}m "
                f"contact_w={np.round(contact_w, 4).tolist()} "
                f"approach_w={np.round(approach_w, 4).tolist()}",
                flush=True,
            )

            contact_steps_per = max(8, int(self.push_cfg.contact_hold_steps))
            approach_budget = max(
                int(self.push_cfg.approach_steps),
                int(max_servo_steps) - contact_steps_per * max(2, int(self.push_cfg.contact_hold_steps)),
            )
            self._execute_handle_reach(
                collector,
                approach_w,
                contact_w,
                quat_w,
                on_control_step=None,
                approach_step_budget=approach_budget,
                contact_steps_per_waypoint=contact_steps_per,
            )

            report = self._summarize_handle_contact(
                contact_w, quat_w, approach_w, hinge_lever_m=lever_m
            )
            self.print_handle_contact_report(report, label="yaml_handle contact-only")

            link_prim = self.task_config.link_prim
            hinge_origin_w, hinge_axis_w = self.env_module.get_hinge_world_frame(
                link_prim,
                self.sampling_config.hinge.origin,
                self.sampling_config.hinge.axis,
            )
            report["approach_w"] = approach_w
            report["hinge_origin_w"] = hinge_origin_w
            report["hinge_axis_w"] = hinge_axis_w
            report["close_poses"] = []
            return report
        finally:
            self._skip_recording = prev_skip

    def _run_handle_contact_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
        contact_w: np.ndarray,
        quat_w: tuple[float, float, float, float],
        *,
        label: str,
        hinge_lever_m: float | None = None,
    ) -> tuple[bool, dict[str, float | None]]:
        """Approach yaml/HDF5 handle, contact, then hinge-relative close from actual EE anchor."""
        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = episode_start_wall_time
        self.last_record = None
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        approach_w = approach_from_contact(
            contact_w,
            quat_w,
            float(self.push_cfg.approach_backoff_m),
        )

        lever_msg = f" hinge_lever={hinge_lever_m:.4f}m" if hinge_lever_m is not None else ""
        print(
            f"[INFO] {label} push:{lever_msg} "
            f"contact_w={np.round(contact_w, 4).tolist()} "
            f"approach_w={np.round(approach_w, 4).tolist()}"
        )

        final_joint_degs: dict[str, float | None] = {}

        def on_step(success_now: bool, joint_degs: dict[str, float | None]) -> bool:
            nonlocal final_joint_degs
            final_joint_degs = joint_degs
            return success_now

        if self._execute_handle_reach(
            collector,
            approach_w,
            contact_w,
            quat_w,
            on_control_step=on_step,
        ):
            return True, final_joint_degs

        actual_contact_pos, actual_contact_quat = self._read_ee_pose()
        contact_drift = float(np.linalg.norm(actual_contact_pos - contact_w))
        rot_drift = (
            _wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_contact_quat).inv()
        ).magnitude()
        print(
            f"[INFO] Contact anchor: planned={np.round(contact_w, 4).tolist()} "
            f"actual={np.round(actual_contact_pos, 4).tolist()} "
            f"drift={contact_drift:.4f}m rot={math.degrees(rot_drift):.1f}deg"
        )

        self._reset_close_phase_tracking()
        self._init_close_anchor(actual_contact_pos, actual_contact_quat)
        close_poses = self._sample_anchored_close_poses()
        if not close_poses:
            return False, final_joint_degs

        if self._execute_hinge_close_anchored(collector, on_step):
            return True, final_joint_degs

        success_now, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
        final_joint_degs = joint_degs
        return success_now, final_joint_degs

    def run_yaml_handle_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
    ) -> tuple[bool, dict[str, float | None]]:
        """Approach link-local handle from yaml, then hinge-relative close (ArticuBot T_rel held)."""
        contact_w, quat_w, lever_m = self._resolve_yaml_handle_world()
        return self._run_handle_contact_push(
            collector,
            episode_start_wall_time,
            contact_w,
            quat_w,
            label="yaml_handle",
            hinge_lever_m=lever_m,
        )

    def run_articulation_calibrated_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
    ) -> tuple[bool, dict[str, float | None]]:
        """Legacy: approach touch contact from HDF5, then hinge-relative close."""
        contact_w, quat_w, ref = self._resolve_touch_contact_world()
        return self._run_handle_contact_push(
            collector,
            episode_start_wall_time,
            contact_w,
            quat_w,
            label="Articulation (HDF5)",
            hinge_lever_m=ref.hinge_lever_m,
        )

    def _run_handle_contact_probe(
        self,
        collector: OfficialEpisodeCollector,
        contact_w: np.ndarray,
        quat_w: tuple[float, float, float, float],
        *,
        label: str,
        hinge_lever_m: float | None,
        max_servo_steps: int,
        hold_control_steps: int,
    ) -> dict[str, object]:
        approach_w = approach_from_contact(
            contact_w,
            quat_w,
            float(self.push_cfg.approach_backoff_m),
        )

        link_prim = self.task_config.link_prim
        hinge_origin_w, hinge_axis_w = self.env_module.get_hinge_world_frame(
            link_prim,
            self.sampling_config.hinge.origin,
            self.sampling_config.hinge.axis,
        )

        lever_msg = f" hinge_lever={hinge_lever_m:.4f}m" if hinge_lever_m is not None else ""
        print(
            f"[INFO] {label} probe:{lever_msg} "
            f"contact_w={np.round(contact_w, 4).tolist()} "
            f"approach_w={np.round(approach_w, 4).tolist()}",
            flush=True,
        )

        contact_steps_per = max(8, int(self.push_cfg.contact_hold_steps))
        approach_budget = max(
            int(self.push_cfg.approach_steps),
            int(max_servo_steps) - contact_steps_per * max(2, int(self.push_cfg.contact_hold_steps)),
        )
        self._execute_handle_reach(
            collector,
            approach_w,
            contact_w,
            quat_w,
            on_control_step=None,
            approach_step_budget=approach_budget,
            contact_steps_per_waypoint=contact_steps_per,
        )

        actual_contact_pos, actual_contact_quat = self._read_ee_pose()
        contact_drift = float(np.linalg.norm(actual_contact_pos - contact_w))
        rot_drift = (
            _wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_contact_quat).inv()
        ).magnitude()
        print(
            f"[INFO] Contact anchor: planned={np.round(contact_w, 4).tolist()} "
            f"actual={np.round(actual_contact_pos, 4).tolist()} "
            f"drift={contact_drift:.4f}m rot={math.degrees(rot_drift):.1f}deg",
            flush=True,
        )

        self._reset_close_phase_tracking()
        self._init_close_anchor(actual_contact_pos, actual_contact_quat)
        close_poses = self._sample_anchored_close_poses()
        if close_poses:
            self._execute_hinge_close_anchored(collector, on_control_step=None)

        success_now, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
        joint_deg = joint_degs.get(self.task_config.joint_prim)
        if joint_deg is not None:
            print(
                f"[INFO] {label} probe finished: success={success_now} "
                f"{self.task_config.joint_prim}={joint_deg:.2f}deg",
                flush=True,
            )
        else:
            print(f"[INFO] {label} probe finished: success={success_now} joints={joint_degs}", flush=True)

        return {
            "planned_contact_w": contact_w,
            "actual_contact_w": actual_contact_pos,
            "approach_w": approach_w,
            "close_poses": close_poses,
            "hinge_origin_w": hinge_origin_w,
            "hinge_axis_w": hinge_axis_w,
            "hinge_lever_m": hinge_lever_m,
            "joint_degs": joint_degs,
            "success": success_now,
            "contact_drift_m": contact_drift,
            "contact_rot_drift_deg": math.degrees(rot_drift),
            "control_steps": self.control_step_count,
        }

    def run_yaml_handle_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        max_servo_steps: int = 400,
        hold_control_steps: int = 120,
    ) -> dict[str, object]:
        """Livestream debug: yaml handle approach + contact + hinge-relative close."""
        prev_skip = self._skip_recording
        self._skip_recording = True
        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = time.perf_counter()
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        try:
            contact_w, quat_w, lever_m = self._resolve_yaml_handle_world()
            return self._run_handle_contact_probe(
                collector,
                contact_w,
                quat_w,
                label="yaml_handle",
                hinge_lever_m=lever_m,
                max_servo_steps=max_servo_steps,
                hold_control_steps=hold_control_steps,
            )
        finally:
            self._skip_recording = prev_skip

    def run_articulation_calibrated_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        max_servo_steps: int = 400,
        hold_control_steps: int = 120,
    ) -> dict[str, object]:
        """Legacy livestream debug: touch HDF5 approach + contact + hinge-relative close."""
        if self._touch_contact_ref is None:
            raise RuntimeError(
                "Call preload_articulation_contact_reference() before articulation probe."
            )

        prev_skip = self._skip_recording
        self._skip_recording = True
        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = time.perf_counter()
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        try:
            contact_w, quat_w, ref = self._resolve_touch_contact_world()
            return self._run_handle_contact_probe(
                collector,
                contact_w,
                quat_w,
                label="Articulation (HDF5)",
                hinge_lever_m=ref.hinge_lever_m,
                max_servo_steps=max_servo_steps,
                hold_control_steps=hold_control_steps,
            )
        finally:
            self._skip_recording = prev_skip

    def run_link_local_axis_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        local_offset: tuple[float, float, float] = (0.0, 0.0, 0.2),
        max_servo_steps: int = 400,
        hold_control_steps: int = 90,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Move EE to link-local offset (default +Z 20 cm) to verify prim frame."""
        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_movable_link_pose()
        target_w = compute_link_local_probe_point(link_pos_np, link_quat, local_offset)
        axes = link_local_axes_world(link_quat)

        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        dist0 = float(np.linalg.norm(target_w - ee_pos))

        print(f"[INFO] Link probe prim: {link_prim}")
        print(f"[INFO] link_1 world pos={np.round(link_pos_np, 4).tolist()}")
        print(f"[INFO] link_1 quat_wxyz={[round(v, 4) for v in link_quat]}")
        print(
            f"[INFO] link local axes (world): "
            f"X={np.round(axes['x'], 3).tolist()}, "
            f"Y={np.round(axes['y'], 3).tolist()}, "
            f"Z={np.round(axes['z'], 3).tolist()}"
        )
        print(
            f"[INFO] Probe local_offset={local_offset} -> world={np.round(target_w, 4).tolist()} "
            f"(EE start={np.round(ee_pos, 4).tolist()}, dist={dist0:.3f}m)"
        )

        self.sim_step_count = 0
        self.control_step_count = 0
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        self._move_to_position(collector, target_w, max_servo_steps, on_control_step=None)

        for _ in range(hold_control_steps):
            ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
            arm_targets = self._ik_targets_for_position(target_w)
            self._control_step_ik(collector, arm_targets, gripper_open=False, on_control_step=None)

        ee_final = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        err = float(np.linalg.norm(target_w - ee_final))
        print(
            f"[INFO] Link probe finished: EE={np.round(ee_final, 4).tolist()}, "
            f"target={np.round(target_w, 4).tolist()}, err={err:.4f}m"
        )
        return target_w, ee_final, err

    def run_top_contact_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        max_servo_steps: int = 400,
        hold_control_steps: int = 120,
    ) -> tuple[ContactCandidate | None, np.ndarray, np.ndarray, float]:
        """Move EE to top-ranked mesh contact approach point (scene.usd workspace)."""
        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_movable_link_pose()
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()

        ranked = prepare_ranked_contact_candidates(
            self.sampling_config,
            link_pos_np,
            link_quat,
            ee_pos,
            rng=self.rng,
        )
        if not ranked:
            print("[WARN] No contact candidates after scene workspace filter.")
            return None, ee_pos, ee_pos, float("inf")

        top = ranked[0]
        geom = candidate_world_geometry(top, link_pos_np, link_quat)
        target_w = np.asarray(geom["approach_w"], dtype=np.float64)
        contact_w = np.asarray(geom["contact_w"], dtype=np.float64)
        approach_dir = scene_approach_direction(ee_pos, link_pos_np, self.sampling_config)

        if not contact_passes_sanity(top, link_pos_np, link_quat, self.sampling_config):
            print(
                f"[WARN] Top candidate failed sanity (contact_w={np.round(contact_w, 4).tolist()}, "
                f"link_y={link_pos_np[1]:.4f}); skipping arm motion."
            )
            return top, target_w, ee_pos, float("inf")

        print(f"[INFO] Scene approach dir (EE->link_1): {np.round(approach_dir, 4).tolist()}")
        source = "scene push-anchor (synthetic)" if top.fps_index < 0 else "mesh FPS"
        print(
            f"[INFO] Top contact ({source}) fps={top.fps_index} yaw={top.yaw_index}: "
            f"contact_w={np.round(contact_w, 4).tolist()}, "
            f"approach_w={np.round(target_w, 4).tolist()}, "
            f"|dy_from_link|={abs(float(contact_w[1] - link_pos_np[1])):.4f}m"
        )
        print(f"[INFO] Ranked batch size (deduped FPS): {len(ranked)}")

        self.sim_step_count = 0
        self.control_step_count = 0
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        clearance_z = float(getattr(self.push_cfg, "approach_clearance_z_m", 0.14))
        approach_path = _build_safe_approach_path(ee_pos, target_w, clearance_z_m=clearance_z)
        steps_per_wp = max(4, int(max_servo_steps) // max(1, len(approach_path) + 1))
        print(
            f"[INFO] Safe approach path ({len(approach_path)} via-points, clearance_z={clearance_z:.2f}m): "
            + " -> ".join(str(np.round(wp, 3).tolist()) for wp in approach_path)
        )
        self._follow_position_path(collector, approach_path, steps_per_wp, on_control_step=None)

        contact_path = _interp_positions(target_w, contact_w, max(2, self.push_cfg.contact_hold_steps))
        self._follow_position_path(
            collector, contact_path, max(4, self.push_cfg.contact_hold_steps), on_control_step=None
        )

        for _ in range(hold_control_steps):
            arm_targets = self._ik_targets_for_position(contact_w)
            self._control_step_ik(collector, arm_targets, gripper_open=False, on_control_step=None)

        ee_final = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        err = float(np.linalg.norm(contact_w - ee_final))
        print(
            f"[INFO] Top-contact probe finished: EE={np.round(ee_final, 4).tolist()}, "
            f"contact={np.round(contact_w, 4).tolist()}, err={err:.4f}m"
        )
        return top, target_w, ee_final, err

    def log_contact_candidate_preview(
        self,
        *,
        max_rows: int = 12,
        apply_filters: bool = True,
    ) -> list[ContactCandidate]:
        """Sample + filter + rank candidates; print world-frame geometry."""
        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_movable_link_pose()
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()

        raw = sample_contact_candidates(self.sampling_config, rng=self.rng)
        raw_count = len(raw)
        if apply_filters:
            candidates = prepare_ranked_contact_candidates(
                self.sampling_config,
                link_pos_np,
                link_quat,
                ee_pos,
                rng=self.rng,
            )
        else:
            candidates = rank_contact_candidates(
                raw, link_pos_np, link_quat, ee_pos, self.sampling_config
            )

        approach_dir = scene_approach_direction(ee_pos, link_pos_np, self.sampling_config)
        print(
            f"[INFO] Contact candidates: raw={raw_count}, ranked={len(candidates)} "
            f"(scene_dir={np.round(approach_dir, 3).tolist()}, "
            f"z=[{self.sampling_config.min_contact_world_z_m}, {self.sampling_config.max_contact_world_z_m}], "
            f"x>={self.sampling_config.min_contact_world_x_m})"
        )
        for row in summarize_candidates_world(
            candidates, link_pos_np, link_quat, max_rows=max_rows
        ):
            print(
                f"  fps={row['fps']} yaw={row['yaw']} "
                f"contact_w={row['contact_world']} approach_w={row['approach_world']}"
            )
        return candidates

    def _recording_context(self) -> RecordingContext:
        return RecordingContext(
            device=self.device,
            sim_dt=self.timing.sim_dt,
            sim_step_count=self.sim_step_count,
            control_step_count=self.control_step_count,
            vision_decimation=self.timing.vision_decimation,
            episode_start_wall_time=self.episode_start_wall_time,
            arm_joint_ids=self.handles.arm_joint_ids,
            ee_body_id=self.handles.ee_body_id,
        )

    def _sim_substeps(self, render: bool = True) -> None:
        sim = self.env_module.sim
        for _ in range(self.timing.control_decimation):
            sim.step(render=render)
            self.robot.update(sim.cfg.dt)
            if self.env_module.scene_articulation is not None:
                self.env_module.scene_articulation.update(sim.cfg.dt)
            self.sim_step_count += 1

    def _reset_ee_tracking_from_robot(self) -> None:
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        ee_quat = tuple(
            float(v) for v in self.robot.data.body_quat_w[0, self.handles.ee_body_id].tolist()
        )
        self._tracking_pos = np.asarray(ee_pos, dtype=np.float64)
        self._tracking_quat_wxyz = ee_quat

    def _advance_tracking_position(
        self,
        target_pos: np.ndarray,
        max_step_m: float | None = None,
    ) -> None:
        step_m = self.max_ee_pos_step_m if max_step_m is None else max_step_m
        delta = np.asarray(target_pos, dtype=np.float64) - self._tracking_pos
        dist = float(np.linalg.norm(delta))
        if dist <= step_m or dist < 1e-9:
            self._tracking_pos = np.asarray(target_pos, dtype=np.float64)
            return
        self._tracking_pos = self._tracking_pos + delta * (step_m / dist)

    def _advance_tracking_pose(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: tuple[float, float, float, float],
        max_pos_step_m: float | None = None,
        max_rot_step_rad: float = 0.08,
    ) -> None:
        pos_step = self.max_ee_pos_step_m if max_pos_step_m is None else max_pos_step_m
        self._advance_tracking_position(target_pos, max_step_m=pos_step)

        r_curr = _wxyz_to_rot(self._tracking_quat_wxyz)
        r_tgt = _wxyz_to_rot(target_quat_wxyz)
        r_delta = r_tgt * r_curr.inv()
        rotvec = r_delta.as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-6:
            self._tracking_quat_wxyz = target_quat_wxyz
            return
        step = min(max_rot_step_rad, angle)
        r_step = R.from_rotvec(rotvec * (step / angle))
        self._tracking_quat_wxyz = _rot_to_wxyz(r_step * r_curr)

    def _ik_targets_for_position(self, target_pos: np.ndarray) -> torch.Tensor:
        target_pos_t = torch.tensor([target_pos.tolist()], dtype=torch.float32, device=self.device)
        ee_quat = self.robot.data.body_quat_w[:, self.handles.ee_body_id]
        self.diff_ik_pos_controller.set_command(target_pos_t, ee_quat=ee_quat)
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.handles.ee_body_id, :, self.handles.arm_joint_ids
        ]
        return self.diff_ik_pos_controller.compute(
            ee_pos=self.robot.data.body_pos_w[:, self.handles.ee_body_id],
            ee_quat=ee_quat,
            jacobian=jacobian,
            joint_pos=self.robot.data.joint_pos[:, self.handles.arm_joint_ids],
        )

    def _ik_targets_for_pose(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: tuple[float, float, float, float],
    ) -> torch.Tensor:
        target_pos_t = torch.tensor([target_pos.tolist()], dtype=torch.float32, device=self.device)
        target_quat_t = torch.tensor([list(target_quat_wxyz)], dtype=torch.float32, device=self.device)
        self.diff_ik_controller.set_command(torch.cat([target_pos_t, target_quat_t], dim=-1))
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.handles.ee_body_id, :, self.handles.arm_joint_ids
        ]
        return self.diff_ik_controller.compute(
            ee_pos=self.robot.data.body_pos_w[:, self.handles.ee_body_id],
            ee_quat=self.robot.data.body_quat_w[:, self.handles.ee_body_id],
            jacobian=jacobian,
            joint_pos=self.robot.data.joint_pos[:, self.handles.arm_joint_ids],
        )

    def _clamp_arm_targets(self, arm_targets: torch.Tensor) -> torch.Tensor:
        current = self.robot.data.joint_pos[0, self.handles.arm_joint_ids]
        delta = arm_targets[0] - current
        delta = torch.clamp(delta, -self.max_joint_step_rad, self.max_joint_step_rad)
        return (current + delta).unsqueeze(0)

    def _apply_arm_command(self, arm_targets: torch.Tensor, gripper_open: bool) -> None:
        gripper_targets = self.gripper_open_target if gripper_open else self.gripper_close_target
        self.robot.set_joint_position_target(arm_targets, joint_ids=self.handles.arm_joint_ids)
        self.robot.set_joint_position_target(gripper_targets, joint_ids=self.handles.gripper_joint_ids)
        self.robot.write_data_to_sim()

    def _reassert_gripper_closed(self) -> None:
        """Keep gripper commanded closed after physics (matches Keyboard_collection)."""
        self.robot.set_joint_position_target(
            self.gripper_close_target, joint_ids=self.handles.gripper_joint_ids
        )
        self.robot.write_data_to_sim()

    def _maybe_record(
        self,
        collector: OfficialEpisodeCollector,
        arm_targets: torch.Tensor,
        gripper_open: bool,
    ) -> bool:
        if self._skip_recording:
            return False
        ctx = self._recording_context()
        rgb_main, rgb_wrist = capture_rgb_if_due(self.env_module, ctx)
        if rgb_main is None or rgb_wrist is None:
            return False
        obs_dict, action, reward, done, state_dict = build_step_tensors(
            self.robot, ctx, arm_targets, gripper_open, rgb_main, rgb_wrist
        )
        collector.add_step(obs_dict, action, reward, done, state_dict)
        self.last_record = StepRecord(obs_dict=obs_dict, action=action, reward=reward, state_dict=state_dict)
        return True

    def _control_step_ik(
        self,
        collector: OfficialEpisodeCollector,
        arm_targets: torch.Tensor,
        gripper_open: bool,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        *,
        clamp_joints: bool = False,
    ) -> bool:
        if clamp_joints:
            arm_targets = self._clamp_arm_targets(arm_targets)
        self.control_step_count += 1
        self._apply_arm_command(arm_targets, gripper_open)
        self._sim_substeps()
        if not gripper_open:
            self._reassert_gripper_closed()
        self._maybe_record(collector, arm_targets, gripper_open)
        if on_control_step is None:
            return False
        success_now, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
        return on_control_step(success_now, joint_degs)

    def _move_to_position(
        self,
        collector: OfficialEpisodeCollector,
        target_pos: np.ndarray,
        max_steps: int,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
    ) -> bool:
        """Position-only servo (hold wrist orientation) — stable approach like keyboard teleop."""
        target_pos = np.asarray(target_pos, dtype=np.float64)
        for _ in range(max_steps):
            ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
            if float(np.linalg.norm(target_pos - ee_pos)) <= POSITION_REACH_TOL_M:
                return False

            self._advance_tracking_position(target_pos)
            arm_targets = self._ik_targets_for_position(self._tracking_pos)
            if self._control_step_ik(collector, arm_targets, gripper_open=False, on_control_step=on_control_step):
                return True
        return False

    def _move_to_pose(
        self,
        collector: OfficialEpisodeCollector,
        target_pos: np.ndarray,
        target_quat_wxyz: tuple[float, float, float, float],
        max_steps: int,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        *,
        clamp_joints: bool = False,
        pos_tol_m: float | None = None,
        rot_tol_rad: float | None = None,
    ) -> bool:
        """Pose servo with incremental target advance (position + rotation)."""
        target_pos = np.asarray(target_pos, dtype=np.float64)
        pos_tol = POSE_REACH_TOL_M if pos_tol_m is None else float(pos_tol_m)
        rot_tol = POSE_REACH_ROT_RAD if rot_tol_rad is None else float(rot_tol_rad)
        for _ in range(max_steps):
            ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
            ee_quat = tuple(
                float(v) for v in self.robot.data.body_quat_w[0, self.handles.ee_body_id].tolist()
            )
            pos_err = float(np.linalg.norm(target_pos - ee_pos))
            r_err = (_wxyz_to_rot(target_quat_wxyz) * _wxyz_to_rot(ee_quat).inv()).magnitude()
            if pos_err <= pos_tol and r_err <= rot_tol:
                return False

            self._advance_tracking_pose(target_pos, target_quat_wxyz)
            arm_targets = self._ik_targets_for_pose(self._tracking_pos, self._tracking_quat_wxyz)
            if self._control_step_ik(
                collector,
                arm_targets,
                gripper_open=False,
                on_control_step=on_control_step,
                clamp_joints=clamp_joints,
            ):
                return True
        return False

    def _follow_position_path(
        self,
        collector: OfficialEpisodeCollector,
        waypoints: list[np.ndarray],
        steps_per_waypoint: int,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
    ) -> bool:
        for waypoint in waypoints:
            if self._move_to_position(collector, waypoint, steps_per_waypoint, on_control_step):
                return True
        return False

    def _follow_pose_path(
        self,
        collector: OfficialEpisodeCollector,
        poses: list[tuple[np.ndarray, tuple[float, float, float, float]]],
        steps_per_waypoint: int,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        *,
        clamp_joints: bool = False,
    ) -> bool:
        for pos, quat in poses:
            if self._move_to_pose(
                collector,
                pos,
                quat,
                steps_per_waypoint,
                on_control_step,
                clamp_joints=clamp_joints,
            ):
                return True
        return False

    def _restore_arm_home(self, collector: OfficialEpisodeCollector, *, record: bool) -> None:
        from motion_planner import interpolate_joint_segment

        current_arm = tuple(
            float(v) for v in self.robot.data.joint_pos[0, self.handles.arm_joint_ids].tolist()
        )
        for joint_rad in interpolate_joint_segment(current_arm, self.home_arm_rad, self.candidate_home_steps):
            self.control_step_count += 1
            arm_targets = torch.tensor([list(joint_rad)], dtype=torch.float32, device=self.device)
            self._apply_arm_command(arm_targets, gripper_open=False)
            self._sim_substeps()
            if record:
                self._maybe_record(collector, arm_targets, gripper_open=False)
        self._reset_ee_tracking_from_robot()

    def _try_candidate(
        self,
        collector: OfficialEpisodeCollector,
        candidate: ContactCandidate,
        on_control_step: Callable[[bool, dict[str, float | None]], bool],
        max_approach_distance_m: float = 0.85,
        log_geometry: bool = False,
    ) -> bool:
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_movable_link_pose()

        approach_w, contact_w, quat_w = link_to_world_candidate(candidate, link_pos_np, link_quat)

        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        approach_dist = float(np.linalg.norm(approach_w - ee_pos))
        if approach_dist > max_approach_distance_m:
            print(
                f"[WARN] Skip candidate fps={candidate.fps_index} yaw={candidate.yaw_index}: "
                f"approach {approach_dist:.2f}m > {max_approach_distance_m}m"
            )
            return False

        if log_geometry:
            print(
                f"[INFO] link_1 pos={np.round(link_pos_np, 3).tolist()} "
                f"quat_wxyz={[round(v, 3) for v in link_quat]}"
            )
            print(
                f"[INFO] EE pos={np.round(ee_pos, 3).tolist()}, "
                f"approach={np.round(approach_w, 3).tolist()} ({approach_dist:.3f}m), "
                f"contact={np.round(contact_w, 3).tolist()}"
            )

        approach_waypoints = _build_safe_approach_path(
            ee_pos,
            approach_w,
            clearance_z_m=float(getattr(self.push_cfg, "approach_clearance_z_m", 0.14)),
        )
        steps_per_wp = max(4, self.push_cfg.approach_steps)
        if self._follow_position_path(collector, approach_waypoints, steps_per_wp, on_control_step=None):
            return True

        contact_waypoints = _interp_positions(approach_w, contact_w, max(2, self.push_cfg.contact_hold_steps))
        if self._follow_position_path(collector, contact_waypoints, max(4, self.push_cfg.contact_hold_steps), None):
            return True

        actual_contact_pos, actual_contact_quat = self._read_ee_pose()
        self._reset_close_phase_tracking()
        self._init_close_anchor(actual_contact_pos, actual_contact_quat)
        close_poses = self._sample_anchored_close_poses()
        if not close_poses:
            return False
        return self._execute_hinge_close_anchored(collector, on_control_step)

    def _resolve_keyboard_reference_hdf5(self) -> Path:
        ref = self.push_cfg.keyboard_reference_hdf5 or self.push_cfg.debug_reference_hdf5
        if not ref:
            raise ValueError("keyboard_aligned requires push.keyboard_reference_hdf5 in task yaml.")
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"keyboard_reference_hdf5 not found: {path}")
        return path

    def _load_keyboard_reference_delta(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load (ref_ee0, ref_contact, push_dir_unit) from keyboard success HDF5."""
        import h5py

        path = self._resolve_keyboard_reference_hdf5()
        demo_idx = int(self.push_cfg.keyboard_reference_demo)
        demo_key = f"demo_{demo_idx}"
        with h5py.File(path, "r") as h5_file:
            if demo_key not in h5_file["data"]:
                raise KeyError(f"Reference demo '{demo_key}' not in {path}")
            ee = h5_file[f"data/{demo_key}/obs/robot_eef_pos"][:, 0].astype(float)
        ref_ee0 = ee[0]
        peak_idx = int(np.argmax(np.linalg.norm(ee - ref_ee0, axis=1)))
        ref_contact = ee[peak_idx]
        push_dir = ref_contact - ref_ee0
        push_norm = float(np.linalg.norm(push_dir))
        if push_norm < 1e-6:
            raise ValueError(f"Reference demo {demo_key} has no EE motion.")
        push_dir = push_dir / push_norm
        print(
            f"[INFO] Keyboard reference {path.name} {demo_key}: "
            f"peak frame {peak_idx}, contact={np.round(ref_contact, 3).tolist()}, "
            f"dir={np.round(push_dir, 3).tolist()}"
        )
        return ref_ee0, ref_contact, push_dir

    def _keyboard_waypoints_for_ee_start(
        self, ee_start: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref_ee0, ref_contact, push_dir = self._load_keyboard_reference_delta()
        delta_contact = ref_contact - ref_ee0
        contact_w = ee_start + delta_contact
        approach_w = contact_w - push_dir * float(self.push_cfg.approach_backoff_m)
        return approach_w, contact_w, push_dir

    def _servo_to_position_keyboard(
        self,
        collector: OfficialEpisodeCollector,
        goal_pos: np.ndarray,
        max_steps: int,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        *,
        label: str,
    ) -> bool:
        """Position-only incremental IK — wrist free to reorient (matches keyboard reach)."""
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        self._tracking_pos = np.asarray(
            self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy(),
            dtype=np.float64,
        )
        for step in range(max_steps):
            ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
            err = float(np.linalg.norm(goal_pos - ee_pos))
            if err <= POSITION_REACH_TOL_M:
                print(f"[INFO] Keyboard servo reached {label} at step {step}, err={err:.4f}m")
                return False

            self._advance_tracking_position(goal_pos)
            arm_targets = self._ik_targets_for_position(self._tracking_pos)
            if self._control_step_ik(
                collector,
                arm_targets,
                gripper_open=False,
                on_control_step=on_control_step,
                clamp_joints=True,
            ):
                return True
            if step > 0 and (step + 1) % 80 == 0:
                print(
                    f"[INFO] Keyboard {label} step {step + 1}: "
                    f"EE={np.round(ee_pos, 3).tolist()} err={err:.4f}m"
                )
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        err = float(np.linalg.norm(goal_pos - ee_pos))
        print(
            f"[WARN] Keyboard servo timed out for {label} after {max_steps} steps "
            f"(EE={np.round(ee_pos, 3).tolist()}, err={err:.4f}m)"
        )
        return False

    def _push_close_keyboard(
        self,
        collector: OfficialEpisodeCollector,
        start_pos: np.ndarray,
        push_dir: np.ndarray,
        on_control_step: Callable[[bool, dict[str, float | None]], bool],
    ) -> bool:
        """Continue pushing along keyboard direction (position-only IK)."""
        push_dir = np.asarray(push_dir, dtype=np.float64)
        push_dir = push_dir / max(float(np.linalg.norm(push_dir)), 1e-9)
        self._tracking_pos = np.asarray(start_pos, dtype=np.float64)
        total_pushed = 0.0
        target_dist = float(self.push_cfg.close_push_distance_m)
        for step in range(int(self.push_cfg.num_close_steps)):
            if total_pushed >= target_dist:
                break
            step_m = min(self.max_ee_pos_step_m, target_dist - total_pushed)
            self._tracking_pos = self._tracking_pos + push_dir * step_m
            total_pushed += step_m
            arm_targets = self._ik_targets_for_position(self._tracking_pos)
            if self._control_step_ik(
                collector,
                arm_targets,
                gripper_open=False,
                on_control_step=on_control_step,
                clamp_joints=True,
            ):
                return True
        return False

    def run_keyboard_aligned_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
        *,
        reference_demo: int | None = None,
    ) -> tuple[bool, dict[str, float | None]]:
        """Phase A: replay keyboard success demo or (fallback) EE servo."""
        mode = getattr(self.push_cfg, "keyboard_control_mode", "joint_replay")
        if mode == "ee_servo":
            return self._run_keyboard_ee_servo_push(collector, episode_start_wall_time)
        return self._run_keyboard_joint_replay_push(
            collector, episode_start_wall_time, reference_demo=reference_demo
        )

    def _run_keyboard_joint_replay_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
        *,
        reference_demo: int | None = None,
    ) -> tuple[bool, dict[str, float | None]]:
        """Open-loop joint replay from keyboard_reference_hdf5 (matches teleop joint path)."""
        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = episode_start_wall_time
        self.last_record = None

        demo_idx = (
            int(reference_demo)
            if reference_demo is not None
            else int(self.push_cfg.keyboard_reference_demo)
        )
        trajectory = self._load_reference_arm_trajectory(demo_idx)
        if not trajectory:
            raise RuntimeError(
                f"Empty keyboard reference trajectory for demo_{demo_idx}. "
                f"Check keyboard_reference_hdf5 in task yaml."
            )

        print(
            f"[INFO] Keyboard joint replay: demo_{demo_idx}, {len(trajectory)} frames "
            f"(direct set_joint_position_target, no IK)."
        )
        return self._execute_arm_trajectory(
            collector, trajectory, label=f"keyboard_replay_demo_{demo_idx}"
        )

    def _run_keyboard_ee_servo_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
    ) -> tuple[bool, dict[str, float | None]]:
        """Legacy: incremental position IK toward keyboard EE waypoints (can bow / stall)."""
        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = episode_start_wall_time
        self.last_record = None
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        ee_start = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        approach_w, contact_w, push_dir = self._keyboard_waypoints_for_ee_start(ee_start)
        max_steps = int(self.push_cfg.max_servo_steps_per_phase)

        print(
            f"[INFO] Keyboard EE servo push: EE start={np.round(ee_start, 3).tolist()}, "
            f"approach={np.round(approach_w, 3).tolist()}, "
            f"contact={np.round(contact_w, 3).tolist()}"
        )

        final_joint_degs: dict[str, float | None] = {}

        def on_step(success_now: bool, joint_degs: dict[str, float | None]) -> bool:
            nonlocal final_joint_degs
            final_joint_degs = joint_degs
            return success_now

        if self._servo_to_position_keyboard(
            collector, approach_w, max_steps, on_step, label="approach"
        ):
            return True, final_joint_degs

        if self._servo_to_position_keyboard(
            collector, contact_w, max_steps, on_step, label="contact"
        ):
            return True, final_joint_degs

        ee_at_contact = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        if self._push_close_keyboard(collector, ee_at_contact, push_dir, on_step):
            return True, final_joint_degs

        success_now, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
        final_joint_degs = joint_degs
        print(f"[INFO] Keyboard EE servo finished: success={success_now}, joints={joint_degs}")
        return success_now, final_joint_degs

    def _rank_candidates_by_approach_distance(self) -> list[ContactCandidate]:
        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_movable_link_pose()
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        return prepare_ranked_contact_candidates(
            self.sampling_config,
            link_pos_np,
            link_quat,
            ee_pos,
            rng=self.rng,
        )

    def run_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
        *,
        reference_demo: int | None = None,
    ) -> tuple[bool, dict[str, float | None]]:
        if getattr(self.push_cfg, "push_strategy", "articubot") == "keyboard_aligned":
            return self.run_keyboard_aligned_push(
                collector, episode_start_wall_time, reference_demo=reference_demo
            )
        if getattr(self.push_cfg, "push_strategy", "articubot") == "yaml_handle":
            return self.run_yaml_handle_push(collector, episode_start_wall_time)
        if getattr(self.push_cfg, "push_strategy", "articubot") == "articulation_calibrated":
            return self.run_articulation_calibrated_push(collector, episode_start_wall_time)

        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = episode_start_wall_time
        self.last_record = None
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        link_prim = self.task_config.link_prim
        link_pos_np, link_quat = self._read_movable_link_pose()
        ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()

        self._all_candidates = prepare_ranked_contact_candidates(
            self.sampling_config,
            link_pos_np,
            link_quat,
            ee_pos,
            rng=self.rng,
        )
        print(
            f"[INFO] Ranked {len(self._all_candidates)} contact candidates "
            f"({self.sampling_config.num_fps_points}×{self.sampling_config.num_yaw_perturbations} mesh samples, "
            f"scene workspace filter)."
        )

        final_joint_degs: dict[str, float | None] = {}
        peak_joint_degs: dict[str, float | None] = {spec.joint_prim: None for spec in self.success_specs}

        def on_step(success_now: bool, joint_degs: dict[str, float | None]) -> bool:
            nonlocal final_joint_degs
            final_joint_degs = joint_degs
            update_peak_joint_degs(peak_joint_degs, joint_degs)
            return success_now

        max_try = min(self.push_cfg.max_candidates_to_try, len(self._all_candidates))
        for idx in range(max_try):
            candidate = self._all_candidates[idx]
            print(
                f"[INFO] Trying candidate {idx + 1}/{max_try} "
                f"(fps={candidate.fps_index}, yaw={candidate.yaw_index})"
            )
            if self._try_candidate(
                collector,
                candidate,
                on_step,
                max_approach_distance_m=self.push_cfg.max_approach_distance_m,
                log_geometry=(idx == 0),
            ):
                return True, final_joint_degs

            if idx + 1 < max_try:
                print("[INFO] Candidate failed — restoring arm home before next candidate.")
                self._restore_arm_home(collector, record=False)

        success_now, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
        final_joint_degs = joint_degs
        return success_now, final_joint_degs

    def _resolve_reference_hdf5(self) -> Path | None:
        ref = self.push_cfg.keyboard_reference_hdf5 or self.push_cfg.debug_reference_hdf5
        if not ref:
            return None
        path = Path(ref).expanduser()
        if not path.is_absolute():
            collect_dir = Path(__file__).resolve().parent
            path = (collect_dir / path).resolve()
        return path

    def _resolve_debug_reference_hdf5(self) -> Path | None:
        return self._resolve_reference_hdf5()

    def _load_reference_arm_trajectory(self, demo_idx: int | None = None) -> list[tuple[float, ...]]:
        return self._load_debug_reference_arm_trajectory(demo_idx)

    def _load_debug_reference_arm_trajectory(self, demo_idx: int | None = None) -> list[tuple[float, ...]]:
        import h5py

        path = self._resolve_reference_hdf5()
        if path is None:
            return []

        if not path.is_file():
            raise FileNotFoundError(f"debug_reference_hdf5 not found: {path}")

        if demo_idx is None:
            demo_idx = int(self.push_cfg.debug_reference_demo)
        demo_key = f"demo_{demo_idx}"
        if demo_idx in self._debug_trajectory_cache:
            cached = self._debug_trajectory_cache[demo_idx]
            print(
                f"[INFO] Using cached DEBUG reference trajectory: {demo_key}, "
                f"{len(cached)} frames."
            )
            return list(cached)

        stride = max(1, int(self.push_cfg.debug_reference_stride))
        max_frames = self.push_cfg.debug_reference_max_frames

        print(
            f"[INFO] Loading DEBUG reference trajectory {path.name} {demo_key}...",
            flush=True,
        )
        with h5py.File(path, "r") as h5_file:
            if "data" not in h5_file or demo_key not in h5_file["data"]:
                available = list(h5_file["data"].keys()) if "data" in h5_file else []
                raise KeyError(
                    f"Reference demo '{demo_key}' not in {path}. Available: {available}"
                )
            joint_ds = h5_file[f"data/{demo_key}/obs/robot_joint_pos"]
            total = int(joint_ds.shape[0])
            end = min(total, max_frames) if max_frames is not None else total
            trajectory: list[tuple[float, ...]] = []
            for frame_idx in range(0, end, stride):
                joints = tuple(float(v) for v in joint_ds[frame_idx, 0, :6])
                trajectory.append(joints)

        print(
            f"[INFO] Loaded DEBUG reference trajectory: {path.name} {demo_key}, "
            f"{len(trajectory)} frames (stride={stride}, source_frames={end})."
        )
        if trajectory:
            ee_key = f"data/{demo_key}/obs/robot_eef_pos"
            with h5py.File(path, "r") as h5_file:
                ee0 = [float(v) for v in h5_file[ee_key][0, 0]]
                ee1 = [float(v) for v in h5_file[ee_key][min(end - 1, total - 1), 0]]
            print(f"[INFO] Reference EE start={np.round(ee0, 3).tolist()} end={np.round(ee1, 3).tolist()}")
        return trajectory

    def _execute_arm_trajectory(
        self,
        collector: OfficialEpisodeCollector,
        trajectory: list[tuple[float, ...]],
        *,
        label: str,
    ) -> tuple[bool, dict[str, float | None]]:
        final_joint_degs: dict[str, float | None] = {}
        for frame_idx, joint_rad in enumerate(trajectory):
            if len(joint_rad) != 6:
                raise ValueError(f"{label} frame {frame_idx} must have 6 arm joints, got {len(joint_rad)}")
            self.control_step_count += 1
            arm_targets = torch.tensor([list(joint_rad)], dtype=torch.float32, device=self.device)
            self._apply_arm_command(arm_targets, gripper_open=False)
            self._sim_substeps()
            self._maybe_record(collector, arm_targets, gripper_open=False)
            success_now, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
            final_joint_degs = joint_degs
            if frame_idx == 0 or frame_idx + 1 == len(trajectory) or (frame_idx + 1) % 80 == 0:
                ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
                print(
                    f"[INFO] DEBUG {label} frame {frame_idx + 1}/{len(trajectory)}: "
                    f"EE={np.round(ee_pos, 3).tolist()}"
                )
            if success_now:
                print(f"[INFO] {label} success at frame {frame_idx + 1}: {joint_degs}")
                return True, final_joint_degs

        success_now, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
        final_joint_degs = joint_degs
        print(f"[INFO] {label} finished: success={success_now}, joints={joint_degs}")
        return success_now, final_joint_degs

    def run_hardcoded_debug_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
        *,
        reference_demo: int | None = None,
    ) -> tuple[bool, dict[str, float | None]]:
        """Joint-space debug push — replay reference HDF5 or interpolate yaml waypoints."""
        from motion_planner import interpolate_joint_segment

        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_start_wall_time = episode_start_wall_time
        self.last_record = None

        ee_pos0 = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
        print(f"[INFO] DEBUG start EE pos={np.round(ee_pos0, 3).tolist()}")

        reference_trajectory = self._load_debug_reference_arm_trajectory(reference_demo)
        if reference_trajectory:
            demo_label = reference_demo if reference_demo is not None else self.push_cfg.debug_reference_demo
            print(
                f"[INFO] DEBUG mode: replay reference HDF5 demo_{demo_label} "
                "(keyboard joint trajectory toward +X/-Z)."
            )
            return self._execute_arm_trajectory(collector, reference_trajectory, label=f"replay_demo_{demo_label}")

        if self.push_cfg.debug_reference_hdf5:
            demo_label = reference_demo if reference_demo is not None else self.push_cfg.debug_reference_demo
            raise RuntimeError(
                f"Empty DEBUG reference trajectory for demo_{demo_label}. "
                f"Check push.debug_reference_hdf5 in task yaml."
            )

        waypoints = self.push_cfg.debug_joint_waypoints
        if not waypoints:
            raise ValueError(
                "Set push.debug_reference_hdf5 or push.debug_joint_waypoints in task yaml."
            )

        steps_per_segment = max(1, int(self.push_cfg.debug_steps_per_waypoint))
        print(
            f"[INFO] DEBUG hardcoded joint-space push: {len(waypoints)} waypoints, "
            f"{steps_per_segment} steps/segment (no sampling/IK)."
        )

        expanded: list[tuple[float, ...]] = []
        current_arm = tuple(
            float(v) for v in self.robot.data.joint_pos[0, self.handles.arm_joint_ids].tolist()
        )
        for wp_idx, target in enumerate(waypoints):
            if len(target) != 6:
                raise ValueError(f"debug_joint_waypoints[{wp_idx}] must have 6 joints, got {len(target)}")
            target_arm = tuple(float(v) for v in target)
            segment = interpolate_joint_segment(current_arm, target_arm, steps_per_segment)
            expanded.extend(tuple(float(v) for v in joints) for joints in segment)
            current_arm = target_arm

        return self._execute_arm_trajectory(collector, expanded, label="waypoints")

    def run_home_reset(self, collector: OfficialEpisodeCollector) -> None:
        self._restore_arm_home(collector, record=True)
