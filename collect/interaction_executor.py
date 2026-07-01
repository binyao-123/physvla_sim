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
    link_local_axes_world,
    prepare_ranked_contact_candidates,
    rank_contact_candidates,
    sample_contact_candidates,
    scene_approach_direction,
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
from reference.grasping_utils import align_gripper_z_with_normal
from reference.handle_reference import derive_contact_quat_link, resolve_yaml_handle_world
from reference.opening_kinematics import (
    compute_articulation_ee_trajectory,
    compose_pose,
    invert_pose,
)
from collection_health import HealthLimits, RecordingHealthError, check_step_payload
from recording_utils import RecordingContext, build_step_tensors, capture_rgb_if_due
from success_utils import evaluate_rollout_success, update_peak_joint_degs
from task_registry import (
    CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ,
    CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_TRANSLATION,
    CLOSE_LAPTOP_TASK_ID,
    CLOSE_MICROWAVE_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ,
    CLOSE_MICROWAVE_HANDLE_CALIBRATION_SCENE_TRANSLATION,
    CLOSE_MICROWAVE_TASK_ID,
    ADJUST_FAUCET_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ,
    ADJUST_FAUCET_HANDLE_CALIBRATION_SCENE_TRANSLATION,
    ADJUST_FAUCET_JOINT_INITIAL_BASE_DEG,
    ADJUST_FAUCET_TASK_ID,
    SHARED_TELEOP_PIPER,
)

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


class EpisodeStepLimitExceeded(RuntimeError):
    """Raised when one collection/probe episode exceeds its control-step budget."""


class EeHandleDistExceeded(RuntimeError):
    """EE drifted too far from the live contact anchor during push/close."""

    def __init__(self, dist_m: float, phase: str):
        self.dist_m = float(dist_m)
        self.phase = str(phase)
        super().__init__(
            f"EE-handle dist {self.dist_m * 1000:.1f}mm exceeds "
            f"{EE_HANDLE_ABORT_DIST_M * 1000:.0f}mm (phase={self.phase})"
        )


# Keyboard teleop matched defaults (150830 demo: EE max step ~0.0067 m).
KEYBOARD_EE_POS_STEP_M = 0.005
KEYBOARD_JOINT_STEP_RAD = 0.02
POSITION_REACH_TOL_M = 0.012
CALIBRATED_XY_ARC_TASK_IDS = frozenset({CLOSE_MICROWAVE_TASK_ID, ADJUST_FAUCET_TASK_ID})


def _interp_extrap_scalar(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Linear extrapolation outside the calibrated joint sample range."""
    if x <= float(xs[0]):
        slope = (ys[1] - ys[0]) / max(float(xs[1] - xs[0]), 1e-9)
        return float(ys[0] + (x - float(xs[0])) * slope)
    if x >= float(xs[-1]):
        slope = (ys[-1] - ys[-2]) / max(float(xs[-1] - xs[-2]), 1e-9)
        return float(ys[-1] + (x - float(xs[-1])) * slope)
    return float(np.interp(x, xs, ys))

POSE_REACH_TOL_M = 0.01
POSE_REACH_ROT_RAD = 0.15
# Abort push/close when EE drifts farther than this from the live contact anchor.
EE_HANDLE_ABORT_DIST_M = 0.03
# Close-phase contact servo: keep EE near the surface point captured at first contact.
CLOSE_CONTACT_ANCHOR_GAIN = 0.8
CLOSE_CONTACT_ANCHOR_MAX_CORRECTION_M = 0.1
# Short articulation tasks need contact correction to activate before the EE drifts far off the handle.
CLOSE_CONTACT_ANCHOR_DEADBAND_M = 0.02
# Calibrated XY arc tasks need tighter anchor pull-in; otherwise ~7mm gaps are ignored.
CLOSE_CONTACT_ANCHOR_DEADBAND_CALIBRATED_M = 0.005
# For calibrated close, use gentler correction to reduce high-frequency IK jitter,
# but keep enough pull-in so EE does not drift away from handle arc.
CLOSE_CONTACT_ANCHOR_GAIN_CALIBRATED = 0.3
CLOSE_CONTACT_ANCHOR_MAX_CORRECTION_CALIBRATED_M = 0.03
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
    """Directly move to the handle approach pose; no extra upward cruise waypoint."""
    del ee_pos, clearance_z_m
    return [np.asarray(approach_w, dtype=np.float64)]


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
        verbose: bool = False,
        trace_ee_handle: bool = False,
        trace_ee_handle_interval: int = 30,
        health_limits: HealthLimits | None = None,
        health_checks_enabled: bool = True,
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
        self.max_ee_pos_step_m = float(getattr(push_config, "max_ee_pos_step_m", KEYBOARD_EE_POS_STEP_M))
        self.max_joint_step_rad = float(getattr(push_config, "max_joint_step_rad", KEYBOARD_JOINT_STEP_RAD))
        self._touch_contact_ref = None
        self._close_anchor: dict[str, object] | None = None
        self._close_hinge_lever_m: float | None = None

        self.sim_step_count = 0
        self.control_step_count = 0
        self.episode_step_limit: int | None = None
        self.episode_step_limit_hit = False
        self.episode_start_wall_time = 0.0
        self.last_record: StepRecord | None = None
        self._last_arm_targets: torch.Tensor | None = None

        home_joints = SHARED_TELEOP_PIPER.joint_pos_dict()
        self.home_arm_rad = tuple(float(home_joints[f"joint{i}"]) for i in range(1, 7))
        self._tracking_pos = np.zeros(3, dtype=np.float64)
        self._tracking_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self._skip_recording = False
        self.health_limits = health_limits or HealthLimits()
        self.health_checks_enabled = bool(health_checks_enabled)
        self.recording_health_failed = False
        self._recording_health_reason = ""
        self.verbose = verbose
        self.trace_ee_handle = trace_ee_handle
        self.trace_ee_handle_interval = max(1, int(trace_ee_handle_interval))
        self.progress_log_interval = 50
        self._trace_phase = ""
        self._scene_root_map_logged = False
        self._last_arc_fit_log_joint: float | None = None

    def _reset_episode_log_state(self) -> None:
        self._scene_root_map_logged = False
        self._last_arc_fit_log_joint = None

    def set_trace_phase(self, phase: str) -> None:
        self._trace_phase = phase

    def reset_recording_health(self) -> None:
        self.recording_health_failed = False
        self._recording_health_reason = ""

    def set_episode_step_limit(self, limit: int | None) -> None:
        if limit is None or int(limit) <= 0:
            self.episode_step_limit = None
        else:
            self.episode_step_limit = int(limit)
        self.episode_step_limit_hit = False

    def _remaining_episode_control_steps(self) -> int | None:
        if self.episode_step_limit is None:
            return None
        return max(0, int(self.episode_step_limit) - int(self.control_step_count))

    def _raise_if_episode_step_limit_reached(self) -> None:
        if self.episode_step_limit is None:
            return
        if self.control_step_count >= self.episode_step_limit:
            self.episode_step_limit_hit = True
            raise EpisodeStepLimitExceeded(
                f"episode control-step limit reached: "
                f"{self.control_step_count}/{self.episode_step_limit}"
            )

    def _episode_success(self) -> tuple[bool, dict[str, float | None]]:
        """Episode success: joint meets task_registry spec AND steps did not exceed limit."""
        joint_ok, joint_degs = evaluate_rollout_success(self.env_module, self.success_specs)
        within_limit = (
            self.episode_step_limit is None
            or int(self.control_step_count) <= int(self.episode_step_limit)
        )
        return joint_ok and within_limit, joint_degs

    @staticmethod
    def _stop_on_joint_success(
        success_now: bool,
        joint_degs: dict[str, float | None],
    ) -> bool:
        return success_now

    def _current_ee_handle_target_world(self) -> tuple[np.ndarray, str] | None:
        """Live anchor world target for EE distance checks during push/close only."""
        if self._trace_phase != "close":
            return None
        joint_deg = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        if joint_deg is None:
            return None
        handle_w = self._close_contact_anchor_world_at_joint(float(joint_deg))
        if handle_w is None:
            return None
        return np.asarray(handle_w, dtype=np.float64), "anchor"

    def _current_ee_handle_dist_m(self) -> float | None:
        target = self._current_ee_handle_target_world()
        if target is None:
            return None
        handle_w, _ = target
        ee_pos, _ = self._read_ee_pose()
        return float(np.linalg.norm(ee_pos - handle_w))

    def _raise_if_ee_handle_dist_exceeded(self) -> None:
        dist_m = self._current_ee_handle_dist_m()
        if dist_m is None or dist_m <= EE_HANDLE_ABORT_DIST_M:
            return
        raise EeHandleDistExceeded(dist_m, self._trace_phase)

    def _maybe_trace_ee_handle(self) -> None:
        """Periodic EE vs handle/contact-anchor world position (probe / debug)."""
        if not self.trace_ee_handle:
            return
        if self.control_step_count % self.trace_ee_handle_interval != 0:
            return
        ee_pos, _ = self._read_ee_pose()
        joint_deg = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        handle_w: np.ndarray | None = None
        handle_label = "handle"
        if self._trace_phase == "close" and joint_deg is not None:
            handle_w = self._close_contact_anchor_world_at_joint(float(joint_deg))
            handle_label = "anchor"
        if handle_w is None:
            try:
                handle_w, _, _ = self._resolve_yaml_handle_world()
            except Exception:
                pass
        phase = f" {self._trace_phase}" if self._trace_phase else ""
        msg = (
            f"[TRACE{phase}] step={self.control_step_count} "
            f"EE={np.round(ee_pos, 4).tolist()}"
        )
        if handle_w is not None:
            dist = float(np.linalg.norm(ee_pos - handle_w))
            msg += f" {handle_label}={np.round(handle_w, 4).tolist()} dist={dist:.4f}m"
        if joint_deg is not None:
            msg += f" joint={float(joint_deg):.1f}°"
        # 此处在实时打印EE位姿和锚定点位置，
        # 打印类似：[TRACE close] step=330 EE=[0.4536, -0.0686, 0.1251] anchor=[0.5515, -0.076, 0.1587] dist=0.1038m joint=104.4°
        if self.verbose:
            print(msg, flush=True)

    def _announce_phase_transition(self, from_phase: str, to_phase: str, *, detail: str = "") -> None:
        """Always-on phase boundary log (approach -> push/close)."""
        suffix = f" | {detail}" if detail else ""
        print(
            f"[INFO] Phase transition: {from_phase} -> {to_phase} "
            f"at control_step={self.control_step_count}{suffix}",
            flush=True,
        )

    def _log_debug(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _log_info(self, msg: str) -> None:
        """Verbose diagnostics (--debug-logs). Prefer _log_debug for new messages."""
        self._log_debug(msg)

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
        self._log_info(
            f"[INFO] Touch contact calibration ({path.name}): "
            f"{summarize_touch_reference(self._touch_contact_ref)}"
        )

    def _resolve_articulation_reference_hdf5(self) -> Path:
        ref = self.push_cfg.keyboard_reference_hdf5
        if not ref:
            raise ValueError(
                "articulation_calibrated requires push.keyboard_reference_hdf5 in task yaml."
            )
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent / path).resolve()
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

    def _calibration_scene_translation(self) -> tuple[float, float, float] | None:
        if self.task_config.task_id == CLOSE_LAPTOP_TASK_ID:
            return CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_TRANSLATION
        if self.task_config.task_id == CLOSE_MICROWAVE_TASK_ID:
            return CLOSE_MICROWAVE_HANDLE_CALIBRATION_SCENE_TRANSLATION
        if self.task_config.task_id == ADJUST_FAUCET_TASK_ID:
            return ADJUST_FAUCET_HANDLE_CALIBRATION_SCENE_TRANSLATION
        return None

    def _calibration_scene_rotation_xyz(self) -> tuple[float, float, float] | None:
        if self.task_config.task_id == CLOSE_LAPTOP_TASK_ID:
            return CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ
        if self.task_config.task_id == CLOSE_MICROWAVE_TASK_ID:
            return CLOSE_MICROWAVE_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ
        if self.task_config.task_id == ADJUST_FAUCET_TASK_ID:
            return ADJUST_FAUCET_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ
        return None

    def _scene_root_translation_delta_world(self) -> np.ndarray:
        """Map yaml world-calibrated handle points when the scene root moves."""
        calib = self._calibration_scene_translation()
        if calib is None:
            return np.zeros(3, dtype=np.float64)
        delta = self.env_module.get_scene_root_translation_delta(calib)
        return np.asarray(delta, dtype=np.float64)

    def _scene_root_yaw_delta_deg(self) -> float:
        """Yaw delta for mapping world-calibrated handle points under Z rotation randomization."""
        calib = self._calibration_scene_rotation_xyz()
        if calib is None:
            return 0.0
        return float(self.env_module.get_scene_root_yaw_delta_deg(calib))

    def _scene_root_yaw_deg(self) -> float:
        calib = self._calibration_scene_rotation_xyz()
        if calib is None:
            calib = CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_ROTATION_XYZ
        return float(calib[2]) + self._scene_root_yaw_delta_deg()

    def _yaw_joint_equivalent_arc_deg(self, joint_deg: float) -> float:
        """Map runtime faucet joint to yaml arc sample index.

        Yaml arc keys are hinge angles recorded at calibration yaw=90°.
        - yaw-only DR (joint_rand≈0): joint is coupled as base-Δyaw, so undo that with +Δyaw.
        - joint_rand≠0: use the live hinge angle; yaw is handled by geometric world mapping only.
        """
        if self.task_config.task_id != ADJUST_FAUCET_TASK_ID:
            return float(joint_deg)
        yaw_delta = self._scene_root_yaw_delta_deg()
        joint_rand_implied = float(joint_deg) + yaw_delta - ADJUST_FAUCET_JOINT_INITIAL_BASE_DEG
        if abs(joint_rand_implied) < 0.5:
            return float(joint_deg) + yaw_delta
        return float(joint_deg)

    def _geometric_map_calibrated_world_position(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64)
        calib_translation = self._calibration_scene_translation()
        calib_t = np.asarray(
            calib_translation or CLOSE_LAPTOP_HANDLE_CALIBRATION_SCENE_TRANSLATION,
            dtype=np.float64,
        )
        delta_t = self._scene_root_translation_delta_world()
        current_t = calib_t + delta_t
        yaw_rad = math.radians(self._scene_root_yaw_delta_deg())
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        rot_z = np.array(
            (
                (c, -s, 0.0),
                (s, c, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        return current_t + rot_z @ (pos - calib_t)

    def _yaw_anchor_bracket(self):
        anchors = sorted(
            tuple(getattr(self.sampling_config, "yaw_contact_anchors", ()) or ()),
            key=lambda p: float(p[0]),
        )
        if not anchors:
            return None
        yaw = self._scene_root_yaw_deg()
        if yaw <= float(anchors[0][0]):
            return anchors[0], anchors[0], 0.0
        if yaw >= float(anchors[-1][0]):
            return anchors[-1], anchors[-1], 0.0
        for left, right in zip(anchors[:-1], anchors[1:]):
            y0 = float(left[0])
            y1 = float(right[0])
            if y0 <= yaw <= y1:
                alpha = (yaw - y0) / max(y1 - y0, 1e-9)
                return left, right, float(alpha)
        return anchors[-1], anchors[-1], 0.0

    def _yaw_anchor_contact_world(self) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        bracket = self._yaw_anchor_bracket()
        if bracket is None:
            return None
        left, right, alpha = bracket
        pos0 = np.asarray(left[1:4], dtype=np.float64)
        pos1 = np.asarray(right[1:4], dtype=np.float64)
        # Anchors are recorded at the calibration scene translation; shift them with
        # the current scene-root XY randomization so yaw correction does not cancel XY.
        pos = (1.0 - alpha) * pos0 + alpha * pos1 + self._scene_root_translation_delta_world()

        q0 = np.asarray(left[4:8], dtype=np.float64)
        q1 = np.asarray(right[4:8], dtype=np.float64)
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        q = (1.0 - alpha) * q0 + alpha * q1
        q /= max(float(np.linalg.norm(q)), 1e-9)
        return pos, tuple(float(v) for v in q)

    def _yaw_anchor_position_correction(self) -> np.ndarray:
        anchor = self._yaw_anchor_contact_world()
        cfg = self.sampling_config
        if anchor is None or cfg.reference_contact_world is None:
            return np.zeros(3, dtype=np.float64)
        anchor_pos, _ = anchor
        geom_ref = self._geometric_map_calibrated_world_position(
            np.asarray(cfg.reference_contact_world, dtype=np.float64)
        )
        return np.asarray(anchor_pos, dtype=np.float64) - geom_ref

    def _map_calibrated_world_position(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64)
        if self.task_config.task_id not in (CLOSE_LAPTOP_TASK_ID, CLOSE_MICROWAVE_TASK_ID, ADJUST_FAUCET_TASK_ID):
            return pos
        mapped = self._geometric_map_calibrated_world_position(pos)
        if self.task_config.task_id == ADJUST_FAUCET_TASK_ID:
            # Faucet: yaw↔joint coupling replaces per-yaw yaml anchors.
            return mapped
        return mapped + self._yaw_anchor_position_correction()

    def _joint_fitted_yaml_handle_world(self, default_contact_w: np.ndarray) -> np.ndarray:
        """Optional hinge-arc handle model from calibrated EE touch samples.

        The legacy yaml handle path is preserved at the reference angle (15°).
        Within push_contact_joint_fit_range_deg (5°..40° for close_laptop_lid),
        calibrated handle points define a circular arc in the laptop X/Z plane.
        This keeps the 15° successful motion unchanged while allowing
        task_registry joint_1.position to move inside the calibrated range.

        World arc samples are mapped by scene-root Δtranslation + Δyaw so moving or
        yaw-rotating the laptop retargets the arm.
        """
        cfg = self.sampling_config
        world_delta = self._scene_root_translation_delta_world()
        yaw_delta = self._scene_root_yaw_delta_deg()
        arc_points = tuple(getattr(cfg, "push_contact_joint_arc_points", ()) or ())
        if len(arc_points) >= 3:
            samples = sorted(
                (
                    float(p[0]),
                    self._map_calibrated_world_position(np.asarray(p[1:4], dtype=np.float64)),
                )
                for p in arc_points
            )
        elif (
            cfg.reference_contact_world is not None
            and cfg.push_contact_joint_fit_deg is not None
            and cfg.push_contact_joint_fit_world is not None
        ):
            samples = sorted(
                (
                    (
                        float(cfg.push_contact_reference_joint_deg),
                        self._map_calibrated_world_position(
                            np.asarray(cfg.reference_contact_world, dtype=np.float64)
                        ),
                    ),
                    (
                        float(cfg.push_contact_joint_fit_deg),
                        self._map_calibrated_world_position(
                            np.asarray(cfg.push_contact_joint_fit_world, dtype=np.float64)
                        ),
                    ),
                )
            )
        else:
            return default_contact_w

        if float(np.linalg.norm(world_delta)) > 1e-6 or abs(yaw_delta) > 1e-6:
            if not self._scene_root_map_logged:
                anchor_delta = self._yaw_anchor_position_correction()
                self._log_debug(
                    f"[INFO] yaml_handle scene-root Δ applied to arc samples: "
                    f"translation_m={np.round(world_delta, 4).tolist()} "
                    f"yaw_deg={yaw_delta:.2f} "
                    f"anchor_correction_m={np.round(anchor_delta, 4).tolist()}"
                )
                self._scene_root_map_logged = True

        joint_read_deg = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        if joint_read_deg is None:
            return default_contact_w
        joint_deg = self._yaw_joint_equivalent_arc_deg(float(joint_read_deg))

        ref_deg = float(cfg.push_contact_reference_joint_deg)
        if abs(joint_deg - ref_deg) < 0.5:
            if cfg.reference_contact_world is not None:
                return self._map_calibrated_world_position(
                    np.asarray(cfg.reference_contact_world, dtype=np.float64)
                )
            return default_contact_w
        fit_range = cfg.push_contact_joint_fit_range_deg
        if fit_range is not None:
            lo, hi = sorted((float(fit_range[0]), float(fit_range[1])))
            boundary_tol_deg = 0.5
            if joint_deg < lo - boundary_tol_deg or joint_deg > hi + boundary_tol_deg:
                print(
                    f"[WARN] yaml_handle arc-fit disabled: joint={joint_deg:.2f}° "
                    f"outside calibrated arc range [{lo:.1f}, {hi:.1f}]°; using link-local default.",
                    flush=True,
                )
                return default_contact_w
            joint_deg = float(np.clip(joint_deg, lo, hi))

        x_dir, y_neg_dir = self._generated_root_yaw_frame()
        y_pos_dir = -y_neg_dir

        sample_degs = np.asarray([deg for deg, _ in samples], dtype=np.float64)
        sample_world = np.asarray([pos for _, pos in samples], dtype=np.float64)
        hinge_axis = np.asarray(cfg.hinge.axis, dtype=np.float64)
        horizontal_arc = abs(float(hinge_axis[2])) >= max(abs(float(hinge_axis[0])), abs(float(hinge_axis[1])))
        if horizontal_arc:
            sample_plane = np.column_stack((sample_world @ x_dir, sample_world @ y_pos_dir))
        else:
            sample_plane = np.column_stack((sample_world @ x_dir, sample_world[:, 2]))

        if len(samples) >= 3:
            design = np.column_stack(
                (2.0 * sample_plane[:, 0], 2.0 * sample_plane[:, 1], np.ones(len(sample_plane)))
            )
            rhs = np.sum(sample_plane * sample_plane, axis=1)
            center_u, center_v, radius_term = np.linalg.lstsq(design, rhs, rcond=None)[0]
            radius = math.sqrt(max(0.0, radius_term + center_u * center_u + center_v * center_v))
            if radius < 1e-6:
                nearest_i = int(np.argmin(np.abs(sample_degs - joint_deg)))
                return np.asarray(sample_world[nearest_i], dtype=np.float64)
            sample_angles = np.unwrap(
                np.arctan2(sample_plane[:, 1] - center_v, sample_plane[:, 0] - center_u)
            )

            def interp_extrap(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
                if x <= xs[0]:
                    slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-9)
                    return float(ys[0] + (x - xs[0]) * slope)
                if x >= xs[-1]:
                    slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)
                    return float(ys[-1] + (x - xs[-1]) * slope)
                return float(np.interp(x, xs, ys))

            theta = interp_extrap(joint_deg, sample_degs, sample_angles)
            contact_plane = np.array(
                [center_u + radius * math.cos(theta), center_v + radius * math.sin(theta)],
                dtype=np.float64,
            )
        else:
            ref_w = sample_world[0]
            fit_w = sample_world[-1]
            fit_deg = float(sample_degs[-1])
            fit_delta_rad = math.radians(fit_deg - ref_deg)
            if abs(fit_delta_rad) < 1e-6:
                return default_contact_w

            ref_plane = sample_plane[0]
            fit_plane = sample_plane[-1]

            def rot2(theta: float) -> np.ndarray:
                c = math.cos(theta)
                s = math.sin(theta)
                return np.array([[c, s], [-s, c]], dtype=np.float64)

            fit_rot = rot2(fit_delta_rad)
            try:
                origin_plane = np.linalg.solve(np.eye(2) - fit_rot, fit_plane - fit_rot @ ref_plane)
            except np.linalg.LinAlgError:
                return default_contact_w

            theta = math.radians(joint_deg - ref_deg)
            contact_plane = origin_plane + rot2(theta) @ (ref_plane - origin_plane)

        if horizontal_arc:
            local_z = float(np.interp(joint_deg, sample_degs, sample_world[:, 2]))
            fitted = x_dir * contact_plane[0] + y_pos_dir * contact_plane[1]
            fitted[2] = local_z
        else:
            local_y = float(np.interp(joint_deg, sample_degs, sample_world @ y_pos_dir))
            fitted = x_dir * contact_plane[0] + y_pos_dir * local_y
            fitted[2] = contact_plane[1]

        if self.verbose:
            log_joint = round(float(joint_read_deg), 1)
            if (
                self._last_arc_fit_log_joint is None
                or abs(log_joint - self._last_arc_fit_log_joint) >= 1.0
            ):
                self._last_arc_fit_log_joint = log_joint
                self._log_debug(
                    f"[INFO] yaml_handle arc-fit: joint={float(joint_read_deg):.2f}° "
                    f"used={joint_deg:.2f}° "
                    f"samples={[round(float(v), 2) for v in sample_degs.tolist()]} "
                    f"range={[round(float(v), 3) for v in fit_range] if fit_range is not None else 'unbounded'} "
                    f"default={np.round(default_contact_w, 4).tolist()} "
                    f"fitted={np.round(fitted, 4).tolist()}"
                )
        return fitted

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
                approach = scene_approach_direction(ee_pos, link_pos_np, cfg)
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
        contact_w = self._joint_fitted_yaml_handle_world(contact_w)
        anchor_contact = self._yaw_anchor_contact_world()
        if anchor_contact is not None:
            _, quat_w = anchor_contact

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

    @staticmethod
    def _unit_or_none(vec: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return np.asarray(vec, dtype=np.float64) / norm

    def _contact_push_normal_world(
        self,
        contact_w: np.ndarray,
        approach_w: np.ndarray,
    ) -> np.ndarray | None:
        """Desired push normal: approach point -> contact point."""
        return self._unit_or_none(np.asarray(contact_w, dtype=np.float64) - np.asarray(approach_w, dtype=np.float64))

    def _link_plane_push_normal_world(
        self,
        contact_w: np.ndarray,
        hinge_origin_w: np.ndarray,
        hinge_axis_w: np.ndarray,
        *,
        fallback_quat_w: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray | None:
        """Normal to the door plane at contact: tangent direction of the hinge arc."""
        axis = self._unit_or_none(np.asarray(hinge_axis_w, dtype=np.float64))
        if axis is None:
            return None
        radial = np.asarray(contact_w, dtype=np.float64) - np.asarray(hinge_origin_w, dtype=np.float64)
        radial = radial - np.dot(radial, axis) * axis
        radial = self._unit_or_none(radial)
        if radial is None:
            return None

        normal = self._unit_or_none(np.cross(axis, radial))
        if normal is None:
            return None

        expected = self._expected_close_delta_dir_world()
        if expected is not None:
            if float(np.dot(normal, expected)) < 0.0:
                normal = -normal
        elif fallback_quat_w is not None:
            fallback_axis = self._gripper_push_axis_world(fallback_quat_w)
            if float(np.dot(normal, fallback_axis)) < 0.0:
                normal = -normal
        return normal

    def _quat_from_world_push_normal(
        self,
        normal_w: np.ndarray,
        *,
        horizontal: bool,
    ) -> tuple[float, float, float, float]:
        rot = align_gripper_z_with_normal(normal_w, horizontal=horizontal)
        return _rot_to_wxyz(rot)

    @staticmethod
    def _axis_alignment_deg(axis_w: np.ndarray, normal_w: np.ndarray) -> float:
        dot = float(np.clip(np.dot(axis_w, normal_w), -1.0, 1.0))
        return math.degrees(math.acos(dot))

    def _normal_alignment_report(
        self,
        contact_w: np.ndarray,
        approach_w: np.ndarray,
        planned_quat_w: tuple[float, float, float, float],
        actual_quat_w: tuple[float, float, float, float],
    ) -> dict[str, object] | None:
        normal_w = self._contact_push_normal_world(contact_w, approach_w)
        if normal_w is None:
            return None
        planned_axis_w = self._gripper_push_axis_world(planned_quat_w)
        actual_axis_w = self._gripper_push_axis_world(actual_quat_w)
        return {
            "push_normal_w": normal_w,
            "planned_push_axis_w": planned_axis_w,
            "actual_push_axis_w": actual_axis_w,
            "planned_normal_dot": float(np.dot(planned_axis_w, normal_w)),
            "actual_normal_dot": float(np.dot(actual_axis_w, normal_w)),
            "planned_normal_angle_deg": self._axis_alignment_deg(planned_axis_w, normal_w),
            "actual_normal_angle_deg": self._axis_alignment_deg(actual_axis_w, normal_w),
        }

    def _log_normal_alignment(
        self,
        label: str,
        contact_w: np.ndarray,
        approach_w: np.ndarray,
        planned_quat_w: tuple[float, float, float, float],
        actual_quat_w: tuple[float, float, float, float],
    ) -> None:
        report = self._normal_alignment_report(contact_w, approach_w, planned_quat_w, actual_quat_w)
        if report is None:
            return
        self._log_debug(
            f"[INFO] {label} normal alignment: "
            f"normal={np.round(report['push_normal_w'], 4).tolist()} "
            f"planned+Z={np.round(report['planned_push_axis_w'], 4).tolist()} "
            f"actual+Z={np.round(report['actual_push_axis_w'], 4).tolist()} "
            f"dot(planned,normal)={float(report['planned_normal_dot']):.4f} "
            f"angle={float(report['planned_normal_angle_deg']):.1f}°; "
            f"dot(actual,normal)={float(report['actual_normal_dot']):.4f} "
            f"angle={float(report['actual_normal_angle_deg']):.1f}° "
            "(0° means EE +Z follows the push normal; 90° means perpendicular to it)"
        )

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

    def _generated_root_yaw_frame(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            _, root_quat = self.env_module.get_prim_world_pose_wxyz("/World/generated")
            rot = _wxyz_to_rot(root_quat)
        except Exception:
            rot = R.identity()

        x_dir = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        y_neg_dir = rot.apply(np.array([0.0, -1.0, 0.0], dtype=np.float64))
        x_dir[2] = 0.0
        y_neg_dir[2] = 0.0
        if float(np.linalg.norm(x_dir)) < 1e-9:
            x_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if float(np.linalg.norm(y_neg_dir)) < 1e-9:
            y_neg_dir = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        x_dir /= max(float(np.linalg.norm(x_dir)), 1e-9)
        y_neg_dir /= max(float(np.linalg.norm(y_neg_dir)), 1e-9)
        return x_dir, y_neg_dir

    def _explicit_laptop_hinge_world_frame(
        self, contact_pos_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Laptop-specific hinge prior from URDF geometry and calibrated handle lever."""
        contact = np.asarray(contact_pos_world, dtype=np.float64)
        radial_x, axis_w = self._generated_root_yaw_frame()
        lever_m = float(self._close_hinge_lever_m or 0.1988)
        x_offset_m = min(
            abs(float(getattr(self.sampling_config, "push_anchor_dist_m", 0.10))),
            max(0.0, lever_m - 1e-4),
        )
        z_offset_m = math.sqrt(max(0.0, lever_m * lever_m - x_offset_m * x_offset_m))
        origin = contact - radial_x * x_offset_m - np.array([0.0, 0.0, z_offset_m], dtype=np.float64)
        return origin, axis_w

    def _horizontal_hinge_world_frame_from_expected_delta(
        self, contact_pos_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal-door prior: hinge axis is vertical, and the planned chord matches demo ΔEE."""
        contact = np.asarray(contact_pos_world, dtype=np.float64)
        expected = self._expected_close_delta_dir_world()
        if expected is None:
            expected = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        expected[2] = 0.0
        expected /= max(float(np.linalg.norm(expected)), 1e-9)

        axis_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        radius = float(
            getattr(self.push_cfg, "close_hinge_radius_m", None)
            or self._close_hinge_lever_m
            or getattr(self.sampling_config, "push_anchor_dist_m", 0.10)
        )
        radius = max(radius, 1e-4)

        joint_init = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        if joint_init is None:
            joint_init = 0.0
        joint_target = self._resolve_close_target_deg(float(joint_init))
        span_rad = math.radians(float(joint_target) - float(joint_init))
        # The full demo tells us the end-to-end chord direction.  For a signed
        # rotation span, the chord direction is the mid-arc tangent, so place
        # the radius vector half a span behind that tangent.
        chord_angle = math.atan2(float(expected[1]), float(expected[0]))
        rel_angle = chord_angle - math.copysign(math.pi / 2.0, span_rad or 1.0) - span_rad / 2.0
        rel = radius * np.array([math.cos(rel_angle), math.sin(rel_angle), 0.0], dtype=np.float64)
        origin = contact - rel
        return origin, axis_w

    def _get_planning_hinge_world_frame(
        self, contact_pos_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Close planning hinge frame: explicit laptop geometry from contact + lever."""
        if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS:
            return self._horizontal_hinge_world_frame_from_expected_delta(contact_pos_world)
        return self._explicit_laptop_hinge_world_frame(contact_pos_world)

    def _expected_close_delta_dir_world(self) -> np.ndarray | None:
        expected = getattr(self.push_cfg, "close_expected_delta_dir_world", None)
        if expected is None:
            return None
        vec = np.asarray(expected, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return vec / norm

    def _close_axis_alignment_score(
        self,
        *,
        eef_pos_world: np.ndarray,
        eef_quat_wxyz: tuple[float, float, float, float],
        link_pos_world: np.ndarray,
        link_quat_wxyz: tuple[float, float, float, float],
        hinge_origin_world: np.ndarray,
        hinge_axis_world: np.ndarray,
        theta_init_rad: float,
        theta_target_rad: float,
    ) -> tuple[float, np.ndarray]:
        expected = self._expected_close_delta_dir_world()
        if expected is None:
            return 0.0, np.zeros(3, dtype=np.float64)
        poses = compute_articulation_ee_trajectory(
            eef_pos_world=eef_pos_world,
            eef_quat_wxyz=eef_quat_wxyz,
            link_pos_world=link_pos_world,
            link_quat_wxyz=link_quat_wxyz,
            hinge_origin_world=hinge_origin_world,
            hinge_axis_world=hinge_axis_world,
            theta_init_rad=theta_init_rad,
            theta_targets_rad=(theta_target_rad,),
        )
        if not poses:
            return 0.0, np.zeros(3, dtype=np.float64)
        delta = np.asarray(poses[-1][0], dtype=np.float64) - np.asarray(eef_pos_world, dtype=np.float64)
        return float(np.dot(delta, expected)), delta

    def _maybe_flip_close_hinge_axis(
        self,
        *,
        eef_pos_world: np.ndarray,
        eef_quat_wxyz: tuple[float, float, float, float],
        link_pos_world: np.ndarray,
        link_quat_wxyz: tuple[float, float, float, float],
        hinge_origin_world: np.ndarray,
        hinge_axis_world: np.ndarray,
        joint_init_deg: float,
        joint_target_deg: float,
    ) -> np.ndarray:
        axis = np.asarray(hinge_axis_world, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        if not bool(getattr(self.push_cfg, "close_auto_flip_hinge_axis", False)):
            return axis
        expected = self._expected_close_delta_dir_world()
        if expected is None:
            return axis

        kwargs = {
            "eef_pos_world": np.asarray(eef_pos_world, dtype=np.float64),
            "eef_quat_wxyz": eef_quat_wxyz,
            "link_pos_world": np.asarray(link_pos_world, dtype=np.float64),
            "link_quat_wxyz": link_quat_wxyz,
            "hinge_origin_world": np.asarray(hinge_origin_world, dtype=np.float64),
            "theta_init_rad": math.radians(float(joint_init_deg)),
            "theta_target_rad": math.radians(float(joint_target_deg)),
        }
        score, delta = self._close_axis_alignment_score(hinge_axis_world=axis, **kwargs)
        flip_score, flip_delta = self._close_axis_alignment_score(hinge_axis_world=-axis, **kwargs)
        if flip_score > score:
            self._log_info(
                f"[INFO] Close hinge axis flipped to align with expected Δdir "
                f"{np.round(expected, 4).tolist()}: "
                f"score {score:.4f}->{flip_score:.4f}, "
                f"ΔEE {np.round(delta, 4).tolist()}->{np.round(flip_delta, 4).tolist()}"
            )
            return -axis
        self._log_info(
            f"[INFO] Close hinge axis kept for expected Δdir {np.round(expected, 4).tolist()}: "
            f"score={score:.4f}, ΔEE={np.round(delta, 4).tolist()}"
        )
        return axis

    def _reset_close_phase_tracking(self) -> None:
        """Re-sync incremental IK tracking before hinge-relative close."""
        self._close_anchor = None
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

    def _close_target_is_less_than(self) -> bool:
        return any(
            spec.joint_prim == self.task_config.joint_prim and spec.angle_lt_deg is not None
            for spec in self.success_specs
        )

    def _close_target_is_less_than(self) -> bool:
        return any(
            spec.joint_prim == self.task_config.joint_prim and spec.angle_lt_deg is not None
            for spec in self.success_specs
        )

    def _close_target_is_less_than(self) -> bool:
        return any(
            spec.joint_prim == self.task_config.joint_prim and spec.angle_lt_deg is not None
            for spec in self.success_specs
        )

    def _resolve_close_target_deg(self, joint_deg: float) -> float:
        """ArticuBot T_rel target: close_ratio, bumped to rollout success threshold if needed."""
        arc_points = tuple(getattr(self.sampling_config, "push_contact_joint_arc_points", ()) or ())
        if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS and arc_points:
            # Calibrated XY arc tasks use YAML handle samples directly; success still
            # stops early from live joint readback, so do not plan from registry over-push limits.
            sample_degs = [float(p[0]) for p in arc_points]
            if self._close_target_is_less_than():
                arc_target = getattr(
                    self.sampling_config, "push_contact_joint_arc_target_deg", None
                )
                if arc_target is not None:
                    return float(arc_target)
                return min(sample_degs)
            return max(sample_degs)
        ratio_target = joint_deg + self.push_cfg.close_ratio * (
            self.joint_upper_limit_deg - joint_deg
        )
        success_floor = self.joint_upper_limit_deg
        for spec in self.success_specs:
            if spec.joint_prim != self.task_config.joint_prim:
                continue
            if spec.angle_gt_deg is not None:
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

    def _init_close_anchor(
        self,
        eef_pos_world: np.ndarray,
        eef_quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        """Capture ArticuBot T_rel at contact: constant eef_in_link while lid rotates."""
        link_pos_np, link_quat = self._read_link_pose()
        hinge_origin_w, hinge_axis_w = self._get_planning_hinge_world_frame(
            np.asarray(eef_pos_world, dtype=np.float64)
        )
        link_pos_inv, link_quat_inv = invert_pose(link_pos_np, link_quat)
        eef_in_link_pos, eef_in_link_quat = compose_pose(
            link_pos_inv,
            link_quat_inv,
            np.asarray(eef_pos_world, dtype=np.float64),
            eef_quat_wxyz,
        )
        joint_init = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        if joint_init is None:
            joint_init = 15.0
        target_joint = self._resolve_close_target_deg(float(joint_init))
        hinge_axis_w = self._maybe_flip_close_hinge_axis(
            eef_pos_world=np.asarray(eef_pos_world, dtype=np.float64),
            eef_quat_wxyz=eef_quat_wxyz,
            link_pos_world=link_pos_np,
            link_quat_wxyz=link_quat,
            hinge_origin_world=hinge_origin_w,
            hinge_axis_world=hinge_axis_w,
            joint_init_deg=float(joint_init),
            joint_target_deg=float(target_joint),
        )
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
        self._log_info(
            f"[INFO] Close anchor T_rel (USD joint_1: 15°≈real90°open, 104°≈real0°closed): "
            f"{joint_init:.2f}° -> {target_joint:.2f}° ({n_wp} waypoints) "
            f"eef_in_link={np.round(eef_in_link_pos, 4).tolist()}"
        )
        self._log_info(
            f"[INFO] Close hinge frame: origin={np.round(hinge_origin_w, 4).tolist()} "
            f"axis={np.round(hinge_axis_w, 4).tolist()}"
        )
        diag0 = self._close_handle_diagnostics(joint_init, joint_deg=float(joint_init))
        if diag0:
            self._log_info(
                f"[INFO] Close anchor handle diag @joint={joint_init:.2f}°USD: "
                f"contact={np.round(diag0['handle_contact'], 4).tolist()} "
                f"handle_usd={np.round(diag0['handle_usd'], 4).tolist()} "
                f"handle_trl_joint={np.round(diag0['handle_trl_joint'], 4).tolist()} "
                f"trl_joint↔handle_usd={diag0['trl_joint_handle_usd_m']:.4f}m "
                f"handle_usd_physx_gap={diag0['handle_usd_physx_gap_m']:.4f}m"
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
        poses = self._build_calibrated_arc_close_trajectory(
            contact_pos,
            anchor["contact_quat_w"],
            joint_init,
            joint_target,
            n_wp,
        )
        if poses:
            ee_end = np.asarray(poses[-1][0], dtype=np.float64)
            delta = ee_end - contact_pos
            arc_m = float(sum(
                np.linalg.norm(np.asarray(poses[i][0]) - np.asarray(poses[i - 1][0]))
                for i in range(1, len(poses))
            ))
            self._log_info(
                f"[INFO] Calibrated-arc close plan: joint {joint_init:.2f}° -> {joint_target:.2f}° "
                f"({len(poses)} waypoints @ {getattr(self.push_cfg, 'close_step_deg_usd', 1.0):.1f}°USD, "
                f"EE arc {arc_m:.3f}m, ΔEE={np.round(delta, 4).tolist()} "
                f"|Δ|={float(np.linalg.norm(delta)):.3f}m)"
            )
            return poses

        if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS:
            self._log_info(
                "[WARN] Calibrated XY arc close unavailable; aborting close instead of falling back to T_rel."
            )
            return []

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
            delta = ee_end - contact_pos
            arc_m = float(sum(
                np.linalg.norm(np.asarray(poses[i][0]) - np.asarray(poses[i - 1][0]))
                for i in range(1, len(poses))
            ))
            self._log_info(
                f"[INFO] T_rel close plan: joint {joint_init:.2f}° -> {joint_target:.2f}° "
                f"({len(poses)} waypoints @ {getattr(self.push_cfg, 'close_step_deg_usd', 1.0):.1f}°USD, "
                f"EE arc {arc_m:.3f}m, ΔEE={np.round(delta, 4).tolist()} "
                f"|Δ|={float(np.linalg.norm(delta)):.3f}m)"
            )
        return poses

    def _build_calibrated_arc_close_trajectory(
        self,
        contact_pos: np.ndarray,
        contact_quat_w: tuple[float, float, float, float],
        joint_init: float,
        joint_target: float,
        n_wp: int,
    ) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        """Close path from calibrated [joint, x, y, z] handle arc points."""
        if self.task_config.task_id not in CALIBRATED_XY_ARC_TASK_IDS:
            return []
        arc_points = tuple(getattr(self.sampling_config, "push_contact_joint_arc_points", ()) or ())
        if len(arc_points) < 3:
            return []
        init_eval = self._calibrated_arc_position_at_joint(joint_init)
        if init_eval is None:
            sample_degs = np.asarray([float(p[0]) for p in arc_points], dtype=np.float64)
            self._log_info(
                f"[WARN] calibrated close arc skipped: joint_init={joint_init:.2f}° "
                f"outside samples [{sample_degs[0]:.1f}, {sample_degs[-1]:.1f}]°"
            )
            return []
        _, info = init_eval
        sample_degs = np.asarray(info["sample_degs"], dtype=np.float64)
        target = float(joint_target)
        joint_targets = np.linspace(float(joint_init), target, max(2, int(n_wp)))

        poses: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
        for joint in joint_targets:
            evaluated = self._calibrated_arc_position_at_joint(float(joint))
            if evaluated is None:
                continue
            pos, _ = evaluated
            poses.append((pos, contact_quat_w))

        z_source = (
            "contact_anchor"
            if self._close_anchor is not None and self._close_target_is_less_than()
            else "yaml"
        )
        self._log_info(
            f"[INFO] Calibrated close arc fit: samples={[round(float(v), 2) for v in sample_degs.tolist()]} "
            f"target={target:.2f}° "
            f"center_xy={[round(float(info['center_u']), 4), round(float(info['center_v']), 4)]} "
            f"radius={float(info['radius']):.4f} "
            f"z_source={z_source}"
        )
        return poses

    def _compute_close_pose_at_joint_deg(
        self, theta_deg: float
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Single T_rel EE pose for θ (USD), fixed eef_in_link from contact anchor."""
        if self._close_anchor is None:
            raise RuntimeError("Call _init_close_anchor before close phase.")
        anchor = self._close_anchor
        if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS:
            evaluated = self._calibrated_arc_position_at_joint(float(theta_deg))
            if evaluated is not None:
                pos, _ = evaluated
                return pos, anchor["contact_quat_w"]
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

    def _calibrated_arc_position_at_joint(
        self,
        joint_deg: float,
    ) -> tuple[np.ndarray, dict[str, object]] | None:
        """Evaluate the YAML [joint, x, y, z] handle circle at a joint angle."""
        cfg = self.sampling_config
        arc_points = tuple(getattr(cfg, "push_contact_joint_arc_points", ()) or ())
        if len(arc_points) < 3:
            return None

        samples = sorted(
            (
                float(p[0]),
                self._map_calibrated_world_position(np.asarray(p[1:4], dtype=np.float64)),
            )
            for p in arc_points
        )
        sample_degs = np.asarray([deg for deg, _ in samples], dtype=np.float64)
        sample_world = np.asarray([pos for _, pos in samples], dtype=np.float64)
        arc_target = getattr(cfg, "push_contact_joint_arc_target_deg", None)
        eval_min = float(sample_degs[0])
        if arc_target is not None and self._close_target_is_less_than():
            eval_min = min(eval_min, float(arc_target))
        boundary_tol_deg = 0.75
        arc_joint = self._yaw_joint_equivalent_arc_deg(float(joint_deg))
        if (
            arc_joint < eval_min - boundary_tol_deg
            or arc_joint > float(sample_degs[-1]) + boundary_tol_deg
        ):
            return None
        used_joint = arc_joint

        x_dir, y_neg_dir = self._generated_root_yaw_frame()
        y_pos_dir = -y_neg_dir
        sample_plane = np.column_stack((sample_world @ x_dir, sample_world @ y_pos_dir))
        design = np.column_stack(
            (2.0 * sample_plane[:, 0], 2.0 * sample_plane[:, 1], np.ones(len(sample_plane)))
        )
        rhs = np.sum(sample_plane * sample_plane, axis=1)
        center_u, center_v, radius_term = np.linalg.lstsq(design, rhs, rcond=None)[0]
        radius = math.sqrt(max(0.0, radius_term + center_u * center_u + center_v * center_v))
        if radius < 1e-6:
            return None

        sample_angles = np.unwrap(
            np.arctan2(sample_plane[:, 1] - center_v, sample_plane[:, 0] - center_u)
        )
        theta = _interp_extrap_scalar(used_joint, sample_degs, sample_angles)
        plane = np.array(
            [center_u + radius * math.cos(theta), center_v + radius * math.sin(theta)],
            dtype=np.float64,
        )
        pos = x_dir * plane[0] + y_pos_dir * plane[1]
        if self._close_anchor is not None and self._close_target_is_less_than():
            # Push phase: XY arc only; keep EE Z fixed at first contact.
            pos[2] = float(np.asarray(self._close_anchor["contact_pos_w"], dtype=np.float64)[2])
        else:
            pos[2] = float(np.interp(used_joint, sample_degs, sample_world[:, 2]))
        info: dict[str, object] = {
            "sample_degs": sample_degs.copy(),
            "used_joint": used_joint,
            "center_u": float(center_u),
            "center_v": float(center_v),
            "radius": float(radius),
        }
        return pos, info

    def _align_close_trajectory_quats_to_tangent(
        self,
        poses: list[tuple[np.ndarray, tuple[float, float, float, float]]],
    ) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        """Microwave close: keep positions, gradually rotate EE +Z toward each arc tangent."""
        if self.task_config.task_id != CLOSE_MICROWAVE_TASK_ID or self._close_anchor is None:
            return poses
        if not poses:
            return poses

        anchor = self._close_anchor
        hinge_origin_w = np.asarray(anchor["hinge_origin_w"], dtype=np.float64)
        hinge_axis_w = np.asarray(anchor["hinge_axis_w"], dtype=np.float64)
        aligned: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
        tangent_angles: list[float] = []
        ramp_start = 0.35
        ramp_span = max(1e-6, 1.0 - ramp_start)
        alphas: list[float] = []
        for i, (pos, quat) in enumerate(poses):
            pos_np = np.asarray(pos, dtype=np.float64)
            tangent_w = self._link_plane_push_normal_world(
                pos_np,
                hinge_origin_w,
                hinge_axis_w,
                fallback_quat_w=quat,
            )
            if tangent_w is None:
                aligned.append((pos_np, quat))
                continue
            tangent_quat = self._quat_from_world_push_normal(
                tangent_w,
                horizontal=self.sampling_config.horizontal_grasp,
            )
            progress = float(i) / float(max(1, len(poses) - 1))
            alpha = float(np.clip((progress - ramp_start) / ramp_span, 0.0, 1.0))
            blended_quat = _slerp_wxyz(quat, tangent_quat, alpha)
            alphas.append(alpha)
            tangent_angles.append(self._axis_alignment_deg(self._gripper_push_axis_world(quat), tangent_w))
            aligned.append((pos_np, blended_quat))

        if tangent_angles:
            first_axis = self._gripper_push_axis_world(aligned[0][1])
            last_axis = self._gripper_push_axis_world(aligned[-1][1])
            self._log_info(
                "[INFO] Microwave close quats tangent-blended: "
                f"alpha_first={alphas[0]:.2f} "
                f"alpha_last={alphas[-1]:.2f} "
                f"old_angle_first={tangent_angles[0]:.1f}° "
                f"old_angle_last={tangent_angles[-1]:.1f}° "
                f"new+Z_first={np.round(first_axis, 4).tolist()} "
                f"new+Z_last={np.round(last_axis, 4).tolist()}"
            )
        return aligned

    def _close_contact_anchor_world_at_joint(self, joint_deg: float) -> np.ndarray | None:
        """Surface point captured at contact, rotated by the live scene hinge angle."""
        if self._close_anchor is None:
            return None
        if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS:
            evaluated = self._calibrated_arc_position_at_joint(float(joint_deg))
            if evaluated is not None:
                pos, _ = evaluated
                return pos
        anchor = self._close_anchor
        origin = np.asarray(anchor["hinge_origin_w"], dtype=np.float64)
        axis = np.asarray(anchor["hinge_axis_w"], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return None
        axis = axis / norm
        theta = math.radians(float(joint_deg) - float(anchor["joint_init_deg"]))
        rel = np.asarray(anchor["contact_pos_w"], dtype=np.float64) - origin
        return origin + R.from_rotvec(axis * theta).apply(rel)

    def _apply_close_contact_anchor_correction(
        self,
        nominal_target_pos: np.ndarray,
    ) -> np.ndarray:
        """Small external correction that pulls the close target back toward the live contact anchor."""
        if self._close_anchor is None:
            return np.asarray(nominal_target_pos, dtype=np.float64)

        joint_deg = self.env_module.read_scene_joint_angle_deg(self.task_config.joint_prim)
        if joint_deg is None:
            return np.asarray(nominal_target_pos, dtype=np.float64)

        anchor_now = self._close_contact_anchor_world_at_joint(float(joint_deg))
        if anchor_now is None:
            return np.asarray(nominal_target_pos, dtype=np.float64)

        ee_pos, _ = self._read_ee_pose()
        error = anchor_now - ee_pos
        error_norm = float(np.linalg.norm(error))
        deadband_m = (
            CLOSE_CONTACT_ANCHOR_DEADBAND_CALIBRATED_M
            if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS
            else CLOSE_CONTACT_ANCHOR_DEADBAND_M
        )
        if error_norm <= deadband_m:
            return np.asarray(nominal_target_pos, dtype=np.float64)

        correction_gain = (
            CLOSE_CONTACT_ANCHOR_GAIN_CALIBRATED
            if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS
            else CLOSE_CONTACT_ANCHOR_GAIN
        )
        correction_limit = (
            CLOSE_CONTACT_ANCHOR_MAX_CORRECTION_CALIBRATED_M
            if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS
            else CLOSE_CONTACT_ANCHOR_MAX_CORRECTION_M
        )
        correction = correction_gain * error
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > correction_limit:
            correction *= correction_limit / max(correction_norm, 1e-9)

        corrected = np.asarray(nominal_target_pos, dtype=np.float64) + correction
        if self.trace_ee_handle and self.control_step_count % self.trace_ee_handle_interval == 0:
            self._log_info(
                f"[INFO] close anchor correction: live_anchor={np.round(anchor_now, 4).tolist()} "
                f"ee={np.round(ee_pos, 4).tolist()} "
                f"err={error_norm:.4f}m z_err={float(error[2]):+.4f}m "
                f"correction={np.round(correction, 4).tolist()}"
            )
        return corrected

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
            corrected_target = self._apply_close_contact_anchor_correction(target_pos)
            self._advance_tracking_pose(corrected_target, target_quat, max_pos_step_m=pos_step)
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

    def _execute_sampled_pose_path_close(
        self,
        collector: OfficialEpisodeCollector,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        trajectory: list[tuple[np.ndarray, tuple[float, float, float, float]]],
    ) -> bool:
        """Execute a coarse sampled T_rel arc using the same pose-path servo as handle reach."""
        if len(trajectory) < 2:
            return False
        continuous_result = self._execute_continuous_calibrated_close(
            collector,
            on_control_step,
            trajectory,
        )
        if continuous_result is not None:
            return continuous_result

        close_poses = list(trajectory[1:])
        sample_count = max(2, int(getattr(self.push_cfg, "close_sampled_waypoints", 12)))
        if len(close_poses) > sample_count:
            idx = np.linspace(0, len(close_poses) - 1, sample_count, dtype=int)
            close_poses = [close_poses[int(i)] for i in idx]

        steps_per_wp = int(
            getattr(self.push_cfg, "close_steps_per_waypoint", None)
            or getattr(self.push_cfg, "contact_hold_steps", 24)
        )
        steps_per_wp = max(8, steps_per_wp)
        close_clamp = bool(getattr(self.push_cfg, "close_clamp_joints", True))

        wp0 = np.asarray(trajectory[0][0], dtype=np.float64)
        wp_last = np.asarray(trajectory[-1][0], dtype=np.float64)
        sampled_last = np.asarray(close_poses[-1][0], dtype=np.float64)
        self._log_info(
            f"[INFO] Close phase (sampled pose path): sampled={len(close_poses)} "
            f"from {len(trajectory) - 1} close waypoints, steps/wp={steps_per_wp}, "
            f"close_clamp_joints={close_clamp}, full_arc_delta={float(np.linalg.norm(wp_last - wp0)):.3f}m, "
            f"sampled_delta={float(np.linalg.norm(sampled_last - wp0)):.3f}m"
        )
        self._log_info(
            "[INFO] Sampled close keyframes: "
            + " -> ".join(str(np.round(np.asarray(pos), 3).tolist()) for pos, _ in close_poses)
        )

        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        stopped = self._follow_pose_path(
            collector,
            close_poses,
            steps_per_wp,
            on_control_step,
            clamp_joints=close_clamp,
            close_anchor_correction=True,
        )
        if not stopped and close_poses:
            last_pos, last_quat = close_poses[-1]
            configured_hold_steps = getattr(self.push_cfg, "close_final_hold_steps", None)
            remaining_steps = self._remaining_episode_control_steps()
            hold_steps = (
                max(0, int(configured_hold_steps))
                if configured_hold_steps is not None
                else (0 if remaining_steps is None else remaining_steps)
            )
            close_step_m = float(getattr(self.push_cfg, "close_push_ee_step_m", 0.003))
            if hold_steps > 0:
                self._log_info(
                    f"[INFO] Close hold: up to {hold_steps} remaining control steps on "
                    f"{np.round(np.asarray(last_pos), 3).tolist()}"
                )
            hold_start = int(self.control_step_count)
            while int(self.control_step_count) - hold_start < hold_steps:
                remaining_hold_steps = hold_steps - (int(self.control_step_count) - hold_start)
                if self._servo_toward_close_pose(
                    collector,
                    np.asarray(last_pos, dtype=np.float64),
                    last_quat,
                    on_control_step,
                    substeps=min(2, max(1, remaining_hold_steps)),
                    max_pos_step_m=close_step_m,
                    clamp_joints=close_clamp,
                ):
                    stopped = True
                    break
                success_now, joint_degs = evaluate_rollout_success(
                    self.env_module, self.success_specs
                )
                if on_control_step is not None:
                    if on_control_step(success_now, joint_degs):
                        stopped = True
                        break
                elif success_now:
                    stopped = True
                    break
                hold_done = int(self.control_step_count) - hold_start
                if hold_done > 0 and hold_done % max(1, hold_steps // 4) == 0:
                    joint_now = self.env_module.read_scene_joint_angle_deg(
                        self.task_config.joint_prim
                    )
                    self._log_info(
                        f"[INFO] Close hold {hold_done}/{hold_steps}: "
                        f"joint_1={joint_now:.2f}°USD, {self._robot_effort_debug_line()}"
                    )
        self._log_info(f"[INFO] Close sampled effort: {self._robot_effort_debug_line()}")
        return stopped

    def _execute_continuous_calibrated_close(
        self,
        collector: OfficialEpisodeCollector,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        trajectory: list[tuple[np.ndarray, tuple[float, float, float, float]]],
    ) -> bool | None:
        """Track calibrated XY handle arc until success or episode step budget runs out."""
        if self.task_config.task_id not in CALIBRATED_XY_ARC_TASK_IDS or self._close_anchor is None:
            return None
        if len(tuple(getattr(self.sampling_config, "push_contact_joint_arc_points", ()) or ())) < 3:
            return None

        anchor = self._close_anchor
        joint_init = float(anchor["joint_init_deg"])
        joint_target = float(anchor["joint_target_deg"])
        close_clamp = bool(getattr(self.push_cfg, "close_clamp_joints", True))
        target_quat = anchor["contact_quat_w"]
        pos_step = float(getattr(self.push_cfg, "close_push_ee_step_m", self.max_ee_pos_step_m))
        lead_deg = max(float(getattr(self.push_cfg, "close_step_deg_usd", 1.0)), 0.5)
        direction = 1.0 if joint_target >= joint_init else -1.0

        first = np.asarray(trajectory[0][0], dtype=np.float64)
        last = np.asarray(trajectory[-1][0], dtype=np.float64)
        remaining = self._remaining_episode_control_steps()
        budget_text = str(remaining) if remaining is not None else "unlimited"
        self._log_info(
            f"[INFO] Close phase (continuous calibrated arc): remaining_steps={budget_text}, "
            f"joint {joint_init:.2f}°->{joint_target:.2f}° (direction={'+' if direction > 0 else '-'}), "
            f"close_clamp_joints={close_clamp}, "
            f"arc_delta={float(np.linalg.norm(last - first)):.3f}m"
        )

        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        stopped = False
        step_i = 0
        log_interval = max(1, (remaining or 200) // 4)
        joint_prim = self.task_config.joint_prim
        # Drive commanded joint monotonically along planned arc so execution does not
        # stall when live joint readback gets stuck.
        joint_cmd_state = float(joint_init)

        while remaining is None or step_i < remaining:
            joint_live = self.env_module.read_scene_joint_angle_deg(joint_prim)
            if joint_live is None:
                joint_live = joint_init
            joint_live = float(joint_live)
            if direction > 0:
                joint_cmd_state = min(joint_target, joint_cmd_state + lead_deg)
            else:
                joint_cmd_state = max(joint_target, joint_cmd_state - lead_deg)
            joint_cmd = float(joint_cmd_state)

            evaluated = self._calibrated_arc_position_at_joint(joint_cmd)
            if evaluated is None:
                step_i += 1
                continue
            target_pos, _ = evaluated
            corrected_target = self._apply_close_contact_anchor_correction(target_pos)
            self._advance_tracking_pose(corrected_target, target_quat, max_pos_step_m=pos_step)
            arm_targets = self._ik_targets_for_pose(self._tracking_pos, self._tracking_quat_wxyz)
            if self._control_step_ik(
                collector,
                arm_targets,
                gripper_open=False,
                on_control_step=on_control_step,
                clamp_joints=close_clamp,
            ):
                stopped = True
                break

            success_now, joint_degs = evaluate_rollout_success(
                self.env_module, self.success_specs
            )
            if on_control_step is not None:
                if on_control_step(success_now, joint_degs):
                    stopped = True
                    break
            elif success_now:
                stopped = True
                break

            step_i += 1
            if step_i % log_interval == 0:
                self._log_info(
                    f"[INFO] Continuous close step={step_i}: "
                    f"cmd_joint={joint_cmd:.2f}°USD live_joint={float(joint_live):.2f}°USD, "
                    f"{self._robot_effort_debug_line()}"
                )

        self._log_info(f"[INFO] Close continuous effort: {self._robot_effort_debug_line()}")
        return stopped

    def _execute_close_phase(
        self,
        collector: OfficialEpisodeCollector,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        *,
        trajectory: list[tuple[np.ndarray, tuple[float, float, float, float]]] | None = None,
    ) -> bool:
        trajectory = trajectory if trajectory is not None else self._build_close_trajectory_from_anchor()
        if len(trajectory) < 2:
            if self._close_anchor is not None:
                anchor = self._close_anchor
                print(
                    "[WARN] Push phase aborted: calibrated close trajectory unavailable "
                    f"(joint {float(anchor['joint_init_deg']):.1f}° -> "
                    f"{float(anchor['joint_target_deg']):.1f}°)",
                    flush=True,
                )
            return False

        if self._close_anchor is not None:
            anchor = self._close_anchor
            arc_mode = (
                "calibrated XY arc"
                if self.task_config.task_id in CALIBRATED_XY_ARC_TASK_IDS
                else "T_rel"
            )
            print(
                f"[INFO] Push phase ({arc_mode}): {len(trajectory)} waypoints, "
                f"joint {float(anchor['joint_init_deg']):.1f}° -> "
                f"{float(anchor['joint_target_deg']):.1f}°",
                flush=True,
            )

        stopped = self._execute_sampled_pose_path_close(
            collector, on_control_step, trajectory=trajectory
        )

        joint_prim = self.task_config.joint_prim
        joint_init = float(self._close_anchor["joint_init_deg"]) if self._close_anchor else 15.0
        joint_fin = self.env_module.read_scene_joint_angle_deg(joint_prim)
        fin = float(joint_fin or joint_init)
        if self.verbose:
            print(
                f"[INFO] Close finished: {joint_prim} {joint_init:.2f}°USD(real≈"
                f"{usd_joint_to_real_lid_deg(joint_init):.0f}°) -> {fin:.2f}°USD(real≈"
                f"{usd_joint_to_real_lid_deg(fin):.0f}°). "
                f"Motion ended; process stays alive until Ctrl+C.",
                flush=True,
            )
        else:
            fin_print = min(fin, 100.0)
            print(
                f"[INFO] Close: {joint_prim} {joint_init:.1f}° -> {fin_print:.1f}°USD",
                flush=True,
            )
        return stopped

    def _execute_hinge_close_anchored(
        self,
        collector: OfficialEpisodeCollector,
        on_control_step: Callable[[bool, dict[str, float | None]], bool] | None,
        *,
        trajectory: list[tuple[np.ndarray, tuple[float, float, float, float]]] | None = None,
    ) -> bool:
        """Close after contact: ArticuBot T_rel from contact anchor (fixed link_init + eef_in_link)."""
        if self._close_anchor is None:
            return False
        return self._execute_close_phase(collector, on_control_step, trajectory=trajectory)

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

        self._log_info(
            f"[INFO] Handle reach: {n_approach} approach + {len(contact_poses)} contact waypoints "
            f"({approach_steps_per} steps/wp approach, {contact_steps_per} steps/wp contact, pose IK)"
        )
        self._log_info(
            f"[INFO] Direct approach path ({len(approach_path)} waypoint): "
            + " -> ".join(str(np.round(wp, 3).tolist()) for wp in approach_path)
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
        normal_alignment = self._normal_alignment_report(
            contact_w,
            approach_w,
            quat_w,
            actual_quat,
        )
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
            ref_contact_w = self._map_calibrated_world_position(
                np.asarray(cfg.reference_contact_world, dtype=np.float64)
            )
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
            "push_normal_w": (
                normal_alignment["push_normal_w"].copy() if normal_alignment is not None else None
            ),
            "planned_normal_dot": (
                normal_alignment["planned_normal_dot"] if normal_alignment is not None else None
            ),
            "actual_normal_dot": (
                normal_alignment["actual_normal_dot"] if normal_alignment is not None else None
            ),
            "planned_normal_angle_deg": (
                normal_alignment["planned_normal_angle_deg"] if normal_alignment is not None else None
            ),
            "actual_normal_angle_deg": (
                normal_alignment["actual_normal_angle_deg"] if normal_alignment is not None else None
            ),
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
        if not self.verbose:
            return
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
        if report.get("push_normal_w") is not None:
            print(
                f"  push normal_w     : {np.round(report['push_normal_w'], 4).tolist()} "
                f"(approach -> contact)",
                flush=True,
            )
            print(
                f"  EE +Z vs normal   : planned dot={float(report['planned_normal_dot']):.4f}, "
                f"angle={float(report['planned_normal_angle_deg']):.1f}°; "
                f"actual dot={float(report['actual_normal_dot']):.4f}, "
                f"angle={float(report['actual_normal_angle_deg']):.1f}° "
                "(0° aligned, 90° perpendicular)",
                flush=True,
            )
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
            f"T={report.get('control_steps', '?')}",
        ]
        if report.get("probe_steps_limit") is not None:
            parts.append(f"probe_steps={report.get('probe_steps_limit')}")
        if report.get("episode_step_limit") is not None:
            parts.append(f"episode_limit={report.get('episode_step_limit')}")
        if report.get("episode_step_limit_hit"):
            parts.append("timeout=True")
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
            f"T={self.control_step_count}",
        ]
        if result.get("probe_steps_limit") is not None:
            parts.append(f"probe_steps={result.get('probe_steps_limit')}")
        if result.get("episode_step_limit") is not None:
            parts.append(f"episode_limit={result.get('episode_step_limit')}")
        if result.get("episode_step_limit_hit"):
            parts.append("timeout=True")
        if joint_deg is not None:
            parts.append(f"{joint_prim}={float(joint_deg):.2f}deg")
        print("\n" + " | ".join(parts), flush=True)
        if not success and joint_deg is not None:
            success_hints = []
            for spec in self.success_specs:
                if getattr(spec, "joint_prim", None) is None:
                    continue
                if getattr(spec, "angle_gt_deg", None) is not None:
                    success_hints.append(f"{spec.joint_prim} > {float(spec.angle_gt_deg):.1f}°")
                if getattr(spec, "angle_lt_deg", None) is not None:
                    success_hints.append(f"{spec.joint_prim} < {float(spec.angle_lt_deg):.1f}°")
            success_msg = ", ".join(success_hints) if success_hints else "task_registry success spec"
            print(
                f"[PUSH+CLOSE EXIT] Success needs {success_msg} (task_registry). "
                "Check Close push logs if lid did not move.",
                flush=True,
            )

    def run_yaml_handle_contact_only_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        max_servo_steps: int = 400,
        episode_step_limit: int | None = None,
    ) -> dict[str, object]:
        """Move to yaml handle contact only; print report and stop (no close phase)."""
        prev_skip = self._skip_recording
        self._skip_recording = True
        self.sim_step_count = 0
        self.control_step_count = 0
        self.set_episode_step_limit(episode_step_limit)
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
            self._log_info(
                f"[INFO] yaml_handle contact-only: hinge_lever={lever_m:.4f}m "
                f"contact_w={np.round(contact_w, 4).tolist()} "
                f"approach_w={np.round(approach_w, 4).tolist()}"
            )

            contact_steps_per = max(8, int(self.push_cfg.contact_hold_steps))
            approach_budget = max(
                int(self.push_cfg.approach_steps),
                int(max_servo_steps) - contact_steps_per * max(2, int(self.push_cfg.contact_hold_steps)),
            )
            try:
                self.set_trace_phase("approach")
                self._execute_handle_reach(
                    collector,
                    approach_w,
                    contact_w,
                    quat_w,
                    on_control_step=None,
                    approach_step_budget=approach_budget,
                    contact_steps_per_waypoint=contact_steps_per,
                )
            except EpisodeStepLimitExceeded as exc:
                print(f"[WARN] yaml_handle contact-only timeout: {exc}; reporting current pose.", flush=True)

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
            report["probe_steps_limit"] = int(max_servo_steps)
            report["episode_step_limit"] = self.episode_step_limit
            report["episode_step_limit_hit"] = self.episode_step_limit_hit
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
        episode_step_limit: int | None = None,
    ) -> tuple[bool, dict[str, float | None]]:
        """Approach yaml/HDF5 handle, contact, then hinge-relative close from actual EE anchor."""
        self.sim_step_count = 0
        self.control_step_count = 0
        self.set_episode_step_limit(episode_step_limit)
        self.episode_start_wall_time = episode_start_wall_time
        self.last_record = None
        self._reset_episode_log_state()
        self._close_hinge_lever_m = float(hinge_lever_m) if hinge_lever_m is not None else None
        self._reset_ee_tracking_from_robot()
        self.diff_ik_controller.reset()
        self.diff_ik_pos_controller.reset()

        approach_w = approach_from_contact(
            contact_w,
            quat_w,
            float(self.push_cfg.approach_backoff_m),
        )

        lever_msg = f" hinge_lever={hinge_lever_m:.4f}m" if hinge_lever_m is not None else ""
        self._log_info(
            f"[INFO] {label} push:{lever_msg} "
            f"contact_w={np.round(contact_w, 4).tolist()} "
            f"approach_w={np.round(approach_w, 4).tolist()}"
        )

        final_joint_degs: dict[str, float | None] = {}

        def on_step(success_now: bool, joint_degs: dict[str, float | None]) -> bool:
            nonlocal final_joint_degs
            final_joint_degs = joint_degs
            return success_now

        try:
            task_success = False
            self.set_trace_phase("approach")
            if self._execute_handle_reach(
                collector,
                approach_w,
                contact_w,
                quat_w,
                on_control_step=on_step,
            ):
                task_success = True
            else:
                actual_contact_pos, actual_contact_quat = self._read_ee_pose()
                contact_drift = float(np.linalg.norm(actual_contact_pos - contact_w))
                rot_drift = (
                    _wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_contact_quat).inv()
                ).magnitude()
                self._announce_phase_transition(
                    "approach",
                    "push",
                    detail=(
                        f"contact_drift={contact_drift:.4f}m "
                        f"rot={math.degrees(rot_drift):.1f}deg"
                    ),
                )
                self.set_trace_phase("close")
                self._log_info(
                    f"[INFO] Contact anchor: planned={np.round(contact_w, 4).tolist()} "
                    f"actual={np.round(actual_contact_pos, 4).tolist()} "
                    f"drift={contact_drift:.4f}m rot={math.degrees(rot_drift):.1f}deg"
                )
                self._log_normal_alignment(
                    "Contact anchor",
                    contact_w,
                    approach_w,
                    quat_w,
                    actual_contact_quat,
                )

                self._reset_close_phase_tracking()
                self._init_close_anchor(actual_contact_pos, actual_contact_quat)
                if self._execute_hinge_close_anchored(collector, on_step):
                    task_success = True
                else:
                    task_success, final_joint_degs = self._episode_success()
        except EpisodeStepLimitExceeded as exc:
            task_success, final_joint_degs = self._episode_success()
            print(f"[WARN] {label} episode timeout: {exc}; mark attempt failed.", flush=True)
        except EeHandleDistExceeded as exc:
            task_success, final_joint_degs = self._episode_success()
            task_success = False
            print(f"[WARN] {label} EE drift abort: {exc}; mark attempt failed.", flush=True)
        except RecordingHealthError as exc:
            task_success, final_joint_degs = self._episode_success()
            task_success = False
            print(f"[WARN] {label} recording health abort: {exc}; mark attempt failed.", flush=True)

        if task_success:
            self.run_home_reset(collector)

        return task_success, final_joint_degs

    def run_yaml_handle_push(
        self,
        collector: OfficialEpisodeCollector,
        episode_start_wall_time: float,
        *,
        episode_step_limit: int | None = None,
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
            episode_step_limit=episode_step_limit,
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
        episode_step_limit: int | None = None,
    ) -> dict[str, object]:
        self.set_episode_step_limit(episode_step_limit)
        self._reset_episode_log_state()
        approach_w = approach_from_contact(
            contact_w,
            quat_w,
            float(self.push_cfg.approach_backoff_m),
        )

        link_prim = self.task_config.link_prim

        lever_msg = f" hinge_lever={hinge_lever_m:.4f}m" if hinge_lever_m is not None else ""
        self._log_info(
            f"[INFO] {label} probe:{lever_msg} "
            f"contact_w={np.round(contact_w, 4).tolist()} "
            f"approach_w={np.round(approach_w, 4).tolist()}"
        )

        contact_steps_per = max(8, int(self.push_cfg.contact_hold_steps))
        approach_budget = max(
            int(self.push_cfg.approach_steps),
            int(max_servo_steps) - contact_steps_per * max(2, int(self.push_cfg.contact_hold_steps)),
        )
        close_poses: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
        actual_contact_pos = np.asarray(contact_w, dtype=np.float64)
        actual_contact_quat = quat_w
        contact_drift = 0.0
        rot_drift = 0.0
        try:
            self.set_trace_phase("approach")
            joint_success_early = self._execute_handle_reach(
                collector,
                approach_w,
                contact_w,
                quat_w,
                on_control_step=self._stop_on_joint_success,
                approach_step_budget=approach_budget,
                contact_steps_per_waypoint=contact_steps_per,
            )
            if joint_success_early:
                actual_contact_pos, actual_contact_quat = self._read_ee_pose()
                contact_drift = float(np.linalg.norm(actual_contact_pos - contact_w))
                rot_drift = (
                    _wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_contact_quat).inv()
                ).magnitude()
                self._log_info(
                    f"[INFO] {label} probe: joint success during handle reach; "
                    f"skip close phase at step={self.control_step_count}."
                )
            else:
                actual_contact_pos, actual_contact_quat = self._read_ee_pose()
                contact_drift = float(np.linalg.norm(actual_contact_pos - contact_w))
                rot_drift = (
                    _wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_contact_quat).inv()
                ).magnitude()
                self._announce_phase_transition(
                    "approach",
                    "push",
                    detail=(
                        f"contact_drift={contact_drift:.4f}m "
                        f"rot={math.degrees(rot_drift):.1f}deg"
                    ),
                )
                self.set_trace_phase("close")
                self._log_info(
                    f"[INFO] Contact anchor: planned={np.round(contact_w, 4).tolist()} "
                    f"actual={np.round(actual_contact_pos, 4).tolist()} "
                    f"drift={contact_drift:.4f}m rot={math.degrees(rot_drift):.1f}deg"
                )
                self._log_normal_alignment(
                    "Contact anchor",
                    contact_w,
                    approach_w,
                    quat_w,
                    actual_contact_quat,
                )

                self._reset_close_phase_tracking()
                self._init_close_anchor(actual_contact_pos, actual_contact_quat)
                close_poses = self._build_close_trajectory_from_anchor()
                if close_poses:
                    self._execute_hinge_close_anchored(
                        collector,
                        on_control_step=self._stop_on_joint_success,
                        trajectory=close_poses,
                    )
        except EpisodeStepLimitExceeded as exc:
            actual_contact_pos, actual_contact_quat = self._read_ee_pose()
            contact_drift = float(np.linalg.norm(actual_contact_pos - contact_w))
            rot_drift = (
                _wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_contact_quat).inv()
            ).magnitude()
            print(f"[WARN] {label} probe step budget exhausted: {exc}", flush=True)
        except EeHandleDistExceeded as exc:
            actual_contact_pos, actual_contact_quat = self._read_ee_pose()
            contact_drift = float(np.linalg.norm(actual_contact_pos - contact_w))
            rot_drift = (
                _wxyz_to_rot(quat_w) * _wxyz_to_rot(actual_contact_quat).inv()
            ).magnitude()
            print(f"[WARN] {label} probe EE drift abort: {exc}", flush=True)

        anchor = self._close_anchor
        if anchor is not None:
            hinge_origin_w = np.asarray(anchor["hinge_origin_w"], dtype=np.float64)
            hinge_axis_w = np.asarray(anchor["hinge_axis_w"], dtype=np.float64)
        else:
            hinge_origin_w, hinge_axis_w = self.env_module.get_hinge_world_frame(
                link_prim,
                self.sampling_config.hinge.origin,
                self.sampling_config.hinge.axis,
            )

        success_now, joint_degs = self._episode_success()
        joint_deg = joint_degs.get(self.task_config.joint_prim)
        if success_now:
            self.run_home_reset(collector)
        elif self.verbose:
            joint_ok, _ = evaluate_rollout_success(self.env_module, self.success_specs)
            within_limit = (
                self.episode_step_limit is None
                or int(self.control_step_count) <= int(self.episode_step_limit)
            )
            if not joint_ok:
                print(
                    "[INFO] Probe failed before success threshold; keeping final pose for inspection.",
                    flush=True,
                )
            elif not within_limit:
                print(
                    "[INFO] Probe failed: episode step limit exceeded; keeping final pose for inspection.",
                    flush=True,
                )
        if self.verbose:
            if joint_deg is not None:
                print(
                    f"[INFO] {label} probe finished: success={success_now} "
                    f"{self.task_config.joint_prim}={joint_deg:.2f}deg",
                    flush=True,
                )
            else:
                print(
                    f"[INFO] {label} probe finished: success={success_now} joints={joint_degs}",
                    flush=True,
                )

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
            "probe_steps_limit": int(max_servo_steps),
            "episode_step_limit": self.episode_step_limit,
            "episode_step_limit_hit": self.episode_step_limit_hit,
        }

    def run_yaml_handle_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        max_servo_steps: int = 400,
        episode_step_limit: int | None = None,
    ) -> dict[str, object]:
        """Livestream debug: yaml handle approach + contact + hinge-relative close."""
        prev_skip = self._skip_recording
        self._skip_recording = True
        self.sim_step_count = 0
        self.control_step_count = 0
        self._reset_episode_log_state()
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
                episode_step_limit=episode_step_limit,
            )
        finally:
            self._skip_recording = prev_skip

    def run_articulation_calibrated_probe(
        self,
        collector: OfficialEpisodeCollector,
        *,
        max_servo_steps: int = 400,
        episode_step_limit: int | None = None,
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
                episode_step_limit=episode_step_limit,
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
            f"[INFO] Direct approach path ({len(approach_path)} waypoint): "
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

    def _tensor_stats_for_ids(self, tensor: torch.Tensor | None, ids) -> dict[str, object] | None:
        if tensor is None:
            return None
        try:
            values = tensor[0, ids].detach().cpu().numpy().astype(float)
        except Exception:
            return None
        if values.size == 0:
            return None
        return {
            "max_abs": float(np.max(np.abs(values))),
            "values": np.round(values, 4).tolist(),
        }

    def _robot_effort_debug_line(self) -> str:
        parts: list[str] = []
        current = self.robot.data.joint_pos[0, self.handles.arm_joint_ids]
        if self._last_arm_targets is not None:
            try:
                err = (self._last_arm_targets[0] - current).detach().cpu().numpy().astype(float)
                parts.append(
                    f"arm_target_err_max={float(np.max(np.abs(err))):.4f}rad "
                    f"err={np.round(err, 4).tolist()}"
                )
            except Exception as exc:
                parts.append(f"arm_target_err=unavailable({exc})")

        for attr_name in (
            "applied_torque",
            "computed_torque",
            "joint_effort",
            "joint_torque",
            "joint_vel",
        ):
            stats = self._tensor_stats_for_ids(
                getattr(self.robot.data, attr_name, None),
                self.handles.arm_joint_ids,
            )
            if stats is not None:
                parts.append(f"robot.{attr_name}_max={stats['max_abs']:.4f} values={stats['values']}")

        scene = self.env_module.scene_articulation
        joint_prim = self.task_config.joint_prim
        if scene is not None:
            joint_name = joint_prim.rsplit("/", 1)[-1]
            try:
                joint_ids, _ = scene.find_joints(joint_name)
                if joint_ids:
                    jid = [int(joint_ids[0])]
                    for attr_name in (
                        "applied_torque",
                        "computed_torque",
                        "joint_effort",
                        "joint_torque",
                        "joint_vel",
                    ):
                        stats = self._tensor_stats_for_ids(getattr(scene.data, attr_name, None), jid)
                        if stats is not None:
                            parts.append(
                                f"scene.{joint_name}.{attr_name}={stats['values'][0]}"
                            )
            except Exception as exc:
                parts.append(f"scene_effort=unavailable({exc})")

        return " | ".join(parts) if parts else "effort_diag=unavailable"

    def _apply_arm_command(self, arm_targets: torch.Tensor, gripper_open: bool) -> None:
        self._last_arm_targets = arm_targets.detach().clone()
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
        self._raise_if_episode_step_limit_reached()
        if clamp_joints:
            arm_targets = self._clamp_arm_targets(arm_targets)
        self.control_step_count += 1
        if self.control_step_count % max(1, int(self.progress_log_interval)) == 0:
            limit_text = (
                str(self.episode_step_limit)
                if self.episode_step_limit is not None
                else "none"
            )
            phase = self._trace_phase or "run"
            print(
                f"[STEP] phase={phase} control_steps={self.control_step_count}/{limit_text}",
                flush=True,
            )
        self._apply_arm_command(arm_targets, gripper_open)
        self._sim_substeps()
        if not gripper_open:
            self._reassert_gripper_closed()
        self._maybe_record(collector, arm_targets, gripper_open)
        if self.recording_health_failed:
            raise RecordingHealthError(self._recording_health_reason)
        self._maybe_trace_ee_handle()
        self._raise_if_ee_handle_dist_exceeded()
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
        close_anchor_correction: bool = False,
    ) -> bool:
        """Pose servo with incremental target advance (position + rotation)."""
        target_pos = np.asarray(target_pos, dtype=np.float64)
        pos_tol = POSE_REACH_TOL_M if pos_tol_m is None else float(pos_tol_m)
        rot_tol = POSE_REACH_ROT_RAD if rot_tol_rad is None else float(rot_tol_rad)
        for _ in range(max_steps):
            servo_target_pos = (
                self._apply_close_contact_anchor_correction(target_pos)
                if close_anchor_correction
                else target_pos
            )
            ee_pos = self.robot.data.body_pos_w[0, self.handles.ee_body_id].detach().cpu().numpy()
            ee_quat = tuple(
                float(v) for v in self.robot.data.body_quat_w[0, self.handles.ee_body_id].tolist()
            )
            pos_err = float(np.linalg.norm(servo_target_pos - ee_pos))
            r_err = (_wxyz_to_rot(target_quat_wxyz) * _wxyz_to_rot(ee_quat).inv()).magnitude()
            if pos_err <= pos_tol and r_err <= rot_tol:
                return False

            self._advance_tracking_pose(servo_target_pos, target_quat_wxyz)
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
        close_anchor_correction: bool = False,
    ) -> bool:
        for pos, quat in poses:
            if self._move_to_pose(
                collector,
                pos,
                quat,
                steps_per_waypoint,
                on_control_step,
                clamp_joints=clamp_joints,
                pos_tol_m=0.006 if close_anchor_correction else None,
                close_anchor_correction=close_anchor_correction,
            ):
                return True
        return False

    def _hold_arm_at_current_pose(
        self,
        collector: OfficialEpisodeCollector,
        *,
        duration_sec: float = 0.5,
        record: bool = False,
    ) -> None:
        """Hold current arm pose (gripper closed) before homing."""
        hold_steps = max(1, int(round(duration_sec * self.timing.control_hz)))
        arm_targets = self.robot.data.joint_pos[0, self.handles.arm_joint_ids].detach().clone()
        arm_targets = arm_targets.unsqueeze(0)
        print(f"[INFO] Success hold: {hold_steps} steps ({duration_sec:.1f}s)", flush=True)
        for _ in range(hold_steps):
            self.control_step_count += 1
            self._apply_arm_command(arm_targets, gripper_open=False)
            self._sim_substeps()
            if record:
                self._maybe_record(collector, arm_targets, gripper_open=False)

    def _restore_arm_home(self, collector: OfficialEpisodeCollector, *, record: bool) -> None:
        from motion_planner import interpolate_joint_segment

        current_arm = tuple(
            float(v) for v in self.robot.data.joint_pos[0, self.handles.arm_joint_ids].tolist()
        )
        num_steps = max(1, int(self.home_steps))
        print(f"[INFO] Go to zero: {num_steps} steps", flush=True)
        for joint_rad in interpolate_joint_segment(current_arm, self.home_arm_rad, num_steps):
            self.control_step_count += 1
            arm_targets = torch.tensor([list(joint_rad)], dtype=torch.float32, device=self.device)
            self._apply_arm_command(arm_targets, gripper_open=False)
            self._sim_substeps()
            if record:
                self._maybe_record(collector, arm_targets, gripper_open=False)
        self._reset_ee_tracking_from_robot()


    def run_home_reset(self, collector: OfficialEpisodeCollector) -> None:
        """Return arm joints to zero after successful task (recorded into episode)."""
        self._hold_arm_at_current_pose(collector, duration_sec=0.5, record=True)
        self._restore_arm_home(collector, record=True)
