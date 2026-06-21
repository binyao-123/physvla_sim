from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from task_registry import SCENE_ARTICULATION_PRIM_PATH


DEFAULT_SCENE_ROOT_PRIM = SCENE_ARTICULATION_PRIM_PATH

# After sim.reset(), scene hinge state may lag until physics steps catch up USD physics:position.
SCENE_JOINT_PHYSICS_WARMUP_STEPS = 24


@dataclass
class CameraPrimSpec:
	name: str
	prim_path: str
	translation: tuple[float, float, float] | None
	# Isaac Sim Transform panel Orient XYZ values, in degrees.
	# None means keeping the value authored in the USD scene.
	rotation_xyz: tuple[float, float, float] | None
	focal_length: float | None
	enable_sensor_capture: bool = True


@dataclass
class JointDrivePrimSpec:
	prim_path: str
	damping: float | None = None
	stiffness: float | None = None
	max_force: float | None = None
	target_position: float | None = None
	target_velocity: float | None = None


@dataclass
class JointLimitPrimSpec:
	prim_path: str
	lower_limit: float | None = None
	upper_limit: float | None = None


@dataclass
class JointInitialPrimSpec:
	prim_path: str
	position: float


@dataclass
class SceneRootPrimSpec:
	prim_path: str
	translation: tuple[float, float, float]
	# Isaac Transform → Orient 欧拉角 XYZ（度），与相机 rotation_xyz 一致
	rotation_xyz: tuple[float, float, float]
	scale: tuple[float, float, float]


@dataclass
class EnvironmentModuleConfig:
	usd_path: str
	camera_width: int = 640
	camera_height: int = 480
	camera_sensor_type: str = "camera"
	warmup_render_steps: int = 6
	reset_robot_root_pose: bool = False
	camera_specs: list[CameraPrimSpec] = field(default_factory=list)
	joint_drive_specs: list[JointDrivePrimSpec] = field(default_factory=list)
	joint_limit_specs: list[JointLimitPrimSpec] = field(default_factory=list)
	joint_initial_specs: list[JointInitialPrimSpec] = field(default_factory=list)
	scene_root_specs: list[SceneRootPrimSpec] = field(default_factory=list)
	quiet_logging: bool = False


def apply_camera_launch_workarounds(args_cli: Any) -> Any:
	"""Enable camera rendering / offscreen / kit_args workarounds for IsaacLab + Isaac Sim 5.1."""
	args_cli.enable_cameras = True

	if getattr(args_cli, "headless", False) and hasattr(args_cli, "offscreen_render"):
		args_cli.offscreen_render = True

	if hasattr(args_cli, "kit_args"):
		stable_kit_args = (
			" --/rtx/post/dlss/execMode=0"
			" --/app/runLoops/main/rateLimitEnabled=false"
			" --/app/runLoops/main/manualModeEnabled=true"
			" --enable omni.kit.loop-isaac"
		)
		args_cli.kit_args = f"{args_cli.kit_args or ''}{stable_kit_args}"

	return args_cli


class IsaacLabEnvironmentModule:
	"""Open USD stage, SimulationContext, articulation, cameras, warmup/reset."""

	def __init__(self, cfg: EnvironmentModuleConfig):
		self.cfg = cfg
		self.sim = None
		self.robot = None
		self.scene_articulation = None
		self.sensor_cameras: dict[str, Any] = {}
		self.device: str | None = None
		self._camera_paths: dict[str, str] = {}
		self._scene_joint_articulation_ids: dict[str, int] = {}
		self._scene_joint_pos_units: dict[str, str] = {}
		self._scene_root_baseline: dict[str, tuple[float, ...] | None] = {
			"translate": None,
			"orient_wxyz": None,
			"scale": None,
		}

	def _robot_is_fixed_base(self) -> bool:
		if self.robot is None:
			return False
		for attr_name in ("is_fixed_base", "fixed_base"):
			value = getattr(self.robot, attr_name, None)
			if isinstance(value, bool):
				return value
			if hasattr(value, "item"):
				try:
					return bool(value.item())
				except Exception:
					pass
		return False

	def _should_reset_root_pose(self) -> bool:
		return bool(self.cfg.reset_robot_root_pose) and (not self._robot_is_fixed_base())

	def apply_joint_drive_overrides(self):
		if not self.cfg.joint_drive_specs:
			return

		import omni.usd
		from pxr import Sdf

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			raise RuntimeError("USD stage is unavailable for joint drive overrides.")

		drive_attrs = {
			"damping": "drive:angular:physics:damping",
			"stiffness": "drive:angular:physics:stiffness",
			"max_force": "drive:angular:physics:maxForce",
			"target_position": "drive:angular:physics:targetPosition",
			"target_velocity": "drive:angular:physics:targetVelocity",
		}

		for spec in self.cfg.joint_drive_specs:
			prim = stage.GetPrimAtPath(spec.prim_path)
			if not prim.IsValid():
				raise RuntimeError(f"Joint drive prim '{spec.prim_path}' does not exist on stage.")

			for field_name, attr_name in drive_attrs.items():
				value = getattr(spec, field_name)
				if value is None:
					continue
				attr = prim.GetAttribute(attr_name)
				if not attr.IsValid():
					attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
				attr.Set(float(value))

	def apply_joint_limit_overrides(self):
		if not self.cfg.joint_limit_specs:
			return

		import omni.usd
		from pxr import Sdf

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			raise RuntimeError("USD stage is unavailable for joint limit overrides.")

		limit_attrs = {
			"lower_limit": "physics:lowerLimit",
			"upper_limit": "physics:upperLimit",
		}

		for spec in self.cfg.joint_limit_specs:
			prim = stage.GetPrimAtPath(spec.prim_path)
			if not prim.IsValid():
				raise RuntimeError(f"Joint limit prim '{spec.prim_path}' does not exist on stage.")

			for field_name, attr_name in limit_attrs.items():
				value = getattr(spec, field_name)
				if value is None:
					continue
				attr = prim.GetAttribute(attr_name)
				if not attr.IsValid():
					attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
				attr.Set(float(value))

	def apply_joint_initial_overrides(self):
		if not self.cfg.joint_initial_specs:
			return

		import omni.usd
		from pxr import Sdf

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			raise RuntimeError("USD stage is unavailable for joint initial overrides.")

		for spec in self.cfg.joint_initial_specs:
			prim = stage.GetPrimAtPath(spec.prim_path)
			if not prim.IsValid():
				raise RuntimeError(f"Joint prim '{spec.prim_path}' does not exist on stage.")

			value = float(spec.position)

			auth_attr = prim.GetAttribute("physics:position")
			if not auth_attr.IsValid():
				auth_attr = prim.CreateAttribute("physics:position", Sdf.ValueTypeNames.Float)
			auth_attr.Set(value)

			# Write state:angular:physics:position ONLY before sim starts. After sim.reset()
			# with PhysX Direct GPU API, writing live state锁死 joint / 触发非法 CPU PhysX API.
			if self.sim is None:
				state_attr = prim.GetAttribute("state:angular:physics:position")
				if state_attr and state_attr.IsValid():
					state_attr.Set(value)

	def _read_usd_authored_joint_angle_deg(self, prim_path: str) -> float | None:
		"""USD physics:position (authoring default), not guaranteed to match live PhysX state."""
		import omni.usd

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			return None
		prim = stage.GetPrimAtPath(prim_path)
		if not prim or not prim.IsValid():
			return None
		attr = prim.GetAttribute("physics:position")
		if not attr or not attr.IsValid():
			return None
		value = attr.Get()
		return float(value) if value is not None else None

	def _read_scene_joint_angle_deg_usd(self, prim_path: str) -> float | None:
		import omni.usd

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			return None
		prim = stage.GetPrimAtPath(prim_path)
		if not prim or not prim.IsValid():
			return None

		state_val: float | None = None
		drive_val: float | None = None
		auth_val: float | None = None
		for attr_name in (
			"state:angular:physics:position",
			"drive:angular:physics:targetPosition",
			"drive:angular:physics:position",
			"physics:position",
		):
			attr = prim.GetAttribute(attr_name)
			if not attr or not attr.IsValid():
				continue
			value = attr.Get()
			if value is None:
				continue
			val = float(value)
			if attr_name == "state:angular:physics:position":
				state_val = val
			elif attr_name.startswith("drive:angular"):
				drive_val = val
			else:
				auth_val = val

		# This is a USD fallback only. Live articulation state should come from
		# scene_articulation.data.joint_pos and is converted from radians above.
		# Prefer authored defaults over stale state:* attributes when possible.
		if auth_val is not None:
			return auth_val
		if state_val is not None:
			return state_val
		if drive_val is not None:
			return drive_val
		return None

	def _articulation_joint_pos_to_deg(self, joint_prim_path: str, raw: float) -> float:
		# IsaacLab/PhysX articulation buffers store angular DOFs in radians.
		# Do not infer units from USD state:* attributes: those can be stale after
		# Direct-GPU-safe tensor resets and caused raw radians to be reported as
		# USD degrees (e.g. 0.30 rad being printed as 0.30°).
		self._scene_joint_pos_units[joint_prim_path] = "rad"
		return math.degrees(float(raw))

	def read_scene_joint_angle_deg(self, prim_path: str) -> float | None:
		"""Read scene hinge angle in USD asset degrees (matches task_registry limits)."""
		usd_deg = self._read_scene_joint_angle_deg_usd(prim_path)

		art_deg: float | None = None
		if self.scene_articulation is not None and self.sim is not None and self.sim.is_playing():
			joint_name = prim_path.rsplit("/", 1)[-1]
			if prim_path not in self._scene_joint_articulation_ids:
				joint_ids, _ = self.scene_articulation.find_joints(joint_name)
				if joint_ids:
					self._scene_joint_articulation_ids[prim_path] = int(joint_ids[0])

			joint_id = self._scene_joint_articulation_ids.get(prim_path)
			if joint_id is not None:
				try:
					raw = self.scene_articulation.data.joint_pos[0, joint_id]
					if hasattr(raw, "item"):
						raw = raw.item()
					art_deg = self._articulation_joint_pos_to_deg(prim_path, float(raw))
				except Exception as exc:
					print(f"[WARN] Scene joint read failed for {prim_path}: {exc}")

		if art_deg is not None:
			return float(art_deg)
		if usd_deg is not None:
			return float(usd_deg)
		return None

	def create_simulation(self, dt: float = 1.0 / 60.0, render_interval: int = 4, use_fabric: bool = True):
		import omni.usd
		import isaaclab.sim as sim_utils

		omni.usd.get_context().open_stage(self.cfg.usd_path)
		self.apply_joint_drive_overrides()
		self.apply_joint_limit_overrides()
		if self.sim is None:
			self.apply_joint_initial_overrides()
		self.apply_scene_root_overrides()
		if self.cfg.scene_root_specs:
			self.seed_scene_root_baseline_from_config()

		try:
			physx_cfg = sim_utils.PhysxCfg(enable_stabilization=True)
		except Exception:
			physx_cfg = None

		if physx_cfg is None:
			sim_cfg = sim_utils.SimulationCfg(
				dt=dt,
				render_interval=render_interval,
				use_fabric=use_fabric,
				render=sim_utils.RenderCfg(enable_translucency=True),
			)
		else:
			sim_cfg = sim_utils.SimulationCfg(
				dt=dt,
				render_interval=render_interval,
				use_fabric=use_fabric,
				physx=physx_cfg,
				render=sim_utils.RenderCfg(enable_translucency=True),
			)
		self.sim = sim_utils.SimulationContext(sim_cfg)
		self.device = self.sim.device
		return self.sim

	def create_robot(self, robot_cfg: Any):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before robot initialization.")

		from isaaclab.assets import Articulation

		self.robot = Articulation(cfg=robot_cfg)
		return self.robot

	def create_scene_articulation(self, articulation_cfg: Any):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before scene articulation init.")

		from isaaclab.assets import Articulation

		self.scene_articulation = Articulation(cfg=articulation_cfg)
		self._scene_joint_articulation_ids.clear()
		return self.scene_articulation

	def apply_task_preset_joint_initial(self, task_preset: Any) -> None:
		"""Set hinge default joint angle before sim.reset() (per-episode randomization).

		Updates env cfg + USD authored defaults, and patches IsaacLab's scene
		Articulation default buffer before sim.reset(). Do not write live
		state:angular:physics:position; that path triggers illegal PhysX Direct GPU
		API setJointPosition / updateKinematic calls.
		"""
		new_specs = [
			JointInitialPrimSpec(prim_path=spec.prim_path, position=float(spec.position))
			for spec in task_preset.joint_initial_specs
		]
		self.cfg.joint_initial_specs = new_specs

		if self.scene_articulation is not None:
			for spec in new_specs:
				joint_name = spec.prim_path.rsplit("/", 1)[-1]
				joint_ids, _ = self.scene_articulation.find_joints(joint_name)
				if not joint_ids:
					continue
				jid = int(joint_ids[0])
				self.scene_articulation.data.default_joint_pos[0, jid] = math.radians(float(spec.position))
				if hasattr(self.scene_articulation.data, "default_joint_vel"):
					self.scene_articulation.data.default_joint_vel[0, jid] = 0.0

		self._scene_joint_pos_units.clear()
		self._scene_joint_articulation_ids.clear()

		if self.sim is None:
			self.apply_joint_initial_overrides()

	def apply_task_preset_scene_root(self, task_preset: Any) -> None:
		"""Set scene root pose from per-episode TaskPreset randomization."""
		self.cfg.scene_root_specs = [
			SceneRootPrimSpec(
				prim_path=spec.prim_path,
				translation=tuple(float(v) for v in spec.translation),
				rotation_xyz=tuple(float(v) for v in spec.rotation_xyz),
				scale=tuple(float(v) for v in spec.scale),
			)
			for spec in task_preset.scene_root_specs
		]
		self.apply_scene_root_overrides()
		self.seed_scene_root_baseline_from_config()

	def sync_scene_joint_initials_to_sim(self) -> None:
		"""Apply scene hinge initials through the GPU-compatible tensor path."""
		if self.sim is None:
			self.apply_joint_initial_overrides()
			return
		self.reset_scene_joint_initials_via_tensor()

	def refresh_scene_joint_physics_from_usd(self, *, include_initials: bool = True) -> None:
		"""Re-apply joint drive / limits from cfg to USD.

		Live joint initials are not written through USD after the simulation starts;
		use reset_scene_joint_initials_via_tensor() instead.
		"""
		self.apply_joint_drive_overrides()
		self.apply_joint_limit_overrides()
		if include_initials:
			self.apply_joint_initial_overrides()

	def reset_scene_joint_initials_via_tensor(self) -> None:
		"""Set scene articulation joint state with IsaacLab tensor API after sim.reset().

		This is the GPU-compatible reset path recommended by IsaacLab for articulation
		state. It avoids live USD state writes that call CPU PhysX APIs such as
		PxArticulationJointReducedCoordinate::setJointPosition().
		"""
		if self.scene_articulation is None or not self.cfg.joint_initial_specs:
			return

		import torch

		device = self.scene_articulation.data.joint_pos.device
		for spec in self.cfg.joint_initial_specs:
			joint_name = spec.prim_path.rsplit("/", 1)[-1]
			joint_ids, _ = self.scene_articulation.find_joints(joint_name)
			if not joint_ids:
				print(f"[WARN] Scene joint tensor reset skipped: joint '{joint_name}' not found", flush=True)
				continue
			joint_ids = [int(joint_ids[0])]
			target_deg = float(spec.position)
			pos = torch.tensor(
				[[math.radians(target_deg)]],
				dtype=self.scene_articulation.data.joint_pos.dtype,
				device=device,
			)
			vel = torch.zeros_like(pos)
			self.scene_articulation.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids)
			self.scene_articulation.set_joint_position_target(pos, joint_ids=joint_ids)
			if not self.cfg.quiet_logging:
				print(
					f"[TRACE] Scene joint tensor reset: {joint_name} "
					f"target={target_deg:.2f}° raw={float(pos.item()):.4f}rad joint_id={joint_ids[0]}",
					flush=True,
				)
		self.scene_articulation.write_data_to_sim()

	def sync_scene_joints_after_sim_reset(
		self,
		*,
		warmup_steps: int = SCENE_JOINT_PHYSICS_WARMUP_STEPS,
		render: bool = True,
		log_angles: bool = True,
	) -> None:
		"""Re-apply task_registry joint initials via tensor API and warm up PhysX."""
		if not self.cfg.quiet_logging:
			print(
				f"[TRACE] sync_scene_joints_after_sim_reset: refresh USD, warmup_steps={warmup_steps}",
				flush=True,
			)
		self.refresh_scene_joint_physics_from_usd(include_initials=False)
		if self.sim is None:
			return
		if not self.cfg.quiet_logging:
			print("[TRACE] sync_scene_joints_after_sim_reset: tensor joint state reset", flush=True)
		self.reset_scene_joint_initials_via_tensor()
		if not self.cfg.quiet_logging:
			print("[TRACE] sync_scene_joints_after_sim_reset: begin warmup", flush=True)
		for _ in range(max(0, int(warmup_steps))):
			self.sim.step(render=render)
			if self.robot is not None:
				self.robot.update(self.sim.cfg.dt)
			if self.scene_articulation is not None:
				self.scene_articulation.update(self.sim.cfg.dt)
		if not self.cfg.quiet_logging:
			print("[TRACE] sync_scene_joints_after_sim_reset: end warmup", flush=True)
		if self.cfg.quiet_logging or not log_angles or not self.cfg.joint_initial_specs:
			return
		for spec in self.cfg.joint_initial_specs:
			sim_deg = self.read_scene_joint_angle_deg(spec.prim_path)
			auth_deg = self._read_usd_authored_joint_angle_deg(spec.prim_path)
			joint_name = spec.prim_path.rsplit("/", 1)[-1]
			if sim_deg is not None:
				msg = (
					f"[INFO] Scene joint {joint_name}: "
					f"target={spec.position:.1f}° sim={sim_deg:.2f}°"
				)
				if auth_deg is not None:
					msg += f" usd_auth={auth_deg:.2f}°"
				print(msg)
				if abs(sim_deg - float(spec.position)) > 1.0:
					print(
						f"[WARN] Scene joint {joint_name}: sim angle {sim_deg:.2f}° "
						f"!= task_registry target {spec.position:.1f}°"
					)
			else:
				print(f"[INFO] Scene joint {joint_name}: target={spec.position:.1f}° (sim read failed)")

	def reset_robot_pose_via_targets(
		self,
		gripper_targets=None,
		gripper_joint_ids=None,
	) -> None:
		"""Reset robot using position targets (GPU-safe; avoids write_joint_state_to_sim)."""
		if self.robot is None:
			return

		if not self.cfg.quiet_logging:
			print("[TRACE] reset_robot_pose_via_targets: begin", flush=True)
		if self._should_reset_root_pose():
			if not self.cfg.quiet_logging:
				print("[TRACE] reset_robot_pose_via_targets: write root pose", flush=True)
			self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
		self.robot.set_joint_position_target(self.robot.data.default_joint_pos)
		if gripper_targets is not None and gripper_joint_ids is not None:
			self.robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
		self.robot.write_data_to_sim()
		if not self.cfg.quiet_logging:
			print("[TRACE] reset_robot_pose_via_targets: end", flush=True)

	def initialize_robot_home_pose(self):
		if self.sim is None or self.robot is None:
			raise RuntimeError("Simulation and robot must be created before home pose init.")

		self.sim.reset()
		self.robot.update(self.sim.cfg.dt)
		if self.scene_articulation is not None:
			self.scene_articulation.update(self.sim.cfg.dt)
		self.refresh_scene_joint_physics_from_usd(include_initials=False)
		self.reset_scene_joint_initials_via_tensor()

		self.reset_robot_pose_via_targets()

		self.sync_scene_joints_after_sim_reset(warmup_steps=1, log_angles=False)

	@staticmethod
	def _find_xform_op(xformable, op_type):
		for op in xformable.GetOrderedXformOps():
			if op.GetOpType() == op_type:
				return op
		return None

	@staticmethod
	def _editor_orient_xyz_to_quatd(orient_xyz):
		from pxr import Gf

		def _axis_quat(axis: str, degrees: float):
			half_angle = math.radians(float(degrees)) * 0.5
			cos_v = math.cos(half_angle)
			sin_v = math.sin(half_angle)
			if axis == "x":
				return (cos_v, sin_v, 0.0, 0.0)
			if axis == "y":
				return (cos_v, 0.0, sin_v, 0.0)
			return (cos_v, 0.0, 0.0, sin_v)

		def _quat_mul(left, right):
			w0, x0, y0, z0 = left
			w1, x1, y1, z1 = right
			return (
				w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
				w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
				w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
				w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
			)

		quat = (1.0, 0.0, 0.0, 0.0)
		for axis, degrees in zip(("x", "y", "z"), orient_xyz):
			quat = _quat_mul(quat, _axis_quat(axis, degrees))
		w, x, y, z = quat
		return Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))

	def _apply_editor_orient_after_standardize(self, cam_prim, orient_xyz):
		from pxr import UsdGeom

		xformable = UsdGeom.Xformable(cam_prim)
		orient_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
		if orient_op is None:
			raise RuntimeError(
				f"Camera '{cam_prim.GetPath()}' is missing xformOp:orient after standardize_xform_ops."
			)

		orient_quat = self._editor_orient_xyz_to_quatd(orient_xyz)
		orient_op.Set(orient_quat)

	def _set_camera_transform(self, cam, translation, rotation_xyz):
		from pxr import Gf, UsdGeom

		xformable = UsdGeom.Xformable(cam.GetPrim())

		if translation is not None:
			translate_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
			if translate_op is None:
				translate_op = xformable.AddTranslateOp()
			translate_op.Set(Gf.Vec3d(*translation))

		if rotation_xyz is not None:
			rotate_xyz_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeRotateXYZ)
			if rotate_xyz_op is None and self._find_xform_op(xformable, UsdGeom.XformOp.TypeOrient) is None:
				rotate_xyz_op = xformable.AddRotateXYZOp()
			if rotate_xyz_op is not None:
				rotate_xyz_op.Set(Gf.Vec3d(*rotation_xyz))

	def refresh_camera_prim(
		self,
		camera_name: str,
		translation_offset: tuple[float, float, float] | None = None,
		translation_override: tuple[float, float, float] | None = None,
		focal_length_override: float | None = None,
	):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before camera prim refresh.")

		import omni.usd
		import isaaclab.sim as sim_utils
		from pxr import UsdGeom

		stage = omni.usd.get_context().get_stage()

		for spec in self.cfg.camera_specs:
			if spec.name == camera_name:
				translation = translation_override if translation_override is not None else spec.translation
				if translation_offset is not None:
					if translation is None:
						raise ValueError(
							"Camera translation_offset cannot be applied when translation is None."
						)
					if len(translation_offset) != 3:
						raise ValueError("Camera translation_offset must contain exactly 3 values.")
					translation = tuple(
						float(translation[index]) + float(translation_offset[index])
						for index in range(3)
					)

				cam = UsdGeom.Camera.Define(stage, spec.prim_path)
				self._set_camera_transform(cam, translation, spec.rotation_xyz)
				focal_length = focal_length_override if focal_length_override is not None else spec.focal_length
				if focal_length is not None:
					cam.GetFocalLengthAttr().Set(float(focal_length))

				cam_prim = cam.GetPrim()
				if not sim_utils.standardize_xform_ops(cam_prim):
					raise RuntimeError(
						f"Failed to standardize camera xform ops at '{spec.prim_path}'."
					)
				if spec.rotation_xyz is not None:
					self._apply_editor_orient_after_standardize(cam_prim, spec.rotation_xyz)
				self._camera_paths[spec.name] = spec.prim_path
				return spec.prim_path

		raise KeyError(f"Unknown camera spec name '{camera_name}'.")

	def define_camera_prims(self):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before camera prim definition.")

		for spec in self.cfg.camera_specs:
			self.refresh_camera_prim(spec.name)

	def create_sensor_cameras(self):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before sensor camera setup.")

		import omni.usd
		import isaaclab.sim as sim_utils

		try:
			from isaaclab.sensors import Camera as IsaacSensorCamera
			from isaaclab.sensors import CameraCfg
			from isaaclab.sensors import TiledCamera
			from isaaclab.sensors import TiledCameraCfg
		except Exception as exc:
			raise RuntimeError(
				"isaaclab.sensors.Camera/TiledCamera is unavailable in current environment."
			) from exc

		sensor_type = self.cfg.camera_sensor_type.lower()
		if sensor_type not in {"camera", "tiled"}:
			raise ValueError("camera_sensor_type must be 'camera' or 'tiled'.")

		if sensor_type == "tiled":
			cfg_cls = TiledCameraCfg
			sensor_cls = TiledCamera
		else:
			cfg_cls = CameraCfg
			sensor_cls = IsaacSensorCamera

		for _ in range(3):
			self.sim.render()

		self.sensor_cameras = {}
		stage = omni.usd.get_context().get_stage()
		for spec in self.cfg.camera_specs:
			if not spec.enable_sensor_capture:
				continue

			cam_prim = stage.GetPrimAtPath(spec.prim_path)
			if not cam_prim.IsValid():
				raise RuntimeError(f"Camera prim '{spec.prim_path}' does not exist on stage.")
			if not sim_utils.standardize_xform_ops(cam_prim):
				raise RuntimeError(
					f"Failed to standardize camera xform ops at '{spec.prim_path}' before sensor initialization."
				)
			if spec.rotation_xyz is not None:
				self._apply_editor_orient_after_standardize(cam_prim, spec.rotation_xyz)
			cam_cfg = cfg_cls(
				prim_path=spec.prim_path,
				update_period=0.0,
				height=self.cfg.camera_height,
				width=self.cfg.camera_width,
				data_types=["rgb"],
				spawn=None,
			)

			sensor = sensor_cls(cfg=cam_cfg)
			self.sensor_cameras[spec.name] = sensor

		self.sim.reset()
		if self.robot is not None:
			self.robot.update(self.sim.cfg.dt)
			self.reset_robot_pose_via_targets()
		if self.scene_articulation is not None:
			self.scene_articulation.update(self.sim.cfg.dt)
		self.sync_scene_joints_after_sim_reset()

		self.define_camera_prims()

		for sensor in self.sensor_cameras.values():
			sensor.reset()

		for _ in range(max(0, int(self.cfg.warmup_render_steps))):
			self.sim.render()

		return self.sensor_cameras

	def capture_rgb(self, camera_name: str, dt: float | None = None):
		sensor = self.sensor_cameras.get(camera_name)
		if sensor is None:
			return None

		if dt is None:
			dt = float(self.sim.cfg.dt)

		sensor.update(dt)
		rgb = sensor.data.output.get("rgb")
		if rgb is None or rgb.numel() == 0:
			return None

		return rgb[0].clone()

	def validate_camera_mount(
		self,
		camera_name: str,
		*,
		expected_parent_path: str | None = None,
		translation_tolerance_m: float = 1e-3,
	) -> tuple[bool, str]:
		"""Validate camera prim hierarchy/local pose and render sensor availability.

		This catches the failure mode where the wrist camera prim/render product is left
		in a stale or detached state after repeated Isaac/RTX resets. It does not rely
		on image content, so lighting/domain randomization does not affect the check.
		"""
		import omni.usd
		from pxr import Gf, UsdGeom

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			return False, "USD stage is unavailable"

		spec = next((item for item in self.cfg.camera_specs if item.name == camera_name), None)
		if spec is None:
			return False, f"unknown camera '{camera_name}'"

		prim = stage.GetPrimAtPath(spec.prim_path)
		if not prim or not prim.IsValid():
			return False, f"camera prim missing: {spec.prim_path}"

		if expected_parent_path:
			parent_path = str(prim.GetParent().GetPath())
			if parent_path != expected_parent_path:
				return False, f"camera parent={parent_path}, expected={expected_parent_path}"

		if spec.translation is not None:
			translate_op = self._find_xform_op(UsdGeom.Xformable(prim), UsdGeom.XformOp.TypeTranslate)
			if translate_op is None:
				return False, f"camera {camera_name} missing local translate op"
			value = translate_op.Get()
			if value is None:
				return False, f"camera {camera_name} local translate is None"
			delta = Gf.Vec3d(float(value[0]), float(value[1]), float(value[2])) - Gf.Vec3d(*spec.translation)
			if delta.GetLength() > float(translation_tolerance_m):
				return (
					False,
					f"camera {camera_name} local translation drift={delta.GetLength():.6f}m "
					f"value=({float(value[0]):.4f},{float(value[1]):.4f},{float(value[2]):.4f}) "
					f"expected={spec.translation}",
				)

		world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
		for row in range(4):
			for col in range(4):
				if not math.isfinite(float(world[row][col])):
					return False, f"camera {camera_name} has non-finite world transform"

		sensor = self.sensor_cameras.get(camera_name)
		if sensor is None:
			return False, f"camera sensor '{camera_name}' is not registered"

		if hasattr(sensor.data, "pos_w") and sensor.data.pos_w is not None:
			sensor_pos = sensor.data.pos_w[0]
			if not sensor_pos.isfinite().all():
				return False, f"camera sensor '{camera_name}' pos_w has non-finite values"
			usd_pos = world.ExtractTranslation()
			sensor_pos_cpu = sensor_pos.detach().cpu()
			delta_m = math.sqrt(
				(sum((float(sensor_pos_cpu[i]) - float(usd_pos[i])) ** 2 for i in range(3)))
			)
			if delta_m > 0.02:
				return (
					False,
					f"camera sensor '{camera_name}' pos_w disagrees with USD world pose "
					f"by {delta_m:.4f}m sensor={sensor_pos_cpu.tolist()} "
					f"usd=({float(usd_pos[0]):.4f},{float(usd_pos[1]):.4f},{float(usd_pos[2]):.4f})",
				)

		rgb = self.capture_rgb(camera_name)
		if rgb is None:
			return False, f"camera sensor '{camera_name}' produced no rgb"
		if not rgb.isfinite().all():
			return False, f"camera sensor '{camera_name}' rgb has non-finite values"
		return True, ""

	@property
	def camera_paths(self) -> dict[str, str]:
		return dict(self._camera_paths)

	def _scene_root_spec_for_prim(self, prim_path: str) -> SceneRootPrimSpec | None:
		for spec in self.cfg.scene_root_specs:
			if spec.prim_path == prim_path:
				return spec
		if self.cfg.scene_root_specs:
			return self.cfg.scene_root_specs[0]
		return None

	def apply_scene_root_overrides(self) -> None:
		"""Write task_registry scene_root_specs to USD xform ops."""
		if not self.cfg.scene_root_specs:
			return

		import omni.usd
		import isaaclab.sim as sim_utils
		from pxr import Gf, UsdGeom

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			raise RuntimeError("USD stage is unavailable for scene root overrides.")

		for spec in self.cfg.scene_root_specs:
			prim = stage.GetPrimAtPath(spec.prim_path)
			if not prim or not prim.IsValid():
				raise RuntimeError(f"Scene root prim '{spec.prim_path}' does not exist on stage.")
			if not sim_utils.standardize_xform_ops(prim):
				raise RuntimeError(f"Failed to standardize xform ops at '{spec.prim_path}'.")

			xformable = UsdGeom.Xformable(prim)
			translate_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
			orient_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
			scale_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeScale)
			if translate_op is None or orient_op is None:
				raise RuntimeError(
					f"Scene root '{spec.prim_path}' missing translate/orient xform ops."
				)

			translate_op.Set(Gf.Vec3d(*spec.translation))
			orient_op.Set(self._editor_orient_xyz_to_quatd(spec.rotation_xyz))
			if scale_op is None:
				scale_op = xformable.AddScaleOp()
			scale_op.Set(Gf.Vec3d(*spec.scale))
			if not self.cfg.quiet_logging:
				print(
					f"[INFO] Scene root override applied: {spec.prim_path} "
					f"xyz={tuple(spec.translation)} orient_xyz_deg={tuple(spec.rotation_xyz)} "
					f"scale={tuple(spec.scale)}",
					flush=True,
				)

	def seed_scene_root_baseline_from_config(
		self, prim_path: str = DEFAULT_SCENE_ROOT_PRIM
	) -> None:
		"""Use task_registry scene_root_specs as DR baseline (no stage read)."""
		spec = self._scene_root_spec_for_prim(prim_path)
		if spec is None:
			return
		orient_quat = self._editor_orient_xyz_to_quatd(spec.rotation_xyz)
		self._scene_root_baseline = {
			"translate": tuple(spec.translation),
			"orient_wxyz": (
				float(orient_quat.GetReal()),
				float(orient_quat.GetImaginary()[0]),
				float(orient_quat.GetImaginary()[1]),
				float(orient_quat.GetImaginary()[2]),
			),
			"scale": tuple(spec.scale),
		}

	def ensure_scene_root_baseline(self, prim_path: str = DEFAULT_SCENE_ROOT_PRIM) -> None:
		"""Registry baseline when configured; otherwise read from live USD."""
		if self._scene_root_spec_for_prim(prim_path) is not None:
			self.seed_scene_root_baseline_from_config(prim_path)
		else:
			self.capture_scene_root_baseline(prim_path)

	def get_scene_root_translation(self, prim_path: str = DEFAULT_SCENE_ROOT_PRIM) -> tuple[float, float, float]:
		"""Current scene root translate (registry baseline or live capture)."""
		baseline = self._scene_root_baseline.get("translate")
		if baseline is not None:
			return tuple(float(v) for v in baseline)
		spec = self._scene_root_spec_for_prim(prim_path)
		if spec is not None:
			return tuple(spec.translation)
		self.capture_scene_root_baseline(prim_path)
		assert self._scene_root_baseline["translate"] is not None
		return tuple(float(v) for v in self._scene_root_baseline["translate"])

	def get_scene_root_translation_delta(
		self,
		calibration_translation: tuple[float, float, float],
		prim_path: str = DEFAULT_SCENE_ROOT_PRIM,
	) -> tuple[float, float, float]:
		"""Δt = current scene root translation − calibration translation."""
		current = self.get_scene_root_translation(prim_path)
		return tuple(
			float(c) - float(b) for c, b in zip(current, calibration_translation)
		)

	def get_scene_root_rotation_xyz(self, prim_path: str = DEFAULT_SCENE_ROOT_PRIM) -> tuple[float, float, float]:
		"""Current scene root editor rotation_xyz from registry config."""
		spec = self._scene_root_spec_for_prim(prim_path)
		if spec is not None:
			return tuple(float(v) for v in spec.rotation_xyz)
		raise RuntimeError("Scene root rotation_xyz is unavailable without registry scene_root_specs.")

	def get_scene_root_yaw_delta_deg(
		self,
		calibration_rotation_xyz: tuple[float, float, float],
		prim_path: str = DEFAULT_SCENE_ROOT_PRIM,
	) -> float:
		"""Δyaw = current rotation_z − calibration rotation_z (degrees)."""
		current = self.get_scene_root_rotation_xyz(prim_path)
		return float(current[2]) - float(calibration_rotation_xyz[2])

	def capture_scene_root_baseline(self, prim_path: str = DEFAULT_SCENE_ROOT_PRIM) -> None:
		"""Store USD default translate/orient/scale for scene root (Z baseline preserved on apply)."""
		import omni.usd
		from pxr import Gf, UsdGeom

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			raise RuntimeError("USD stage is unavailable for scene baseline capture.")
		prim = stage.GetPrimAtPath(prim_path)
		if not prim or not prim.IsValid():
			raise RuntimeError(f"Scene root prim '{prim_path}' does not exist on stage.")

		xformable = UsdGeom.Xformable(prim)
		translate = (0.0, 0.0, 0.0)
		scale = (1.0, 1.0, 1.0)
		orient_wxyz = (1.0, 0.0, 0.0, 0.0)

		for op in xformable.GetOrderedXformOps():
			value = op.Get()
			if value is None:
				continue
			op_type = op.GetOpType()
			if op_type == UsdGeom.XformOp.TypeTranslate:
				translate = tuple(float(v) for v in value)
			elif op_type == UsdGeom.XformOp.TypeScale:
				scale = tuple(float(v) for v in value)
			elif op_type == UsdGeom.XformOp.TypeOrient and hasattr(value, "GetImaginary"):
				im = value.GetImaginary()
				orient_wxyz = (
					float(im[2]),
					float(value.GetReal()),
					float(im[0]),
					float(im[1]),
				)

		self._scene_root_baseline = {
			"translate": translate,
			"orient_wxyz": orient_wxyz,
			"scale": scale,
		}

	def apply_scene_root_xy_yaw_delta(
		self,
		dx: float,
		dy: float,
		yaw_deg: float,
		prim_path: str = DEFAULT_SCENE_ROOT_PRIM,
	) -> None:
		"""Apply relative XY + yaw on captured baseline; Z uses baseline unchanged."""
		import omni.usd
		import isaaclab.sim as sim_utils
		from pxr import Gf, UsdGeom

		baseline = self._scene_root_baseline
		if baseline["translate"] is None:
			self.ensure_scene_root_baseline(prim_path)

		base_t = baseline["translate"]
		base_q = baseline["orient_wxyz"]
		assert base_t is not None and base_q is not None

		stage = omni.usd.get_context().get_stage()
		prim = stage.GetPrimAtPath(prim_path)
		if not prim or not prim.IsValid():
			raise RuntimeError(f"Scene root prim '{prim_path}' does not exist on stage.")

		if not sim_utils.standardize_xform_ops(prim):
			raise RuntimeError(f"Failed to standardize xform ops at '{prim_path}'.")

		xformable = UsdGeom.Xformable(prim)
		translate_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
		orient_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
		if translate_op is None or orient_op is None:
			raise RuntimeError(f"Scene root '{prim_path}' missing translate/orient xform ops.")

		new_translate = Gf.Vec3d(base_t[0] + dx, base_t[1] + dy, base_t[2])
		translate_op.Set(new_translate)

		yaw_rad = math.radians(float(yaw_deg))
		delta_q = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(yaw_rad))
		base_rot = Gf.Rotation(Gf.Quatd(base_q[0], Gf.Vec3d(base_q[1], base_q[2], base_q[3])))
		combined = delta_q * base_rot
		combined_quat = combined.GetQuat()
		orient_op.Set(combined_quat)

	def apply_scene_scale_relative(
		self,
		scale_delta: float,
		prim_path: str = DEFAULT_SCENE_ROOT_PRIM,
	) -> None:
		"""Scale scene root uniformly: baseline_scale * (1 + scale_delta)."""
		import omni.usd
		import isaaclab.sim as sim_utils
		from pxr import Gf, UsdGeom

		if self._scene_root_baseline["scale"] is None:
			self.ensure_scene_root_baseline(prim_path)

		base_s = self._scene_root_baseline["scale"]
		assert base_s is not None
		factor = max(0.05, 1.0 + float(scale_delta))
		new_scale = tuple(float(v) * factor for v in base_s)

		stage = omni.usd.get_context().get_stage()
		prim = stage.GetPrimAtPath(prim_path)
		if not prim or not prim.IsValid():
			raise RuntimeError(f"Scene root prim '{prim_path}' does not exist on stage.")
		if not sim_utils.standardize_xform_ops(prim):
			raise RuntimeError(f"Failed to standardize xform ops at '{prim_path}'.")

		xformable = UsdGeom.Xformable(prim)
		scale_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeScale)
		if scale_op is None:
			scale_op = xformable.AddScaleOp()
		scale_op.Set(Gf.Vec3d(*new_scale))

	def set_scene_joint_initial_deg(self, joint_prim: str, angle_deg: float) -> None:
		"""Override one scene hinge initial angle (deg) in cfg, then push safely."""
		for spec in self.cfg.joint_initial_specs:
			if spec.prim_path == joint_prim:
				object.__setattr__(spec, "position", float(angle_deg))
				break
		else:
			raise KeyError(f"joint_prim '{joint_prim}' not in joint_initial_specs.")
		if self.sim is None:
			self.apply_joint_initial_overrides()
			return
		self.reset_scene_joint_initials_via_tensor()

	def apply_camera_main_jitter(
		self,
		translation_std: float = 0.0,
		rotation_std_deg: float = 0.0,
		camera_name: str = "main",
	) -> None:
		"""Randomize main camera pose around TaskCameraSpec baseline."""
		import random

		if translation_std <= 0.0 and rotation_std_deg <= 0.0:
			return

		tx = random.gauss(0.0, translation_std) if translation_std > 0 else 0.0
		ty = random.gauss(0.0, translation_std) if translation_std > 0 else 0.0
		tz = random.gauss(0.0, translation_std) if translation_std > 0 else 0.0
		self.refresh_camera_prim(camera_name, translation_offset=(tx, ty, tz))

		if rotation_std_deg > 0.0 and camera_name in self.sensor_cameras:
			# Sensor re-init after prim move is handled by caller via reset if needed.
			pass

	def _joint_initial_target_deg(self, joint_prim: str) -> float | None:
		for spec in self.cfg.joint_initial_specs:
			if spec.prim_path == joint_prim:
				return float(spec.position)
		return None

	def _hinge_world_from_link_bind_pose(
		self,
		link_pos: tuple[float, float, float],
		link_quat_wxyz: tuple[float, float, float, float],
		hinge_origin_link: tuple[float, float, float],
		hinge_axis_link: tuple[float, float, float],
	) -> tuple[Any, Any]:
		import numpy as np
		from scipy.spatial.transform import Rotation as R

		w, x, y, z = link_quat_wxyz
		rot = R.from_quat([x, y, z, w])
		origin_w = rot.apply(np.asarray(hinge_origin_link, dtype=np.float64)) + np.asarray(link_pos)
		axis_w = rot.apply(np.asarray(hinge_axis_link, dtype=np.float64))
		return origin_w, axis_w

	def _read_scene_articulation_body_pose_wxyz(
		self,
		link_prim: str,
	) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
		if self.scene_articulation is None:
			return None
		link_name = link_prim.rsplit("/", 1)[-1]
		try:
			body_ids, _ = self.scene_articulation.find_bodies(link_name)
			if not body_ids:
				return None
			body_id = int(body_ids[0])
			pos = self.scene_articulation.data.body_pos_w[0, body_id].detach().cpu().numpy()
			quat = tuple(
				float(v) for v in self.scene_articulation.data.body_quat_w[0, body_id].tolist()
			)
			return (
				(float(pos[0]), float(pos[1]), float(pos[2])),
				quat,
			)
		except Exception:
			return None

	def get_movable_link_world_pose_wxyz(
		self,
		link_prim: str,
		joint_prim: str,
		*,
		hinge_origin_link: tuple[float, float, float] = (0.0, 0.0, 0.0),
		hinge_axis_link: tuple[float, float, float] = (1.0, 0.0, 0.0),
		bind_joint_deg: float = 0.0,
	) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
		"""Return link pose with live hinge angle applied.

		USD link prim xform is authored at URDF rest (joint=0). task_registry
		joint_initial_specs (e.g. 15° for close_laptop_lid) must be applied
		before mapping link-local handle offsets to world coordinates.
		"""
		import numpy as np
		from reference.opening_kinematics import link_pose_at_delta_angle

		body_pose = self._read_scene_articulation_body_pose_wxyz(link_prim)
		if body_pose is not None:
			return body_pose

		joint_deg = self.read_scene_joint_angle_deg(joint_prim)
		if joint_deg is None:
			joint_deg = self._joint_initial_target_deg(joint_prim)

		link_pos, link_quat = self.get_prim_world_pose_wxyz(link_prim)
		if joint_deg is None or abs(float(joint_deg) - float(bind_joint_deg)) < 1e-6:
			return link_pos, link_quat

		delta_rad = math.radians(float(joint_deg) - float(bind_joint_deg))
		hinge_origin_w, hinge_axis_w = self._hinge_world_from_link_bind_pose(
			link_pos,
			link_quat,
			hinge_origin_link,
			hinge_axis_link,
		)
		link_pos_np, link_quat_out = link_pose_at_delta_angle(
			np.asarray(link_pos, dtype=np.float64),
			link_quat,
			hinge_origin_w,
			hinge_axis_w,
			delta_rad,
		)
		return (float(link_pos_np[0]), float(link_pos_np[1]), float(link_pos_np[2])), link_quat_out

	def get_prim_world_pose_wxyz(
		self,
		prim_path: str,
	) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
		"""Return (position_xyz, quat_wxyz) for a stage prim in world frame."""
		import omni.usd
		from pxr import Gf, UsdGeom

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			raise RuntimeError("USD stage unavailable.")
		prim = stage.GetPrimAtPath(prim_path)
		if not prim or not prim.IsValid():
			raise RuntimeError(f"Prim '{prim_path}' does not exist on stage.")

		xformable = UsdGeom.Xformable(prim)
		world_xf = xformable.ComputeLocalToWorldTransform(0.0)
		translation = world_xf.ExtractTranslation()
		rotation = world_xf.ExtractRotationQuat()
		imag = rotation.GetImaginary()
		pos = (float(translation[0]), float(translation[1]), float(translation[2]))
		quat = (
			float(rotation.GetReal()),
			float(imag[0]),
			float(imag[1]),
			float(imag[2]),
		)
		return pos, quat

	def get_hinge_world_frame(
		self,
		link_prim: str,
		hinge_origin_link: tuple[float, float, float],
		hinge_axis_link: tuple[float, float, float],
	) -> tuple[Any, Any]:
		"""Map hinge origin/axis from link local frame to world frame."""
		link_pos, link_quat_wxyz = self.get_prim_world_pose_wxyz(link_prim)
		return self._hinge_world_from_link_bind_pose(
			link_pos,
			link_quat_wxyz,
			hinge_origin_link,
			hinge_axis_link,
		)


