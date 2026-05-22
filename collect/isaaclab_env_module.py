from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from task_registry import SCENE_ARTICULATION_PRIM_PATH


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
class EnvironmentModuleConfig:
	usd_path: str
	camera_width: int = 400
	camera_height: int = 400
	camera_sensor_type: str = "camera"
	warmup_render_steps: int = 6
	reset_robot_root_pose: bool = False
	camera_specs: list[CameraPrimSpec] = field(default_factory=list)
	joint_drive_specs: list[JointDrivePrimSpec] = field(default_factory=list)
	joint_limit_specs: list[JointLimitPrimSpec] = field(default_factory=list)
	joint_initial_specs: list[JointInitialPrimSpec] = field(default_factory=list)


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

		state_pos_attr = "state:angular:physics:position"
		state_vel_attr = "state:angular:physics:velocity"

		for spec in self.cfg.joint_initial_specs:
			prim = stage.GetPrimAtPath(spec.prim_path)
			if not prim.IsValid():
				raise RuntimeError(f"Joint prim '{spec.prim_path}' does not exist on stage.")

			value = float(spec.position)

			auth_attr = prim.GetAttribute("physics:position")
			if not auth_attr.IsValid():
				auth_attr = prim.CreateAttribute("physics:position", Sdf.ValueTypeNames.Float)
			auth_attr.Set(value)

			sta = prim.GetAttribute(state_pos_attr)
			if sta.IsValid():
				sta.Set(value)

			stv = prim.GetAttribute(state_vel_attr)
			if stv.IsValid():
				stv.Set(0.0)

	def _read_scene_joint_angle_deg_usd(self, prim_path: str) -> float | None:
		import omni.usd

		stage = omni.usd.get_context().get_stage()
		if stage is None:
			return None
		prim = stage.GetPrimAtPath(prim_path)
		if not prim or not prim.IsValid():
			return None
		for attr_name in (
			"state:angular:physics:position",
			"drive:angular:physics:position",
			"physics:position",
		):
			attr = prim.GetAttribute(attr_name)
			if attr and attr.IsValid():
				value = attr.Get()
				if value is not None:
					return float(value)
		return None

	def _articulation_joint_pos_to_deg(self, joint_prim_path: str, raw: float) -> float:
		mode = self._scene_joint_pos_units.get(joint_prim_path)
		if mode == "deg":
			return float(raw)
		if mode == "rad":
			return math.degrees(float(raw))
		usd_deg = self._read_scene_joint_angle_deg_usd(joint_prim_path)
		if usd_deg is not None:
			if abs(raw - usd_deg) <= abs(math.degrees(raw) - usd_deg) + 0.5:
				mode = "deg"
			else:
				mode = "rad"
			self._scene_joint_pos_units[joint_prim_path] = mode
			return float(raw) if mode == "deg" else math.degrees(float(raw))
		return math.degrees(float(raw))

	def read_scene_joint_angle_deg(self, prim_path: str) -> float | None:
		"""Read scene hinge angle (deg) via Isaac Lab Articulation."""
		if self.scene_articulation is None or self.sim is None or not self.sim.is_playing():
			return self._read_scene_joint_angle_deg_usd(prim_path)

		joint_name = prim_path.rsplit("/", 1)[-1]
		if prim_path not in self._scene_joint_articulation_ids:
			joint_ids, _ = self.scene_articulation.find_joints(joint_name)
			if not joint_ids:
				return None
			self._scene_joint_articulation_ids[prim_path] = int(joint_ids[0])

		joint_id = self._scene_joint_articulation_ids[prim_path]
		try:
			raw = self.scene_articulation.data.joint_pos[0, joint_id]
			if hasattr(raw, "item"):
				raw = raw.item()
			return self._articulation_joint_pos_to_deg(prim_path, float(raw))
		except Exception as exc:
			print(f"[WARN] Scene joint read failed for {prim_path}: {exc}")
			return None

	def create_simulation(self, dt: float = 1.0 / 60.0, render_interval: int = 4, use_fabric: bool = True):
		import omni.usd
		import isaaclab.sim as sim_utils

		omni.usd.get_context().open_stage(self.cfg.usd_path)
		self.apply_joint_drive_overrides()
		self.apply_joint_limit_overrides()
		self.apply_joint_initial_overrides()

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

	def sync_scene_joint_initials_to_sim(self) -> None:
		"""Write scene hinge initial angles to PhysX after sim.reset()."""
		if self.scene_articulation is None or not self.cfg.joint_initial_specs:
			return

		import torch

		device = self.device or "cuda:0"
		scene_prefix = f"{SCENE_ARTICULATION_PRIM_PATH}/"
		for spec in self.cfg.joint_initial_specs:
			if not spec.prim_path.startswith(scene_prefix):
				continue
			joint_name = spec.prim_path.rsplit("/", 1)[-1]
			joint_ids, _ = self.scene_articulation.find_joints(joint_name)
			if not joint_ids:
				continue
			pos = torch.tensor(
				[[math.radians(float(spec.position))]],
				device=device,
				dtype=torch.float32,
			)
			vel = torch.zeros((1, len(joint_ids)), device=device, dtype=torch.float32)
			self.scene_articulation.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids)
		self.scene_articulation.write_data_to_sim()

	def initialize_robot_home_pose(self):
		if self.sim is None or self.robot is None:
			raise RuntimeError("Simulation and robot must be created before home pose init.")

		self.sim.reset()
		self.robot.update(self.sim.cfg.dt)
		if self.scene_articulation is not None:
			self.scene_articulation.reset()
			self.scene_articulation.update(self.sim.cfg.dt)
		self.apply_joint_initial_overrides()
		self.sync_scene_joint_initials_to_sim()

		if self._should_reset_root_pose():
			self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
		self.robot.write_joint_state_to_sim(
			self.robot.data.default_joint_pos,
			self.robot.data.default_joint_vel,
		)

		self.sim.step(render=True)
		self.robot.update(self.sim.cfg.dt)
		if self.scene_articulation is not None:
			self.scene_articulation.update(self.sim.cfg.dt)

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
	):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before camera prim refresh.")

		import omni.usd
		import isaaclab.sim as sim_utils
		from pxr import UsdGeom

		stage = omni.usd.get_context().get_stage()

		for spec in self.cfg.camera_specs:
			if spec.name == camera_name:
				translation = spec.translation
				if translation_offset is not None:
					if spec.translation is None:
						raise ValueError(
							"Camera translation_offset cannot be applied when translation is None."
						)
					if len(translation_offset) != 3:
						raise ValueError("Camera translation_offset must contain exactly 3 values.")
					translation = tuple(
						float(spec.translation[index]) + float(translation_offset[index])
						for index in range(3)
					)

				cam = UsdGeom.Camera.Define(stage, spec.prim_path)
				self._set_camera_transform(cam, translation, spec.rotation_xyz)
				if spec.focal_length is not None:
					cam.GetFocalLengthAttr().Set(spec.focal_length)

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
			if self._should_reset_root_pose():
				self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
			self.robot.write_joint_state_to_sim(
				self.robot.data.default_joint_pos,
				self.robot.data.default_joint_vel,
			)
		if self.scene_articulation is not None:
			self.scene_articulation.reset()
			self.scene_articulation.update(self.sim.cfg.dt)
		self.apply_joint_initial_overrides()
		self.sync_scene_joint_initials_to_sim()
		self.sim.step(render=True)
		if self.robot is not None:
			self.robot.update(self.sim.cfg.dt)
		if self.scene_articulation is not None:
			self.scene_articulation.update(self.sim.cfg.dt)

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

	@property
	def camera_paths(self) -> dict[str, str]:
		return dict(self._camera_paths)

