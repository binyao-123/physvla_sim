"""RobotWin2-style visual / environment domain randomization for trajectory collection.

Physical pose randomization (hinge angle, scene root XY / yaw / scale) lives in
``task_registry`` and is applied via ``get_task_preset()`` before ``sim.reset()``.

This module handles non-pose randomization: camera jitter, lighting, and
environment USD scene selection (RobotWin2-style DR).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_env_module import IsaacLabEnvironmentModule
    from task_registry import TaskPreset


@dataclass
class RobotWin2RandomizationConfig:
    seed: int | None = None
    camera_main_enable: bool = False
    camera_translation_std: float = 0.02
    camera_rotation_std_deg: float = 3.0
    camera_main_translation_ranges: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None = None
    camera_main_focal_length_range: tuple[float, float] | None = None
    lighting_enable: bool = False
    lighting_mode: str = "stage"
    lighting_mode_candidates: tuple[str, ...] = ()
    lighting_rig_name: str = "Default"
    lighting_rig_candidates: tuple[str, ...] = ()
    lighting_auto_light_rig_on_startup: bool = True
    lighting_import_rig_to_stage: bool = False
    lighting_prim_paths: tuple[str, ...] = ()
    lighting_intensity_scale_range: tuple[float, float] = (0.7, 1.3)
    lighting_exposure_delta_range: tuple[float, float] = (-0.5, 0.5)
    lighting_color_temperature_range: tuple[float, float] = (4500.0, 7500.0)
    lighting_enable_color_temperature: bool = True
    environment_enable: bool = False
    environment_prim_path: str = "/World/RobotWin2Environment"
    environment_ground_plane_prim_paths: tuple[str, ...] = ()
    environment_usd_scale: float = 1.0
    environment_default_asset: tuple[str, str | None] | None = None
    environment_asset_candidates: tuple[tuple[str, str | None], ...] = ()


@dataclass
class RobotWin2RandomizationSample:
    camera_translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_translation: tuple[float, float, float] | None = None
    camera_rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_focal_length: float | None = None
    lighting_intensity_scale: float = 1.0
    lighting_exposure_delta: float = 0.0
    lighting_color_temperature: float | None = None
    lighting_mode: str | None = None
    lighting_rig_name: str | None = None
    environment_name: str | None = None
    environment_usd_path: str | None = None


# Backward-compatible aliases for existing imports.
RandomizationConfig = RobotWin2RandomizationConfig
RandomizationSample = RobotWin2RandomizationSample


def randomization_config_from_registry(task_preset: TaskPreset) -> RobotWin2RandomizationConfig:
    spec = task_preset.randomization
    return RobotWin2RandomizationConfig(
        camera_main_enable=spec.camera_main_enable,
        camera_translation_std=float(spec.camera_translation_std),
        camera_rotation_std_deg=float(spec.camera_rotation_std_deg),
        camera_main_translation_ranges=spec.camera_main_translation_ranges,
        camera_main_focal_length_range=spec.camera_main_focal_length_range,
        lighting_enable=bool(spec.lighting_enable),
        lighting_mode=str(spec.lighting_mode),
        lighting_mode_candidates=tuple(spec.lighting_mode_candidates),
        lighting_rig_name=str(spec.lighting_rig_name),
        lighting_rig_candidates=tuple(spec.lighting_rig_candidates),
        lighting_auto_light_rig_on_startup=bool(spec.lighting_auto_light_rig_on_startup),
        lighting_import_rig_to_stage=bool(spec.lighting_import_rig_to_stage),
        lighting_prim_paths=tuple(spec.lighting_prim_paths),
        lighting_intensity_scale_range=tuple(spec.lighting_intensity_scale_range),
        lighting_exposure_delta_range=tuple(spec.lighting_exposure_delta_range),
        lighting_color_temperature_range=tuple(spec.lighting_color_temperature_range),
        lighting_enable_color_temperature=bool(spec.lighting_enable_color_temperature),
        environment_enable=bool(spec.environment_enable),
        environment_prim_path=str(spec.environment_prim_path),
        environment_ground_plane_prim_paths=tuple(spec.environment_ground_plane_prim_paths),
        environment_usd_scale=float(spec.environment_usd_scale),
        environment_default_asset=spec.environment_default_asset,
        environment_asset_candidates=tuple(spec.environment_asset_candidates),
    )


def add_randomization_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("domain randomization (RobotWin2 visual)")
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible randomization.",
    )
    group.add_argument(
        "--rand_camera_main",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: jitter main camera translation.",
    )
    group.add_argument(
        "--rand_lighting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: randomize Isaac USD light intensity/exposure/color temperature.",
    )
    group.add_argument(
        "--lighting_mode",
        choices=("stage", "off", "camera", "rig"),
        default=None,
        help="Override registry: Isaac viewport lighting menu mode.",
    )
    group.add_argument(
        "--lighting_rig",
        choices=("Colored Lights", "Default", "Grey Studio"),
        default=None,
        help="Override registry: Isaac built-in light rig when --lighting_mode rig.",
    )
    group.add_argument(
        "--rand_environment_scene",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: randomize Isaac environment USD scene reference.",
    )


def randomization_config_from_args(
    args: Any,
    task_preset: TaskPreset,
) -> RobotWin2RandomizationConfig:
    config = randomization_config_from_registry(task_preset)
    if getattr(args, "seed", None) is not None:
        config = replace(config, seed=int(args.seed))
    if getattr(args, "rand_camera_main", None) is not None:
        config = replace(config, camera_main_enable=bool(args.rand_camera_main))
    if getattr(args, "rand_lighting", None) is not None:
        config = replace(config, lighting_enable=bool(args.rand_lighting))
    if getattr(args, "lighting_mode", None) is not None:
        config = replace(config, lighting_mode=str(args.lighting_mode))
    if getattr(args, "lighting_rig", None) is not None:
        config = replace(config, lighting_rig_name=str(args.lighting_rig))
    if getattr(args, "rand_environment_scene", None) is not None:
        config = replace(config, environment_enable=bool(args.rand_environment_scene))
    return config


def _uniform(rng: Any, low: float, high: float) -> float:
    if hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    return float(random.uniform(low, high))


def _gauss(rng: Any, mean: float, std: float) -> float:
    if hasattr(rng, "gauss"):
        return float(rng.gauss(mean, std))
    if hasattr(rng, "normal"):
        return float(rng.normal(mean, std))
    return float(random.gauss(mean, std))


def _choice(rng: Any, values: tuple[str, ...]) -> str:
    if not values:
        raise ValueError("Cannot choose from an empty tuple.")
    if hasattr(rng, "choice"):
        return str(rng.choice(values))
    return str(random.choice(values))


def _choice_any(rng: Any, values: tuple[Any, ...]) -> Any:
    if not values:
        raise ValueError("Cannot choose from an empty tuple.")
    if hasattr(rng, "integers"):
        return values[int(rng.integers(0, len(values)))]
    if hasattr(rng, "randrange"):
        return values[int(rng.randrange(len(values)))]
    return random.choice(values)


def sample_randomization(
    config: RobotWin2RandomizationConfig,
    rng: random.Random | Any,
) -> RobotWin2RandomizationSample:
    sample = RobotWin2RandomizationSample()
    if config.camera_main_enable and config.camera_main_translation_ranges is not None:
        sample.camera_translation = tuple(
            _uniform(rng, float(low), float(high))
            for low, high in config.camera_main_translation_ranges
        )
    elif config.camera_main_enable and config.camera_translation_std > 0:
        sample.camera_translation_offset = (
            _gauss(rng, 0.0, config.camera_translation_std),
            _gauss(rng, 0.0, config.camera_translation_std),
            0.0,
        )
    if config.camera_main_enable and config.camera_main_focal_length_range is not None:
        f_min, f_max = config.camera_main_focal_length_range
        sample.camera_focal_length = _uniform(rng, float(f_min), float(f_max))
    if config.camera_main_enable and config.camera_rotation_std_deg > 0:
        sample.camera_rotation_offset_deg = (
            _gauss(rng, 0.0, config.camera_rotation_std_deg),
            _gauss(rng, 0.0, config.camera_rotation_std_deg),
            _gauss(rng, 0.0, config.camera_rotation_std_deg),
        )
    if config.lighting_enable:
        i_min, i_max = config.lighting_intensity_scale_range
        e_min, e_max = config.lighting_exposure_delta_range
        t_min, t_max = config.lighting_color_temperature_range
        sample.lighting_intensity_scale = _uniform(rng, float(i_min), float(i_max))
        sample.lighting_exposure_delta = _uniform(rng, float(e_min), float(e_max))
        if config.lighting_enable_color_temperature:
            sample.lighting_color_temperature = _uniform(rng, float(t_min), float(t_max))
        sample.lighting_mode = (
            _choice(rng, config.lighting_mode_candidates)
            if config.lighting_mode_candidates
            else config.lighting_mode
        )
        sample.lighting_rig_name = (
            _choice(rng, config.lighting_rig_candidates)
            if config.lighting_rig_candidates
            else config.lighting_rig_name
        )
    if config.environment_default_asset is not None or config.environment_asset_candidates:
        if config.environment_enable and config.environment_asset_candidates:
            env_name, env_path = _choice_any(rng, config.environment_asset_candidates)
        else:
            env_name, env_path = config.environment_default_asset or config.environment_asset_candidates[0]
        sample.environment_name = str(env_name)
        sample.environment_usd_path = str(env_path) if env_path is not None else None
    return sample


_LIGHT_BASELINES: dict[str, dict[str, Any]] = {}
# Baselines parsed from source USD on disk (never from the live stage after overrides).
_ENVIRONMENT_FILE_BASELINES: dict[
    str,
    tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float, float]]],
] = {}


def _is_usd_lux_light(prim: Any) -> bool:
    try:
        from pxr import UsdLux

        light_types = (
            UsdLux.DistantLight,
            UsdLux.SphereLight,
            UsdLux.DiskLight,
            UsdLux.RectLight,
            UsdLux.CylinderLight,
            UsdLux.DomeLight,
        )
        return any(prim.IsA(light_type) for light_type in light_types)
    except Exception:
        type_name = prim.GetTypeName()
        return type_name in {
            "DistantLight",
            "SphereLight",
            "DiskLight",
            "RectLight",
            "CylinderLight",
            "DomeLight",
        }


def _light_prims_from_stage(stage: Any, prim_paths: tuple[str, ...]) -> list[Any]:
    if prim_paths:
        prims = []
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                prims.append(prim)
            else:
                print(f"[WARN] RobotWin2 lighting DR: light prim not found: {prim_path}", flush=True)
        return prims
    return [prim for prim in stage.Traverse() if _is_usd_lux_light(prim)]


def _get_or_create_attr(prim: Any, name: str, value_type: Any) -> Any:
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        attr = prim.CreateAttribute(name, value_type)
    return attr


def _capture_light_baseline(prim: Any) -> dict[str, Any]:
    from pxr import Sdf

    key = str(prim.GetPath())
    if key in _LIGHT_BASELINES:
        return _LIGHT_BASELINES[key]

    def attr_value(name: str, default: Any) -> Any:
        attr = prim.GetAttribute(name)
        if attr.IsValid():
            value = attr.Get()
            if value is not None:
                return value
        return default

    baseline = {
        "intensity": float(attr_value("inputs:intensity", 1.0)),
        "exposure": float(attr_value("inputs:exposure", 0.0)),
        "colorTemperature": float(attr_value("inputs:colorTemperature", 6500.0)),
        "enableColorTemperature": bool(attr_value("inputs:enableColorTemperature", False)),
        "types": {
            "float": Sdf.ValueTypeNames.Float,
            "bool": Sdf.ValueTypeNames.Bool,
        },
    }
    _LIGHT_BASELINES[key] = baseline
    return baseline


def _menu_lighting_mode(mode_value: str | None) -> str:
    mode = (mode_value or "stage").strip().lower()
    if mode not in {"stage", "off", "camera", "rig"}:
        print(f"[WARN] Unknown lighting_mode={mode_value!r}; fallback to 'stage'.", flush=True)
        return "stage"
    return mode


def _apply_viewport_lighting_menu(
    config: RobotWin2RandomizationConfig,
    sample: RobotWin2RandomizationSample,
) -> None:
    """Apply Isaac viewport Stage Lights menu settings when available.

    This uses the same Kit extension behind the UI menu:
    Lights Off / Camera Light / Stage Lights / Light Rigs.
    """
    mode = _menu_lighting_mode(sample.lighting_mode or config.lighting_mode)
    rig_name = sample.lighting_rig_name or config.lighting_rig_name
    rig_or_mode = rig_name if mode == "rig" else mode

    try:
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set(
            "/persistent/exts/omni.kit.viewport.menubar.lighting/autoLightRig/enabled",
            bool(config.lighting_auto_light_rig_on_startup),
        )
    except Exception as exc:
        print(f"[WARN] RobotWin2 lighting menu settings unavailable: {exc}", flush=True)

    try:
        import omni.usd
        from omni.kit.viewport.menubar.lighting.actions import _import_light_rig, _set_lighting_mode

        usd_context = omni.usd.get_context()
        success, applied_mode, previous_mode = _set_lighting_mode(
            rig_or_mode,
            usd_context=usd_context,
        )
        if config.lighting_import_rig_to_stage and mode == "rig" and success:
            _import_light_rig(usd_context_name=usd_context.get_name())
            _set_lighting_mode("stage", usd_context=usd_context)
            applied_mode = "stage(imported rig)"
        print(
            "[INFO] RobotWin2 lighting menu: "
            f"mode={mode} rig={rig_name!r} "
            f"auto_startup={config.lighting_auto_light_rig_on_startup} "
            f"import_rig={config.lighting_import_rig_to_stage} "
            f"success={success} previous={previous_mode!r} applied={applied_mode!r}",
            flush=True,
        )
    except Exception as exc:
        print(
            "[WARN] RobotWin2 lighting menu mode not applied "
            f"(mode={mode}, rig={rig_name!r}): {exc}",
            flush=True,
        )


def _apply_lighting_sample(
    config: RobotWin2RandomizationConfig,
    sample: RobotWin2RandomizationSample,
) -> None:
    if not config.lighting_enable:
        return

    _apply_viewport_lighting_menu(config, sample)

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is unavailable for RobotWin2 lighting randomization.")

    prims = _light_prims_from_stage(stage, config.lighting_prim_paths)
    if not prims:
        print("[WARN] RobotWin2 lighting DR enabled but no UsdLux lights were found.", flush=True)
        return

    applied: list[str] = []
    details: list[str] = []
    for prim in prims:
        baseline = _capture_light_baseline(prim)
        value_types = baseline["types"]
        intensity = baseline["intensity"] * sample.lighting_intensity_scale
        exposure = baseline["exposure"] + sample.lighting_exposure_delta
        _get_or_create_attr(prim, "inputs:intensity", value_types["float"]).Set(float(intensity))
        _get_or_create_attr(prim, "inputs:exposure", value_types["float"]).Set(float(exposure))
        if sample.lighting_color_temperature is not None:
            _get_or_create_attr(prim, "inputs:enableColorTemperature", value_types["bool"]).Set(True)
            _get_or_create_attr(prim, "inputs:colorTemperature", value_types["float"]).Set(
                float(sample.lighting_color_temperature)
            )
        applied.append(str(prim.GetPath()))
        details.append(
            f"{prim.GetPath()} "
            f"intensity {baseline['intensity']:.3g}->{float(intensity):.3g}, "
            f"exposure {baseline['exposure']:.3g}->{float(exposure):.3g}, "
            f"colorTemp {baseline['colorTemperature']:.0f}->"
            f"{sample.lighting_color_temperature if sample.lighting_color_temperature is not None else 'unchanged'}"
        )

    print(
        "[INFO] RobotWin2 lighting DR: "
        f"lights={applied} intensity_scale={sample.lighting_intensity_scale:.3f} "
        f"exposure_delta={sample.lighting_exposure_delta:.3f} "
        f"color_temp={sample.lighting_color_temperature}",
        flush=True,
    )
    for detail in details:
        print(f"[INFO] RobotWin2 lighting DR detail: {detail}", flush=True)


def _environment_asset_prim_key(prim_path: str, env_root_path: str | None = None) -> str:
    if env_root_path:
        prefix = env_root_path.rstrip("/") + "/"
        if prim_path.startswith(prefix):
            return prim_path[len(prefix) :]
    parts = prim_path.strip("/").split("/")
    if parts and parts[0] in ("World", "Stage"):
        return "/".join(parts[1:])
    return prim_path.strip("/")


def _load_environment_file_baselines(
    usd_path: str,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float, float]]]:
    if usd_path in _ENVIRONMENT_FILE_BASELINES:
        return _ENVIRONMENT_FILE_BASELINES[usd_path]

    from pxr import Usd, UsdGeom, UsdShade

    file_stage = Usd.Stage.Open(usd_path)
    if file_stage is None:
        raise RuntimeError(f"Failed to open environment USD for baselines: {usd_path}")

    texture_baselines: dict[str, tuple[float, float]] = {}
    cube_baselines: dict[str, tuple[float, float, float]] = {}
    for prim in file_stage.Traverse():
        file_path = str(prim.GetPath())
        rel_key = _environment_asset_prim_key(file_path)
        if prim.IsA(UsdShade.Shader):
            texture_attr = prim.GetAttribute("inputs:texture_scale")
            if texture_attr and texture_attr.HasAuthoredValue():
                value = texture_attr.Get()
                if value is not None:
                    texture_baselines[rel_key] = (float(value[0]), float(value[1]))
        if prim.IsA(UsdGeom.Cube) and "Grid_Cubes" in file_path:
            cube_xform = UsdGeom.Xformable(prim)
            for op in cube_xform.GetOrderedXformOps():
                if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                    continue
                value = op.Get()
                if value is None:
                    continue
                cube_baselines[rel_key] = (
                    float(value[0]),
                    float(value[1]),
                    float(value[2]),
                )
                break
    _ENVIRONMENT_FILE_BASELINES[usd_path] = (texture_baselines, cube_baselines)
    return texture_baselines, cube_baselines


def _apply_environment_visual_scale(
    env_prim: Any,
    env_root_path: str,
    usd_path: str,
    usd_scale: float,
) -> tuple[int, int]:
    """Shrink grid visuals from on-disk USD baselines (idempotent every episode reset)."""
    from pxr import Gf, Usd, UsdGeom, UsdShade

    if not env_prim or not env_prim.IsValid():
        return 0, 0

    texture_baselines, cube_baselines = _load_environment_file_baselines(usd_path)
    texture_uv_multiplier = 1.0 / usd_scale
    texture_count = 0
    cube_count = 0
    for prim in Usd.PrimRange(env_prim):
        rel_key = _environment_asset_prim_key(str(prim.GetPath()), env_root_path)
        if prim.IsA(UsdShade.Shader) and rel_key in texture_baselines:
            texture_attr = prim.GetAttribute("inputs:texture_scale")
            if texture_attr and texture_attr.IsValid():
                base = texture_baselines[rel_key]
                texture_attr.Set(
                    Gf.Vec2f(
                        base[0] * texture_uv_multiplier,
                        base[1] * texture_uv_multiplier,
                    )
                )
                texture_count += 1
        if prim.IsA(UsdGeom.Cube) and rel_key in cube_baselines:
            cube_xform = UsdGeom.Xformable(prim)
            for op in cube_xform.GetOrderedXformOps():
                if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                    continue
                base = cube_baselines[rel_key]
                op.Set(
                    Gf.Vec3d(
                        base[0] * usd_scale,
                        base[1] * usd_scale,
                        base[2] * usd_scale,
                    )
                )
                cube_count += 1
                break
    return texture_count, cube_count


def _apply_environment_scene_sample(
    config: RobotWin2RandomizationConfig,
    sample: RobotWin2RandomizationSample,
) -> None:
    import omni.usd
    from pxr import Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is unavailable for RobotWin2 environment scene DR.")

    prim = stage.OverridePrim(config.environment_prim_path)
    use_ground_plane = sample.environment_usd_path is None
    prim.GetReferences().SetReferences(
        [] if use_ground_plane else [Sdf.Reference(sample.environment_usd_path)]
    )

    env_scale = 1.0 if use_ground_plane else float(config.environment_usd_scale)
    texture_count = 0
    cube_count = 0
    if not use_ground_plane and env_scale != 1.0 and sample.environment_usd_path:
        texture_count, cube_count = _apply_environment_visual_scale(
            prim,
            config.environment_prim_path,
            sample.environment_usd_path,
            env_scale,
        )

    ground_plane_visible = use_ground_plane
    touched_ground_planes: list[str] = []
    for prim_path in config.environment_ground_plane_prim_paths:
        ground_prim = stage.GetPrimAtPath(prim_path)
        if not ground_prim or not ground_prim.IsValid():
            continue
        imageable = UsdGeom.Imageable(ground_prim)
        imageable.GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if ground_plane_visible else UsdGeom.Tokens.invisible
        )
        touched_ground_planes.append(prim_path)

    if not touched_ground_planes:
        print(
            "[WARN] RobotWin2 environment scene DR: no configured ground plane prim was found; "
            f"candidates={config.environment_ground_plane_prim_paths}",
            flush=True,
        )

    print(
        "[INFO] RobotWin2 environment scene DR: "
        f"enabled={config.environment_enable} name={sample.environment_name!r} "
        f"prim={config.environment_prim_path} usd={sample.environment_usd_path} "
        f"ground_plane_visible={ground_plane_visible} ground_planes={touched_ground_planes} "
        f"usd_scale={env_scale} texture_overrides={texture_count} "
        f"grid_cube_overrides={cube_count}",
        flush=True,
    )


def apply_randomization_sample(
    env_module: IsaacLabEnvironmentModule,
    config: RobotWin2RandomizationConfig,
    sample: RobotWin2RandomizationSample,
) -> None:
    if config.camera_main_enable:
        if sample.camera_translation is not None or sample.camera_focal_length is not None:
            env_module.refresh_camera_prim(
                "main",
                translation_override=sample.camera_translation,
                focal_length_override=sample.camera_focal_length,
            )
        else:
            tx, ty, tz = sample.camera_translation_offset
            if tx or ty or tz:
                env_module.refresh_camera_prim("main", translation_offset=(tx, ty, tz))
        rx, ry, rz = sample.camera_rotation_offset_deg
        if config.camera_rotation_std_deg > 0 and (rx or ry or rz):
            print(
                "[WARN] RobotWin2 camera rotation DR sampled but not applied yet "
                f"(offset_deg={sample.camera_rotation_offset_deg}).",
                flush=True,
            )
    _apply_lighting_sample(config, sample)
    _apply_environment_scene_sample(config, sample)


def format_randomization_sample(sample: RobotWin2RandomizationSample) -> str:
    parts: list[str] = []
    if sample.camera_translation is not None:
        parts.append(f"cam_t_abs={sample.camera_translation}")
    if any(sample.camera_translation_offset):
        parts.append(f"cam_t={sample.camera_translation_offset}")
    if sample.camera_focal_length is not None:
        parts.append(f"cam_f={sample.camera_focal_length:.3f}")
    if any(sample.camera_rotation_offset_deg):
        parts.append(f"cam_r_deg={sample.camera_rotation_offset_deg}")
    if sample.lighting_intensity_scale != 1.0:
        parts.append(f"light_i_scale={sample.lighting_intensity_scale:.3f}")
    if sample.lighting_exposure_delta:
        parts.append(f"light_exp_delta={sample.lighting_exposure_delta:.3f}")
    if sample.lighting_color_temperature is not None:
        parts.append(f"light_temp={sample.lighting_color_temperature:.0f}K")
    if sample.lighting_mode is not None:
        parts.append(f"light_mode={sample.lighting_mode}")
    if sample.lighting_rig_name is not None:
        parts.append(f"light_rig={sample.lighting_rig_name}")
    if sample.environment_name is not None:
        parts.append(f"env_scene={sample.environment_name}")
    return ", ".join(parts) if parts else "none"
