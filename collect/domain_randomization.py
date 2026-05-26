"""CLI-driven domain randomization for auto trajectory collection."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_env_module import IsaacLabEnvironmentModule
    from task_registry import TaskPreset


@dataclass
class RandomizationConfig:
    seed: int | None = None

    obj_xy_enable: bool = False
    obj_x_min: float = -0.04
    obj_x_max: float = 0.04
    obj_y_min: float = -0.03
    obj_y_max: float = 0.03

    obj_yaw_enable: bool = False
    obj_yaw_min_deg: float = -12.0
    obj_yaw_max_deg: float = 12.0

    obj_scale_enable: bool = False
    obj_scale_delta: float = 0.3

    joint_initial_enable: bool = False
    joint_initial_min_deg: float = 10.0
    joint_initial_max_deg: float = 30.0
    joint_initial_baseline_deg: float | None = None
    joint_initial_delta_deg: float | None = None
    joint_initial_prim: str | None = None

    camera_main_enable: bool = False
    camera_translation_std: float = 0.02
    camera_rotation_std_deg: float = 3.0


@dataclass
class RandomizationSample:
    dx: float = 0.0
    dy: float = 0.0
    yaw_deg: float = 0.0
    scale_delta: float = 0.0
    joint_initial_deg: float | None = None
    camera_translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def randomization_config_from_registry(task_preset: TaskPreset) -> RandomizationConfig:
    spec = task_preset.randomization
    return RandomizationConfig(
        obj_xy_enable=spec.obj_xy_enable,
        obj_x_min=float(spec.obj_x_range[0]),
        obj_x_max=float(spec.obj_x_range[1]),
        obj_y_min=float(spec.obj_y_range[0]),
        obj_y_max=float(spec.obj_y_range[1]),
        obj_yaw_enable=spec.obj_yaw_enable,
        obj_yaw_min_deg=float(spec.obj_yaw_range_deg[0]),
        obj_yaw_max_deg=float(spec.obj_yaw_range_deg[1]),
        obj_scale_enable=spec.obj_scale_enable,
        obj_scale_delta=float(spec.obj_scale_delta),
        joint_initial_enable=spec.joint_initial_enable,
        joint_initial_delta_deg=float(spec.joint_initial_delta_deg),
        camera_main_enable=spec.camera_main_enable,
        camera_translation_std=float(spec.camera_translation_std),
        camera_rotation_std_deg=float(spec.camera_rotation_std_deg),
    )


def add_randomization_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("domain randomization")
    group.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible randomization.")
    group.add_argument(
        "--rand_obj_xy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: randomize scene root XY offset.",
    )
    group.add_argument(
        "--rand_obj_yaw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: randomize scene root yaw (deg).",
    )
    group.add_argument(
        "--rand_obj_scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: randomize scene uniform scale (±delta).",
    )
    group.add_argument(
        "--rand_joint_initial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: randomize hinge initial angle.",
    )
    group.add_argument(
        "--rand_camera_main",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override registry: jitter main camera translation.",
    )


def randomization_config_from_args(args: Any, task_preset: TaskPreset) -> RandomizationConfig:
    config = randomization_config_from_registry(task_preset)
    config.seed = args.seed

    if args.rand_obj_xy is not None:
        config.obj_xy_enable = bool(args.rand_obj_xy)
    if args.rand_obj_yaw is not None:
        config.obj_yaw_enable = bool(args.rand_obj_yaw)
    if args.rand_obj_scale is not None:
        config.obj_scale_enable = bool(args.rand_obj_scale)
    if args.rand_joint_initial is not None:
        config.joint_initial_enable = bool(args.rand_joint_initial)
    if args.rand_camera_main is not None:
        config.camera_main_enable = bool(args.rand_camera_main)

    return config


def attach_joint_initial_baseline(
    config: RandomizationConfig,
    task_preset: TaskPreset,
    joint_prim: str | None,
) -> RandomizationConfig:
    merged = replace(config)
    merged.joint_initial_prim = joint_prim
    for spec in task_preset.joint_initial_specs:
        if spec.prim_path == joint_prim:
            merged.joint_initial_baseline_deg = float(spec.position)
            break
    return merged


def merge_randomization_defaults(
    config: RandomizationConfig,
    task_defaults: dict[str, Any] | None,
    joint_prim: str | None,
) -> RandomizationConfig:
    """Deprecated: yaml `defaults` block. Prefer task_registry.randomization."""
    if not task_defaults:
        return config

    merged = RandomizationConfig(**{f.name: getattr(config, f.name) for f in fields(RandomizationConfig)})

    if "rand_obj_xy_enable" in task_defaults:
        merged.obj_xy_enable = bool(task_defaults["rand_obj_xy_enable"])
    if "rand_obj_x" in task_defaults:
        merged.obj_x_min, merged.obj_x_max = task_defaults["rand_obj_x"]
    if "rand_obj_y" in task_defaults:
        merged.obj_y_min, merged.obj_y_max = task_defaults["rand_obj_y"]
    if "rand_obj_yaw_enable" in task_defaults:
        merged.obj_yaw_enable = bool(task_defaults["rand_obj_yaw_enable"])
    if "rand_obj_yaw_deg" in task_defaults:
        merged.obj_yaw_min_deg, merged.obj_yaw_max_deg = task_defaults["rand_obj_yaw_deg"]
    if "rand_obj_scale_enable" in task_defaults:
        merged.obj_scale_enable = bool(task_defaults["rand_obj_scale_enable"])
    if "rand_obj_scale_delta" in task_defaults:
        merged.obj_scale_delta = float(task_defaults["rand_obj_scale_delta"])
    if "rand_joint_initial_enable" in task_defaults:
        merged.joint_initial_enable = bool(task_defaults["rand_joint_initial_enable"])
    if "rand_joint_initial_deg" in task_defaults:
        merged.joint_initial_min_deg, merged.joint_initial_max_deg = task_defaults["rand_joint_initial_deg"]
    if "rand_joint_initial_delta_deg" in task_defaults:
        merged.joint_initial_delta_deg = float(task_defaults["rand_joint_initial_delta_deg"])
    if "rand_camera_main_enable" in task_defaults:
        merged.camera_main_enable = bool(task_defaults["rand_camera_main_enable"])
    if "rand_camera_translation_std" in task_defaults:
        merged.camera_translation_std = float(task_defaults["rand_camera_translation_std"])
    if "rand_camera_rotation_std_deg" in task_defaults:
        merged.camera_rotation_std_deg = float(task_defaults["rand_camera_rotation_std_deg"])

    merged.joint_initial_prim = joint_prim
    return merged


def sample_randomization(config: RandomizationConfig, rng: random.Random) -> RandomizationSample:
    sample = RandomizationSample()

    if config.obj_xy_enable:
        sample.dx = rng.uniform(config.obj_x_min, config.obj_x_max)
        sample.dy = rng.uniform(config.obj_y_min, config.obj_y_max)

    if config.obj_yaw_enable:
        sample.yaw_deg = rng.uniform(config.obj_yaw_min_deg, config.obj_yaw_max_deg)

    if config.obj_scale_enable:
        sample.scale_delta = rng.uniform(-config.obj_scale_delta, config.obj_scale_delta)

    if config.joint_initial_enable:
        if (
            config.joint_initial_baseline_deg is not None
            and config.joint_initial_delta_deg is not None
        ):
            base = float(config.joint_initial_baseline_deg)
            delta = float(config.joint_initial_delta_deg)
            sample.joint_initial_deg = rng.uniform(base - delta, base + delta)
        else:
            sample.joint_initial_deg = rng.uniform(
                config.joint_initial_min_deg, config.joint_initial_max_deg
            )

    if config.camera_main_enable and config.camera_translation_std > 0.0:
        sample.camera_translation_offset = (
            rng.gauss(0.0, config.camera_translation_std),
            rng.gauss(0.0, config.camera_translation_std),
            0.0,
        )

    return sample


def apply_randomization_sample(
    env_module: IsaacLabEnvironmentModule,
    config: RandomizationConfig,
    sample: RandomizationSample,
    joint_prim: str | None = None,
) -> None:
    env_module.capture_scene_root_baseline()

    if config.obj_xy_enable or config.obj_yaw_enable:
        env_module.apply_scene_root_xy_yaw_delta(sample.dx, sample.dy, sample.yaw_deg)

    if config.obj_scale_enable:
        env_module.apply_scene_scale_relative(sample.scale_delta)

    if config.joint_initial_enable and sample.joint_initial_deg is not None:
        target_joint = joint_prim or config.joint_initial_prim
        if target_joint is None:
            raise ValueError("joint_initial randomization requires joint_prim in task config.")
        env_module.set_scene_joint_initial_deg(target_joint, sample.joint_initial_deg)

    if config.camera_main_enable:
        tx, ty, tz = sample.camera_translation_offset
        if tx or ty or tz:
            env_module.refresh_camera_prim("main", translation_offset=(tx, ty, tz))


def format_randomization_sample(sample: RandomizationSample) -> str:
    parts = [
        f"dx={sample.dx:.4f}",
        f"dy={sample.dy:.4f}",
        f"yaw={sample.yaw_deg:.2f}°",
        f"scale_delta={sample.scale_delta:.3f}",
    ]
    if sample.joint_initial_deg is not None:
        parts.append(f"joint_init={sample.joint_initial_deg:.2f}°")
    if any(sample.camera_translation_offset):
        parts.append(f"cam_offset={sample.camera_translation_offset}")
    return ", ".join(parts)
