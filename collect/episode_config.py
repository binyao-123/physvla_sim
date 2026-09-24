"""Serialize and restore the complete initial configuration of one episode.

An episode's initial state comes from two independent random streams:

  * object pose / articulated joint initial -- sampled by ``task_registry``'s
    ``get_task_preset()`` through the *global* ``random`` module;
  * RobotWin2 visual DR (camera, lighting, environment USD, clutter) --
    sampled by ``sample_randomization()`` through a ``random.Random`` instance.

Both are derived here from one run seed plus the episode index, using separate
sub-streams so pose and DR draws stay uncorrelated. Because every episode is
seeded from scratch, episode N no longer depends on how many random draws
episodes 0..N-1 happened to make (retries, early termination, etc.).

The resulting ``EpisodeConfig`` is written next to the rollout artifacts before
the episode runs, so a crashed episode still leaves a reproducible record.
``physvla_sim/rollout_sim/prts_policy_rollout.py --replay_episode_config`` reads
it back and re-applies it instead of sampling fresh.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from domain_randomization_robotwin2 import (
    RobotWin2ClutterPlacement,
    RobotWin2RandomizationConfig,
    RobotWin2RandomizationSample,
)
from task_registry import TaskJointInitialSpec, TaskSceneRootSpec

EPISODE_CONFIG_SCHEMA_VERSION = 1
EPISODE_CONFIG_DIRNAME = "episode_configs"

# Sub-stream labels; changing these changes every derived seed.
POSE_STREAM = "pose"
DR_STREAM = "dr"
# Base seed for the policy's own sampling noise (PRTS flow matching starts from
# Gaussian noise, so an unseeded server returns a different chunk every call).
POLICY_STREAM = "policy"
# Per-inference sub-stream derived from the episode's policy seed.
INFER_STREAM = "infer"
# PLD probing length, so an episode's probe prefix replays with its scene.
PROBE_STREAM = "probe"


def derive_episode_seed(run_seed: int, episode_index: int, stream: str) -> int:
    """Stable per-episode seed. Not Python's ``hash()``: that is salted per process."""

    payload = f"{int(run_seed)}:{int(episode_index)}:{stream}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF


def _to_tuples(value: Any) -> Any:
    """JSON round-trips tuples into lists; put them back."""

    if isinstance(value, list):
        return tuple(_to_tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_tuples(v) for k, v in value.items()}
    return value


def _drop_unknown(cls: type, data: dict[str, Any], label: str) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        print(
            f"[episode_config] ignoring unknown {label} keys from saved config: "
            f"{sorted(unknown)}",
            flush=True,
        )
    return {k: v for k, v in data.items() if k in known}


@dataclass
class EpisodeConfig:
    """Everything needed to rebuild one episode's initial scene."""

    schema_version: int
    task_id: str
    episode_index: int
    run_seed: int
    pose_seed: int
    dr_seed: int
    policy_seed: int
    fixed_task_pose: bool
    usd_path: str
    scene_root_specs: list[dict[str, Any]]
    joint_initial_specs: list[dict[str, Any]]
    randomization_sample: dict[str, Any]
    randomization_config: dict[str, Any]
    # Set when this episode was replayed rather than sampled; ``run_seed`` then
    # carries the seed of the run that originally produced the scene.
    replayed_from: str | None = None


def build_episode_config(
    *,
    task_id: str,
    episode_index: int,
    run_seed: int,
    pose_seed: int,
    dr_seed: int,
    policy_seed: int,
    fixed_task_pose: bool,
    episode_preset: Any,
    randomization_sample: RobotWin2RandomizationSample,
    randomization_config: RobotWin2RandomizationConfig,
    replayed_from: str | None = None,
) -> EpisodeConfig:
    return EpisodeConfig(
        schema_version=EPISODE_CONFIG_SCHEMA_VERSION,
        task_id=task_id,
        episode_index=int(episode_index),
        run_seed=int(run_seed),
        pose_seed=int(pose_seed),
        dr_seed=int(dr_seed),
        policy_seed=int(policy_seed),
        fixed_task_pose=bool(fixed_task_pose),
        usd_path=str(Path(episode_preset.usd_path).resolve()),
        scene_root_specs=[asdict(spec) for spec in episode_preset.scene_root_specs],
        joint_initial_specs=[asdict(spec) for spec in episode_preset.joint_initial_specs],
        randomization_sample=asdict(randomization_sample),
        randomization_config=asdict(randomization_config),
        replayed_from=replayed_from,
    )


def write_episode_config(config_dir: Path, config: EpisodeConfig) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"episode_{config.episode_index:04d}.json"
    path.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_episode_config(path: Path) -> EpisodeConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(data.get("schema_version", -1))
    if version != EPISODE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {version}, this build expects "
            f"{EPISODE_CONFIG_SCHEMA_VERSION}"
        )
    return EpisodeConfig(**_drop_unknown(EpisodeConfig, data, "EpisodeConfig"))


def load_episode_configs(path: str | Path) -> list[EpisodeConfig]:
    """Accept one json file, or a directory / rollout output dir holding many."""

    p = Path(path).expanduser().resolve()
    if p.is_file():
        return [load_episode_config(p)]
    if not p.is_dir():
        raise FileNotFoundError(f"episode config path not found: {p}")
    search = p / EPISODE_CONFIG_DIRNAME if (p / EPISODE_CONFIG_DIRNAME).is_dir() else p
    files = sorted(search.glob("episode_*.json"))
    if not files:
        raise FileNotFoundError(f"no episode_*.json under {search}")
    configs = [load_episode_config(f) for f in files]
    configs.sort(key=lambda c: c.episode_index)
    return configs


def randomization_sample_from_config(config: EpisodeConfig) -> RobotWin2RandomizationSample:
    data = dict(config.randomization_sample)
    placements = data.pop("clutter_placements", ()) or ()
    data = {k: _to_tuples(v) for k, v in data.items()}
    data = _drop_unknown(RobotWin2RandomizationSample, data, "randomization_sample")
    sample = RobotWin2RandomizationSample(**data)
    sample.clutter_placements = tuple(
        RobotWin2ClutterPlacement(
            **_drop_unknown(
                RobotWin2ClutterPlacement,
                {k: _to_tuples(v) for k, v in placement.items()},
                "clutter_placement",
            )
        )
        for placement in placements
    )
    return sample


def randomization_config_from_config(config: EpisodeConfig) -> RobotWin2RandomizationConfig:
    data = {k: _to_tuples(v) for k, v in config.randomization_config.items()}
    data = _drop_unknown(RobotWin2RandomizationConfig, data, "randomization_config")
    return RobotWin2RandomizationConfig(**data)


def episode_preset_from_config(base_preset: Any, config: EpisodeConfig) -> Any:
    """Rebuild the per-episode TaskPreset (pose + joint initial + usd) from a record."""

    if config.task_id != base_preset.task_id:
        raise ValueError(
            f"episode config is for task '{config.task_id}', "
            f"but the run is task '{base_preset.task_id}'"
        )
    scene_root_specs = tuple(
        TaskSceneRootSpec(
            prim_path=str(spec["prim_path"]),
            translation=tuple(float(v) for v in spec["translation"]),
            rotation_xyz=tuple(float(v) for v in spec["rotation_xyz"]),
            scale=tuple(float(v) for v in spec["scale"]),
        )
        for spec in config.scene_root_specs
    )
    joint_initial_specs = tuple(
        TaskJointInitialSpec(
            prim_path=str(spec["prim_path"]),
            position=float(spec["position"]),
            joint_type=str(spec.get("joint_type", "revolute")),
        )
        for spec in config.joint_initial_specs
    )
    return replace(
        base_preset,
        usd_path=config.usd_path,
        scene_root_specs=scene_root_specs,
        joint_initial_specs=joint_initial_specs,
    )
