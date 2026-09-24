#!/usr/bin/env python3
"""Offline reproducibility checks for episode initial configuration (no Isaac Sim).

Guards acceptance item 4: the same (run_seed, episode_index) must rebuild the same
scene, independently of what earlier episodes drew, and an EpisodeConfig must survive
a json round trip unchanged.

  python3 rollout_sim/episode_config_selfcheck.py [--task close_laptop_lid]

Run it under a few different PYTHONHASHSEED values too; derived seeds must not move.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

_PHYSVLA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PHYSVLA_ROOT / "collect"))

from domain_randomization_robotwin2 import (  # noqa: E402
    randomization_config_from_registry,
    sample_randomization,
)
from episode_config import (  # noqa: E402
    DR_STREAM,
    POLICY_STREAM,
    POSE_STREAM,
    build_episode_config,
    derive_episode_seed,
    episode_preset_from_config,
    load_episode_configs,
    randomization_config_from_config,
    randomization_sample_from_config,
    write_episode_config,
)
from task_registry import TASK_PRESETS, get_task_preset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="close_laptop_lid")
    parser.add_argument("--run_seed", type=int, default=20260920)
    args = parser.parse_args()

    base = TASK_PRESETS[args.task]
    rand_config = randomization_config_from_registry(base)
    failures: list[str] = []

    def sample_episode(episode_index: int):
        pose_seed = derive_episode_seed(args.run_seed, episode_index, POSE_STREAM)
        dr_seed = derive_episode_seed(args.run_seed, episode_index, DR_STREAM)
        policy_seed = derive_episode_seed(args.run_seed, episode_index, POLICY_STREAM)
        random.seed(pose_seed)
        preset = get_task_preset(args.task)
        sample = sample_randomization(rand_config, random.Random(dr_seed))
        return preset, sample, pose_seed, dr_seed, policy_seed

    # Sub-streams must not collide, or pose would be correlated with DR.
    seeds0 = [derive_episode_seed(args.run_seed, 0, s) for s in (POSE_STREAM, DR_STREAM, POLICY_STREAM)]
    print(f"[seeds] ep0 pose/dr/policy = {seeds0}")
    if len(set(seeds0)) != len(seeds0):
        failures.append("pose / dr / policy sub-streams collide")

    # Same episode index twice -> identical, even with the global stream disturbed.
    p0, s0, pose0, dr0, pol0 = sample_episode(0)
    random.random()
    random.random()
    p0b, s0b, _, _, _ = sample_episode(0)
    if [asdict(x) for x in p0.scene_root_specs] != [asdict(x) for x in p0b.scene_root_specs]:
        failures.append("scene root not reproducible for a fixed episode index")
    if [asdict(x) for x in p0.joint_initial_specs] != [asdict(x) for x in p0b.joint_initial_specs]:
        failures.append("joint initial not reproducible for a fixed episode index")
    if asdict(s0) != asdict(s0b):
        failures.append("DR sample not reproducible for a fixed episode index")
    print(
        f"[ep0] joint={p0.joint_initial_specs[0].position:.6f} "
        f"xyz={tuple(round(v, 6) for v in p0.scene_root_specs[0].translation)}"
    )

    # Episode 1 must differ from 0, and must not depend on preceding draws.
    p1, s1, _, _, _ = sample_episode(1)
    p1b, s1b, _, _, _ = sample_episode(1)
    if asdict(p1.joint_initial_specs[0]) == asdict(p0.joint_initial_specs[0]):
        failures.append("episode 0 and 1 produced the same pose")
    if asdict(s1) != asdict(s1b):
        failures.append("episode 1 depends on how many draws preceded it")
    print(f"[ep1] joint={p1.joint_initial_specs[0].position:.6f}")

    # json round trip must preserve every scene-describing field, tuples included.
    config = build_episode_config(
        task_id=args.task,
        episode_index=0,
        run_seed=args.run_seed,
        pose_seed=pose0,
        dr_seed=dr0,
        policy_seed=pol0,
        fixed_task_pose=False,
        episode_preset=p0,
        randomization_sample=s0,
        randomization_config=rand_config,
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_episode_config(Path(tmp), config)
        (loaded,) = load_episode_configs(tmp)
        restored_sample = randomization_sample_from_config(loaded)
        restored_config = randomization_config_from_config(loaded)
        restored_preset = episode_preset_from_config(base, loaded)
        if asdict(restored_sample) != asdict(s0):
            failures.append("DR sample changed across the json round trip")
        if asdict(restored_config) != asdict(rand_config):
            failures.append("DR config changed across the json round trip")
        if [asdict(x) for x in restored_preset.scene_root_specs] != [
            asdict(x) for x in p0.scene_root_specs
        ]:
            failures.append("scene root changed across the json round trip")
        if [asdict(x) for x in restored_preset.joint_initial_specs] != [
            asdict(x) for x in p0.joint_initial_specs
        ]:
            failures.append("joint initial changed across the json round trip")
        print(
            f"[roundtrip] clutter={len(restored_sample.clutter_placements)} "
            f"env={restored_sample.environment_name} "
            f"clutter_type={type(restored_sample.clutter_placements).__name__}"
        )

    print("")
    if failures:
        print("EPISODE CONFIG SELFCHECK FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("EPISODE CONFIG SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
