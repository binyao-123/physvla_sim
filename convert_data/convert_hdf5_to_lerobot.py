#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import shutil
import sys
from typing import Literal

_convert_data_dir = Path(__file__).resolve().parent
_physvla_sim_root = _convert_data_dir.parent
for _p in (_convert_data_dir, _physvla_sim_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import h5py
import numpy as np
import torch

import piper_physical_units as _piper_phys

JointValueSource = Literal["isaac_sim_radians", "piper_sdk_can_float"]
GripperObsMode = Literal["binary", "mimic_joint_radians", "sdk_ctrl_scalar"]

JointSlot = Literal["left", "right"]
VectorDim = Literal[7, 14]

GRIPPER_OBS_INDICES_7 = (6,)
GRIPPER_OBS_INDICES_14 = (6, 13)


def _apply_joint_encoding(state: np.ndarray, action: np.ndarray, *, source: JointValueSource) -> tuple[np.ndarray, np.ndarray]:
    """Map HDF5 state/action to SI-aligned floats."""

    if source == "isaac_sim_radians":
        return state, action

    d_f = state.shape[1]
    if d_f == 7:
        return (
            _piper_phys.decode_joint7_vector_if_sdk_floats(state),
            _piper_phys.decode_joint7_vector_if_sdk_floats(action),
        )
    if d_f == 14:
        state_out = np.concatenate(
            [
                _piper_phys.decode_joint7_vector_if_sdk_floats(state[:, :7]),
                _piper_phys.decode_joint7_vector_if_sdk_floats(state[:, 7:]),
            ],
            axis=1,
        )
        action_out = np.concatenate(
            [
                _piper_phys.decode_joint7_vector_if_sdk_floats(action[:, :7]),
                _piper_phys.decode_joint7_vector_if_sdk_floats(action[:, 7:]),
            ],
            axis=1,
        )
        return state_out, action_out

    raise ValueError(f"--joint-value-source piper_sdk_can_float only supports HDF5 widths 7 or 14; got state dim {d_f}")


def _gripper_indices_for_width(width: int) -> tuple[int, ...]:
    if width == 7:
        return GRIPPER_OBS_INDICES_7
    if width == 14:
        return GRIPPER_OBS_INDICES_14
    raise ValueError(f"unsupported joint width for gripper obs: {width}")


def _apply_gripper_obs_mode(
    state: np.ndarray, action: np.ndarray, *, mode: GripperObsMode
) -> np.ndarray:
    """Align observation.state gripper dims with training / real-robot 0/1 semantics."""

    if mode == "mimic_joint_radians":
        return state

    state = state.copy()
    width = state.shape[1]
    ctrl_max = _piper_phys.GRIPPER_CTRL_SCALAR_MAX

    for gi in _gripper_indices_for_width(width):
        col = state[:, gi].astype(np.float64)
        act = np.clip(action[:, gi], 0.0, 1.0)

        if mode == "sdk_ctrl_scalar":
            state[:, gi] = _piper_phys.sdk_ctrl_scalar_to_policy_binary(col)
            continue

        unique = np.unique(np.round(col, 4))
        already_binary = (
            col.size > 0
            and np.nanmin(col) >= -0.01
            and np.nanmax(col) <= 1.01
            and set(unique.tolist()).issubset({0.0, 1.0})
        )
        if already_binary:
            state[:, gi] = np.clip(col, 0.0, 1.0).astype(np.float32)
        elif col.size > 0 and np.nanmax(col) <= ctrl_max + 0.02:
            # 真机 record_data / piper_controller：grippers_angle÷1e6，全开≈0.1
            state[:, gi] = _piper_phys.sdk_ctrl_scalar_to_policy_binary(col)
        elif col.size > 0 and np.nanmax(col) > 1.0:
            # 原始 SDK 整数 0…100_000（若 HDF5 直接存 int cast float）
            state[:, gi] = _piper_phys.sdk_gripper_int_to_policy_binary(col)
        else:
            state[:, gi] = act
    return state


def _expand_single_arm_7_to_dual_14(
    state: np.ndarray, action: np.ndarray, *, active_arm: Literal["left", "right"]
) -> tuple[np.ndarray, np.ndarray]:
    """Layout: dims 0–6 left arm (6 joints + gripper), 7–13 right arm. Idle arm zeros."""

    if state.shape[1] != 7 or action.shape[1] != 7:
        raise ValueError(f"Expected (T,7) state/action before dual expansion, got {state.shape}, {action.shape}")

    zeros = np.zeros((state.shape[0], 7), dtype=np.float32)

    if active_arm == "left":
        dual_state = np.concatenate([state, zeros], axis=1)
        dual_action = np.concatenate([action, zeros], axis=1)
    else:
        dual_state = np.concatenate([zeros, state], axis=1)
        dual_action = np.concatenate([zeros, action], axis=1)

    return dual_state, dual_action


def _expected_hdf5_joint_width(vector_dim: VectorDim, *, dual_arm_recorded_in_hdf5: bool) -> int:
    """HDF5 joint width on disk (last dim of robot_joint_pos / actions)."""

    if vector_dim == 7:
        if dual_arm_recorded_in_hdf5:
            raise ValueError("dual_arm_recorded_in_hdf5 incompatible with vector_dim == 7")
        return 7
    if vector_dim == 14:
        return 14 if dual_arm_recorded_in_hdf5 else 7
    raise ValueError(f"unsupported vector_dim: {vector_dim}")


JOINT_NAMES_SINGLE_7 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
JOINT_NAMES_DUAL_14 = [f"left_{n}" for n in JOINT_NAMES_SINGLE_7] + [f"right_{n}" for n in JOINT_NAMES_SINGLE_7]

# HDF5 paths under episode group `obs/` (Isaac Keyboard_collection).
HDF5_CAMERA_KEYS = ("rgb_main", "rgb_wrist")
# LeRobot / downstream policy keys: head, left_wrist, right_wrist (HDF5 只有 rgb_main / rgb_wrist →
# right_wrist 用同分辨率全零图 RGB 全0 占位).
LERO_IMAGE_KEYS = ("head", "left_wrist", "right_wrist")


def _episode_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        suffix = name.removeprefix("demo_")
        if suffix.isdigit():
            return int(suffix), name
    return sys.maxsize, name


def _iter_hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix not in {".hdf5", ".h5"}:
            raise ValueError(f"Input file must be .hdf5 or .h5, got: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted([*input_path.glob("*.hdf5"), *input_path.glob("*.h5")])
    if not files:
        files = sorted([*input_path.rglob("*.hdf5"), *input_path.rglob("*.h5")])
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under: {input_path}")
    return files


def _iter_episode_names(h5_file: h5py.File) -> list[str]:
    if "data" not in h5_file:
        return []
    return sorted(h5_file["data"].keys(), key=_episode_sort_key)


def _read_success(episode: h5py.Group) -> bool | None:
    if "success" in episode.attrs:
        return bool(np.asarray(episode.attrs["success"]).item())
    if "success" in episode:
        return bool(np.asarray(episode["success"][()]).item())
    return None


def _squeeze_batch_axis(array: np.ndarray, *, key: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim >= 2 and value.shape[1] == 1:
        value = np.squeeze(value, axis=1)
    if value.ndim != 2:
        raise ValueError(f"Expected {key} to have shape (T, D) or (T, 1, D), got {value.shape}.")
    return value


def _to_float32_2d(dataset: h5py.Dataset, *, key: str, expected_dim: int) -> np.ndarray:
    value = _squeeze_batch_axis(dataset[:], key=key).astype(np.float32, copy=False)
    if value.shape[-1] != expected_dim:
        raise ValueError(f"Expected {key} last dimension to be {expected_dim}, got {value.shape}.")
    return value


def _to_uint8_rgb(frame: np.ndarray, *, key: str) -> np.ndarray:
    image = np.asarray(frame)
    image = np.squeeze(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected {key} image frame with 2 or 3 dims, got shape {image.shape}.")

    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected {key} image frame with 1, 3, or 4 channels, got shape {image.shape}.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _camera_dataset(episode: h5py.Group, camera_key: str) -> h5py.Dataset:
    path = f"obs/{camera_key}"
    if path not in episode:
        available = sorted(episode["obs"].keys()) if "obs" in episode else []
        raise ValueError(f"Missing camera dataset '{path}'. Available obs keys: {available}")
    value = episode[path]
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"'{path}' is not a dataset.")
    return value


def _vision_fresh_mask(episode: h5py.Group) -> np.ndarray | None:
    """Return per-step bool mask from obs/vision_is_fresh, or None if field is absent."""

    path = "obs/vision_is_fresh"
    if path not in episode:
        return None
    fresh = np.asarray(episode[path][:]).astype(bool).reshape(-1)
    return fresh


def _export_frame_indices(
    episode: h5py.Group,
    num_frames: int,
    *,
    frame_filter: Literal["vision_fresh", "all"],
    file_label: str,
    episode_name: str,
) -> np.ndarray:
    """Indices to export so each LeRobot step has a new image paired with state/action."""

    all_idx = np.arange(num_frames, dtype=np.int64)
    if frame_filter == "all":
        fresh = _vision_fresh_mask(episode)
        if fresh is not None and fresh.shape[0] == num_frames:
            ratio = float(fresh.mean())
            if ratio < 0.95:
                print(
                    f"[WARN] {file_label}:{episode_name} --frame-filter all: "
                    f"only {ratio:.1%} steps have vision_is_fresh=True. "
                    "Prefer --frame-filter vision_fresh and matching --fps (e.g. 10 when vision_hz=10)."
                )
        return all_idx

    fresh = _vision_fresh_mask(episode)
    if fresh is None:
        print(
            f"[WARN] {file_label}:{episode_name} missing obs/vision_is_fresh; "
            "exporting all control steps (--frame-filter vision_fresh unavailable)."
        )
        return all_idx
    if fresh.shape[0] != num_frames:
        raise ValueError(
            f"{file_label}:{episode_name} vision_is_fresh length {fresh.shape[0]} != {num_frames}"
        )
    idx = np.flatnonzero(fresh)
    if idx.size == 0:
        raise ValueError(f"{file_label}:{episode_name} has no vision_is_fresh=True frames.")
    return idx


def _episode_frame_count(episode: h5py.Group, *, expected_joint_dim: int) -> int:
    state = _to_float32_2d(
        episode["obs/robot_joint_pos"], key="obs/robot_joint_pos", expected_dim=expected_joint_dim
    )
    action = _to_float32_2d(episode["actions"], key="actions", expected_dim=expected_joint_dim)
    lengths = [state.shape[0], action.shape[0]]
    lengths.extend(_camera_dataset(episode, camera_key).shape[0] for camera_key in HDF5_CAMERA_KEYS)
    if len(set(lengths)) != 1:
        raise ValueError(f"Episode length mismatch: {dict(zip(('state', 'action', *HDF5_CAMERA_KEYS), lengths, strict=True))}")
    return lengths[0]


def _first_valid_episode(files: Iterable[Path], *, include_failed: bool, expected_joint_dim: int) -> tuple[h5py.File, h5py.Group]:
    for file_path in files:
        h5_file = h5py.File(file_path, "r")
        try:
            for episode_name in _iter_episode_names(h5_file):
                episode = h5_file["data"][episode_name]
                success = _read_success(episode)
                if include_failed or success is not False:
                    _episode_frame_count(episode, expected_joint_dim=expected_joint_dim)
                    return h5_file, episode
        except Exception:
            h5_file.close()
            raise
        h5_file.close()
    raise ValueError("No convertible episodes found. Check success flags or pass --include-failed.")


def _episode_image_hw(episode: h5py.Group) -> tuple[int, int]:
    sample = _to_uint8_rgb(_camera_dataset(episode, HDF5_CAMERA_KEYS[0])[0], key=f"obs/{HDF5_CAMERA_KEYS[0]}")
    return int(sample.shape[0]), int(sample.shape[1])


def _create_dataset(
    repo_id: str,
    *,
    fps: int,
    mode: Literal["video", "image"],
    output_dir: Path | None,
    overwrite: bool,
    resume: bool,
    video_backend: str | None,
    vector_dim: VectorDim,
    image_hw: tuple[int, int],
):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME
    except ImportError as exc:
        raise RuntimeError(
            "Could not import LeRobot. From the LeRobot repo venv run, for example: "
            "cd /home/ubuntu/workspace/lerobot && uv sync && uv pip install h5py 'lerobot[dataset]' && uv run python "
            "/home/ubuntu/workspace/physvla_sim/convert_data/convert_hdf5_to_lerobot.py ..."
        ) from exc

    joint_names = JOINT_NAMES_DUAL_14 if vector_dim == 14 else JOINT_NAMES_SINGLE_7

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": [joint_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": [joint_names],
        },
    }
    image_h, image_w = image_hw
    for camera_key in LERO_IMAGE_KEYS:
        features[f"observation.images.{camera_key}"] = {
            "dtype": mode,
            "shape": (3, image_h, image_w),
            "names": ["channels", "height", "width"],
        }

    dataset_root = output_dir if output_dir is not None else Path(HF_LEROBOT_HOME) / repo_id
    cache_dataset_dir = dataset_root
    if cache_dataset_dir.exists():
        if not overwrite and not resume:
            raise FileExistsError(f"LeRobot cache dataset already exists: {cache_dataset_dir}. Pass --overwrite or --resume.")
        if overwrite and not resume:
            shutil.rmtree(cache_dataset_dir)

    if resume and cache_dataset_dir.exists():
        print(f"[INFO] Resuming: opening existing dataset at {cache_dataset_dir}")
        return LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_root,
            batch_encoding_size=1,
            image_writer_processes=4,
            image_writer_threads=4,
            video_backend=video_backend,
        )

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=dataset_root,
        robot_type="piper",
        features=features,
        use_videos=(mode == "video"),
        tolerance_s=0.0001,
        image_writer_processes=4,
        image_writer_threads=4,
        video_backend=video_backend,
    )


def _populate_dataset(
    dataset,
    files: list[Path],
    *,
    task: str,
    include_failed: bool,
    max_episodes: int | None,
    resume: int,
    joint_value_source: JointValueSource,
    gripper_obs_mode: GripperObsMode,
    vector_dim: VectorDim,
    expected_hdf5_joint_width: int,
    hdf5_seven_in_slot: JointSlot | None,
    frame_filter: Literal["vision_fresh", "all"],
):
    converted = 0
    skipped_failed = 0
    missing_success = 0
    skipped_io_errors = 0
    global_idx = 0

    for file_path in files:
        with h5py.File(file_path, "r") as h5_file:
            for episode_name in _iter_episode_names(h5_file):
                if max_episodes is not None and converted >= max_episodes:
                    return converted, skipped_failed, missing_success, skipped_io_errors

                # Resume: skip already-converted episodes
                if global_idx < resume:
                    global_idx += 1
                    continue
                global_idx += 1

                episode = h5_file["data"][episode_name]
                success = _read_success(episode)
                if success is None:
                    missing_success += 1
                if success is False and not include_failed:
                    skipped_failed += 1
                    continue

                try:
                    state = _to_float32_2d(
                        episode["obs/robot_joint_pos"],
                        key="obs/robot_joint_pos",
                        expected_dim=expected_hdf5_joint_width,
                    )
                    action = _to_float32_2d(episode["actions"], key="actions", expected_dim=expected_hdf5_joint_width)
                except OSError as e:
                    skipped_io_errors += 1
                    print(f"[WARN] I/O error reading {file_path.name}:{episode_name} joints, skipping: {e}")
                    continue
                state, action = _apply_joint_encoding(state, action, source=joint_value_source)
                state = _apply_gripper_obs_mode(state, action, mode=gripper_obs_mode)

                if vector_dim == 14 and expected_hdf5_joint_width == 7:
                    if hdf5_seven_in_slot is None:
                        raise RuntimeError("internal error: pad-to-14 mode requires --hdf5-seven-in-slot")
                    state, action = _expand_single_arm_7_to_dual_14(state, action, active_arm=hdf5_seven_in_slot)
                elif vector_dim == 14 and expected_hdf5_joint_width == 14:
                    if state.shape[1] != 14:
                        raise ValueError(f"After decode, expected (T,14), got state shape {state.shape}")
                elif vector_dim == 7:
                    if state.shape[1] != 7:
                        raise ValueError(f"Expected (T,7) for vector-dim 7, got state shape {state.shape}")

                camera_datasets = {
                    k: _camera_dataset(episode, k) for k in HDF5_CAMERA_KEYS
                }
                lengths = [state.shape[0], action.shape[0], *(d.shape[0] for d in camera_datasets.values())]
                if len(set(lengths)) != 1:
                    raise ValueError(
                        f"{file_path}:{episode_name} length mismatch: "
                        f"{dict(zip(('state', 'action', *HDF5_CAMERA_KEYS), lengths, strict=True))}"
                    )

                export_idx = _export_frame_indices(
                    episode,
                    state.shape[0],
                    frame_filter=frame_filter,
                    file_label=file_path.name,
                    episode_name=episode_name,
                )
                try:
                    head0 = _to_uint8_rgb(
                        camera_datasets["rgb_main"][int(export_idx[0])],
                        key="obs/rgb_main",
                    )
                except OSError as e:
                    skipped_io_errors += 1
                    print(f"[WARN] I/O error reading {file_path.name}:{episode_name} images, skipping: {e}")
                    continue
                h, w = int(head0.shape[0]), int(head0.shape[1])

                try:
                    for frame_idx in export_idx:
                        i = int(frame_idx)
                        frame = {
                            "observation.state": torch.from_numpy(state[i]),
                            "action": torch.from_numpy(action[i]),
                            "task": task,
                            "observation.images.head": _to_uint8_rgb(
                                camera_datasets["rgb_main"][i],
                                key="obs/rgb_main",
                            ),
                            "observation.images.left_wrist": _to_uint8_rgb(
                                camera_datasets["rgb_wrist"][i],
                                key="obs/rgb_wrist",
                            ),
                            "observation.images.right_wrist": np.zeros((h, w, 3), dtype=np.uint8),
                        }
                        dataset.add_frame(frame)
                except OSError as e:
                    skipped_io_errors += 1
                    print(f"[WARN] I/O error mid-episode {file_path.name}:{episode_name}, discarding partial frames: {e}")
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                converted += 1
                print(
                    f"[INFO] Converted {file_path.name}:{episode_name} "
                    f"({export_idx.size}/{state.shape[0]} frames, filter={frame_filter})"
                )

    return converted, skipped_failed, missing_success, skipped_io_errors


def main():
    parser = argparse.ArgumentParser(
        description="Convert PhysVLA IsaacLab HDF5 demos to a LeRobot dataset for OpenPI fine-tuning."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input HDF5 file or directory containing HDF5 files.")
    parser.add_argument("--output", type=Path, default=None, help="Optional final dataset directory.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="LeRobot dataset repo_id written into dataset metadata (e.g. physvla/piper_dual14_close_laptop_lid_sim_r1).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Language conditioning string copied into every frame's `task` field (must match training / inference).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help=(
            "LeRobot dataset FPS; must match exported frame rate. Default 30 matches Keyboard_collection "
            "when control_hz=vision_hz=30. Legacy HDF5 (vision_hz=10): use --fps 10 --frame-filter vision_fresh."
        ),
    )
    parser.add_argument(
        "--frame-filter",
        choices=["vision_fresh", "all"],
        default="all",
        help=(
            "all (default): export every control step — correct when vision_hz=control_hz (each step has a new image). "
            "vision_fresh: subsample to fresh camera steps only (legacy vision_hz=10 HDF5; pair with --fps 10)."
        ),
    )
    parser.add_argument("--mode", choices=["video", "image"], default="video", help="Store images as videos or images.")
    parser.add_argument("--include-failed", action="store_true", help="Convert episodes explicitly marked success=False.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for smoke conversions.")
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Skip the first N episodes (already converted). Use for resuming after a crash.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output/cache dataset before converting.")
    parser.add_argument("--video-backend", default=None, help="Optional LeRobot video backend override.")
    parser.add_argument(
        "--joint-value-source",
        choices=["isaac_sim_radians", "piper_sdk_can_float"],
        default="isaac_sim_radians",
        help=(
            "Isaac Keyboard_collection HDF5 defaults to radians in obs joints and IK radians in actions[:6]; "
            "choose piper_sdk_can_float only when rows store Piper CAN ints as floats "
            "(see .cursor/plans/6_lerobot_pi05微调.plan.md §2.5)."
        ),
    )
    parser.add_argument(
        "--gripper-obs-mode",
        choices=["binary", "mimic_joint_radians", "sdk_ctrl_scalar"],
        default="binary",
        help=(
            "observation.state gripper dim for LeRobot (π0.5 trains on 0/1). "
            "binary: new Isaac HDF5 0/1, or infer from action / ÷1e6 / SDK int. "
            "sdk_ctrl_scalar: force grippers_angle÷1e6 (open≈0.1) → 0/1. "
            "mimic_joint_radians: keep raw Isaac mimic joint angle in obs."
        ),
    )
    parser.add_argument(
        "--vector-dim",
        type=int,
        choices=[7, 14],
        default=7,
        help="LeRobot observation.state / action width: 7 single arm or 14 dual-arm (left 0:7, right 7:14, idle zeros).",
    )
    parser.add_argument(
        "--hdf5-seven-in-slot",
        choices=["left", "right"],
        default=None,
        help="With --vector-dim 14: HDF5 rows are still 7D; place active arm in left (0:7) or right (7:14); idle half zeros.",
    )
    parser.add_argument(
        "--dual-arm-recorded-in-hdf5",
        action="store_true",
        help=(
            "With --vector-dim 14: HDF5 robot_joint_pos and actions are already (T,14) — "
            "left [:7], right [7:], no zero-padding from this converter. Mutually exclusive with --hdf5-seven-in-slot."
        ),
    )
    args = parser.parse_args()

    dual_native = getattr(args, "dual_arm_recorded_in_hdf5", False)

    if args.vector_dim == 7 and dual_native:
        parser.error("--dual-arm-recorded-in-hdf5 only applies with --vector-dim 14")
    if args.vector_dim == 7 and args.hdf5_seven_in_slot is not None:
        parser.error("--hdf5-seven-in-slot is only valid with --vector-dim 14")

    expected_w = 7 if args.vector_dim == 7 else (14 if dual_native else 7)

    if args.vector_dim == 14:
        if dual_native and args.hdf5_seven_in_slot is not None:
            parser.error("use either --dual-arm-recorded-in-hdf5 or --hdf5-seven-in-slot, not both")
        if not dual_native and args.hdf5_seven_in_slot is None:
            parser.error("--vector-dim 14 requires either --hdf5-seven-in-slot left|right (single-arm HDF5) or "
                         "--dual-arm-recorded-in-hdf5 (14D HDF5)")

    files = _iter_hdf5_files(args.input)
    print(f"[INFO] Found {len(files)} HDF5 file(s).")
    print(f"[INFO] frame_filter={args.frame_filter}, fps={args.fps}, gripper_obs_mode={args.gripper_obs_mode}")

    first_h5, first_episode = _first_valid_episode(
        files, include_failed=args.include_failed, expected_joint_dim=expected_w
    )
    image_hw = _episode_image_hw(first_episode)
    first_h5.close()
    print(f"[INFO] image_hw={image_hw[0]}x{image_hw[1]}")

    if args.vector_dim == 14:
        if dual_native:
            print(
                "[INFO] Dual-arm layout: HDF5 native 14D (left [:7], right [7:]) → LeRobot (no zero-padding)."
            )
        else:
            print(
                "[INFO] Dual-arm layout: idx 0–6 = left, 7–13 = right; "
                f"single-arm HDF5 → '--hdf5-seven-in-slot {args.hdf5_seven_in_slot}', idle half zeros."
            )

    dataset = _create_dataset(
        args.repo_id,
        fps=args.fps,
        mode=args.mode,
        output_dir=args.output,
        overwrite=args.overwrite,
        resume=args.resume > 0,
        video_backend=args.video_backend,
        vector_dim=args.vector_dim,
        image_hw=image_hw,
    )
    try:
        converted, skipped_failed, missing_success, skipped_io_errors = _populate_dataset(
            dataset,
            files,
            task=args.task,
            include_failed=args.include_failed,
            max_episodes=args.max_episodes,
            resume=args.resume,
            joint_value_source=args.joint_value_source,
            gripper_obs_mode=args.gripper_obs_mode,
            vector_dim=args.vector_dim,
            expected_hdf5_joint_width=expected_w,
            hdf5_seven_in_slot=None if dual_native or args.vector_dim == 7 else args.hdf5_seven_in_slot,
            frame_filter=args.frame_filter,
        )
        dataset.finalize()
    except Exception:
        writer = getattr(dataset, "writer", None)
        if writer is not None:
            writer.stop_image_writer()
        raise

    print(f"[INFO] LeRobot dataset root: {dataset.root}")
    print(
        "[INFO] Done. "
        f"converted={converted}, skipped_failed={skipped_failed}, "
        f"missing_success_attr={missing_success}, skipped_io_errors={skipped_io_errors}"
    )
    print(
        "[INFO] meta/stats.json (incl. QUANTILES q01/q99) is written during dataset.finalize(). "
        "Use a new --output dir or --overwrite when gripper_obs_mode or data change; "
        "only run augment_dataset_quantile_stats if an old dataset lacks q01/q99."
    )


if __name__ == "__main__":
    main()
