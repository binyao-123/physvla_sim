#!/usr/bin/env python3
from __future__ import annotations

# 离线查看采集到的 HDF5 相机帧。后续可扩展基于 Isaac Sim 的预览模式，
# 在场景中按动作重播轨迹，而不仅仅查看图像。

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np


def _episode_names(h5_file: h5py.File) -> list[str]:
    if "data" not in h5_file:
        return []
    return sorted(h5_file["data"].keys(), key=lambda name: int(name.split("_")[-1]) if name.startswith("demo_") else name)


def _select_episode(h5_file: h5py.File, demo: str | None) -> h5py.Group:
    names = _episode_names(h5_file)
    if not names:
        raise ValueError("No episodes found under /data.")

    if demo is None:
        demo = names[0]
    elif demo.isdigit():
        demo = f"demo_{int(demo)}"

    if demo not in h5_file["data"]:
        raise ValueError(f"Episode '{demo}' not found. Available: {', '.join(names)}")
    return h5_file["data"][demo]


def _timestamp_utc8() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")


def _default_output_h264_path(demo_name: str) -> Path:
    return Path(f"hdf5_camera_replay_{demo_name}_{_timestamp_utc8()}.mp4")


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    frame = np.asarray(array)
    frame = np.squeeze(frame)

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim != 3:
        raise ValueError(f"Expected image frame with 2 or 3 dims, got shape {frame.shape}.")

    if frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    elif frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {frame.shape}.")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _get_camera_dataset(episode: h5py.Group, camera_name: str) -> h5py.Dataset:
    path = f"obs/{camera_name}"
    if path not in episode:
        available = []
        if "obs" in episode:
            available = sorted(name for name in episode["obs"].keys() if name.startswith("rgb"))
        raise ValueError(f"Camera dataset '{path}' not found. Available camera datasets: {available}")
    dataset = episode[path]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"'{path}' is not a dataset.")
    return dataset


def _make_canvas(frames: list[np.ndarray]) -> np.ndarray:
    heights = [frame.shape[0] for frame in frames]
    max_height = max(heights)
    padded = []
    for frame in frames:
        if frame.shape[0] == max_height:
            padded.append(frame)
            continue
        pad_height = max_height - frame.shape[0]
        padded.append(np.pad(frame, ((0, pad_height), (0, 0), (0, 0)), mode="constant"))
    return np.concatenate(padded, axis=1)


def _iter_canvases(episode: h5py.Group, cameras: list[str], max_frames: int | None, stride: int):
    datasets = [_get_camera_dataset(episode, camera) for camera in cameras]
    lengths = [dataset.shape[0] for dataset in datasets]
    total = min(lengths)
    if max_frames is not None:
        total = min(total, max_frames * stride)

    for frame_idx in range(0, total, stride):
        frames = [_to_uint8_rgb(dataset[frame_idx]) for dataset in datasets]
        yield frame_idx, _make_canvas(frames)


def print_summary(h5_file: h5py.File):
    print(f"File: {h5_file.filename}")
    print(f"Root attrs: {dict(h5_file.attrs)}")
    for name in _episode_names(h5_file):
        episode = h5_file["data"][name]
        print(f"\n{name} attrs={dict(episode.attrs)}")
        if "obs" in episode:
            for key, value in episode["obs"].items():
                if isinstance(value, h5py.Dataset):
                    print(f"  obs/{key}: shape={value.shape}, dtype={value.dtype}")
        for key in ("actions", "rewards", "dones"):
            if key in episode:
                value = episode[key]
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")


def export_h264(
    episode: h5py.Group,
    cameras: list[str],
    output_video: Path,
    fps: float,
    max_frames: int | None,
    stride: int,
):
    try:
        import imageio
    except Exception as exc:
        raise RuntimeError("H.264 export requires imageio with an ffmpeg backend.") from exc

    output_video.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with imageio.get_writer(
            output_video,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            for _, canvas in _iter_canvases(episode, cameras, max_frames, stride):
                writer.append_data(canvas)
                count += 1
    except Exception as exc:
        raise RuntimeError(
            "Failed to export H.264 MP4. The current Python environment may be missing "
            "imageio-ffmpeg or an ffmpeg binary."
        ) from exc

    print(f"Exported {count} frames to H.264 MP4: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Offline viewer for camera frames in collected HDF5 demos.")
    parser.add_argument("--file", required=True, help="Path to the HDF5 dataset file.")
    parser.add_argument("--demo", default=None, help="Episode name or index, e.g. demo_0 or 0. Defaults to first demo.")
    parser.add_argument("--cameras", nargs="+", default=["rgb_wrist", "rgb_main"], help="Camera datasets under obs/.")
    parser.add_argument("--mode", choices=["summary", "h264"], default="summary")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-h264", default=None, help="Output MP4 path. Defaults to a timestamped filename.")
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")

    with h5py.File(args.file, "r") as h5_file:
        if args.mode == "summary":
            print_summary(h5_file)
            return

        episode = _select_episode(h5_file, args.demo)
        if args.mode == "h264":
            output_h264 = Path(args.output_h264) if args.output_h264 else _default_output_h264_path(episode.name.rsplit("/", 1)[-1])
            export_h264(episode, args.cameras, output_h264, args.fps, args.max_frames, args.stride)


if __name__ == "__main__":
    main()
