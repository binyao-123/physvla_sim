#!/usr/bin/env python3
from __future__ import annotations

# 离线查看采集到的 HDF5 相机帧。MP4 默认输出到本目录（hdf5_replays/）。

import argparse
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

REPLAYS_DIR = Path(__file__).resolve().parent


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


def _default_output_h264_path(demo_name: str, output_dir: Path | None = None) -> Path:
    stem = f"hdf5_camera_replay_{demo_name}_{_timestamp_utc8()}.mp4"
    base = output_dir if output_dir is not None else REPLAYS_DIR
    return base / stem


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


def _resolve_ffmpeg(ffmpeg_path: str | None) -> str:
    if ffmpeg_path:
        return ffmpeg_path
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or pass --ffmpeg.") from exc


def _ffmpeg_supports_encoder(ffmpeg_exe: str, encoder: str) -> bool:
    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return encoder in result.stdout


def _resolve_encoder(requested: str, ffmpeg_exe: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "nvenc":
        if not _ffmpeg_supports_encoder(ffmpeg_exe, "h264_nvenc"):
            raise RuntimeError(
                f"Requested NVENC encoder but '{ffmpeg_exe}' has no h264_nvenc. "
                "Use system ffmpeg (/usr/bin/ffmpeg) or --encoder cpu."
            )
        return "nvenc"
    if _ffmpeg_supports_encoder(ffmpeg_exe, "h264_nvenc"):
        return "nvenc"
    return "cpu"


def _ensure_even_canvas(canvas: np.ndarray) -> np.ndarray:
    height, width = canvas.shape[:2]
    even_h = height - (height % 2)
    even_w = width - (width % 2)
    if even_h != height or even_w != width:
        return canvas[:even_h, :even_w]
    return canvas


def _count_export_frames(episode: h5py.Group, cameras: list[str], max_frames: int | None, stride: int) -> int:
    datasets = [_get_camera_dataset(episode, camera) for camera in cameras]
    total_frames = min(dataset.shape[0] for dataset in datasets)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames * stride)
    return len(range(0, total_frames, stride))


def _export_h264_nvenc(
    episode: h5py.Group,
    cameras: list[str],
    output_video: Path,
    fps: float,
    max_frames: int | None,
    stride: int,
    ffmpeg_exe: str,
) -> int:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    total_to_write = _count_export_frames(episode, cameras, max_frames, stride)
    print(
        f"Exporting {episode.name} -> {output_video} "
        f"({total_to_write} frames, stride={stride}, fps={fps}, encoder=h264_nvenc)..."
    )

    canvas_iter = _iter_canvases(episode, cameras, max_frames, stride)
    _, first_canvas = next(canvas_iter)
    first_canvas = _ensure_even_canvas(first_canvas)
    height, width = first_canvas.shape[:2]

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    count = 0
    try:
        proc.stdin.write(first_canvas.tobytes())
        count = 1
        if count == 1 or count % 200 == 0 or count == total_to_write:
            print(f"  [{episode.name}] {count}/{total_to_write} frames written")

        for _, canvas in canvas_iter:
            canvas = _ensure_even_canvas(canvas)
            if canvas.shape[0] != height or canvas.shape[1] != width:
                raise ValueError(
                    f"Frame size changed to {canvas.shape[1]}x{canvas.shape[0]}, expected {width}x{height}."
                )
            proc.stdin.write(canvas.tobytes())
            count += 1
            if count % 200 == 0 or count == total_to_write:
                print(f"  [{episode.name}] {count}/{total_to_write} frames written")
    finally:
        proc.stdin.close()

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg NVENC export failed (code={return_code}): {stderr.strip()}")

    print(f"Exported {count} frames to H.264 MP4 (NVENC): {output_video}")
    return count


def _export_h264_cpu(
    episode: h5py.Group,
    cameras: list[str],
    output_video: Path,
    fps: float,
    max_frames: int | None,
    stride: int,
) -> int:
    try:
        import imageio
    except Exception as exc:
        raise RuntimeError("CPU export requires imageio with an ffmpeg backend.") from exc

    output_video.parent.mkdir(parents=True, exist_ok=True)
    total_to_write = _count_export_frames(episode, cameras, max_frames, stride)
    print(
        f"Exporting {episode.name} -> {output_video} "
        f"({total_to_write} frames, stride={stride}, fps={fps}, encoder=libx264/cpu)..."
    )

    count = 0
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
            if count == 1 or count % 200 == 0 or count == total_to_write:
                print(f"  [{episode.name}] {count}/{total_to_write} frames written")

    print(f"Exported {count} frames to H.264 MP4 (CPU): {output_video}")
    return count


def export_h264(
    episode: h5py.Group,
    cameras: list[str],
    output_video: Path,
    fps: float,
    max_frames: int | None,
    stride: int,
    encoder: str = "auto",
    ffmpeg_path: str | None = None,
):
    ffmpeg_exe = _resolve_ffmpeg(ffmpeg_path)
    resolved = _resolve_encoder(encoder, ffmpeg_exe)
    if resolved == "nvenc":
        return _export_h264_nvenc(
            episode, cameras, output_video, fps, max_frames, stride, ffmpeg_exe
        )
    return _export_h264_cpu(episode, cameras, output_video, fps, max_frames, stride)


def export_all_h264(
    h5_file: h5py.File,
    cameras: list[str],
    output_dir: Path,
    fps: float,
    max_frames: int | None,
    stride: int,
    encoder: str = "auto",
    ffmpeg_path: str | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for demo_name in _episode_names(h5_file):
        episode = h5_file["data"][demo_name]
        success = episode.attrs.get("success", None)
        if success in (True, 1, "True", "true"):
            success_tag = "ok"
        elif success in (False, 0, "False", "false"):
            success_tag = "fail"
        else:
            success_tag = "unk"
        output_video = output_dir / f"{demo_name}_{success_tag}.mp4"
        export_h264(
            episode,
            cameras,
            output_video,
            fps,
            max_frames,
            stride,
            encoder=encoder,
            ffmpeg_path=ffmpeg_path,
        )
        exported.append(output_video)
    return exported


def main():
    parser = argparse.ArgumentParser(description="Offline viewer for camera frames in collected HDF5 demos.")
    parser.add_argument("--file", required=True, help="Path to the HDF5 dataset file.")
    parser.add_argument(
        "--demo",
        default=None,
        help="Episode name or index, e.g. demo_0 or 0. Use 'all' to export every demo (h264 mode).",
    )
    parser.add_argument("--cameras", nargs="+", default=["rgb_wrist", "rgb_main"], help="Camera datasets under obs/.")
    parser.add_argument("--mode", choices=["summary", "h264"], default="summary")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-h264", default=None, help="Output MP4 path for a single demo export.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for --demo all batch export. Defaults to hdf5_replays/<hdf5_stem>/.",
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "nvenc", "cpu"],
        default="auto",
        help="Video encoder: auto prefers GPU h264_nvenc via system ffmpeg, cpu uses libx264.",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="ffmpeg executable path. Default: system ffmpeg, then imageio bundled ffmpeg.",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")

    h5_path = Path(args.file)
    with h5py.File(h5_path, "r") as h5_file:
        if args.mode == "summary":
            print_summary(h5_file)
            return

        if args.demo == "all":
            if args.output_h264:
                raise ValueError("--output-h264 cannot be used with --demo all; use --output-dir instead.")
            output_dir = Path(args.output_dir) if args.output_dir else REPLAYS_DIR / h5_path.stem
            exported = export_all_h264(
                h5_file,
                args.cameras,
                output_dir,
                args.fps,
                args.max_frames,
                args.stride,
                encoder=args.encoder,
                ffmpeg_path=args.ffmpeg,
            )
            print(f"Exported {len(exported)} videos to {output_dir.resolve()}")
            return

        episode = _select_episode(h5_file, args.demo)
        if args.mode == "h264":
            demo_name = episode.name.rsplit("/", 1)[-1]
            if args.output_h264:
                output_h264 = Path(args.output_h264)
            elif args.output_dir:
                output_h264 = Path(args.output_dir) / f"{demo_name}.mp4"
            else:
                output_h264 = _default_output_h264_path(demo_name)
            export_h264(
                episode,
                args.cameras,
                output_h264,
                args.fps,
                args.max_frames,
                args.stride,
                encoder=args.encoder,
                ffmpeg_path=args.ffmpeg,
            )


if __name__ == "__main__":
    main()
