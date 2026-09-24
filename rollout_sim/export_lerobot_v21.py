#!/usr/bin/env python3
"""Convert a PLD collection run into a LeRobot v2.1 dataset.

The collector writes one npz of states/actions plus one mp4 per camera per attempt.
That is already the medium LeRobot v2.1 uses for images, so this is mostly a relayout
plus the metadata LeRobot needs but that cannot be known until collection ends:

  * only successful attempts are distilled, and v2.1 wants ``episode_index`` contiguous
    from 0, so episodes are renumbered here. ``meta/source_mapping.json`` keeps the link
    back to the original attempt index and its episode_configs entry, which is the only
    way to rebuild the scene a trajectory came from.
  * ``info.json`` needs dataset-wide totals, and ``episodes_stats.jsonl`` needs per-episode
    statistics.

Image statistics follow LeRobot: per channel, on [0,1] floats, over an evenly spaced
subsample of frames rather than every frame.

  python3 export_lerobot_v21.py <collection_dir> <output_dir> [--max-image-samples 100]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
# Collector camera name -> LeRobot feature key.
CAMERA_KEYS = {"head": "observation.images.head", "wrist": "observation.images.left_wrist"}
STATE_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]


def video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(video=0))


def sample_video_stats(path: Path, max_samples: int) -> tuple[dict[str, list], int]:
    """Per-channel min/max/mean/std on [0,1] floats over evenly spaced frames."""

    with av.open(str(path)) as container:
        total = int(container.streams.video[0].frames) or 0
        wanted = (
            set(np.linspace(0, total - 1, min(max_samples, total)).astype(int).tolist())
            if total > 0
            else None
        )
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if wanted is not None and i not in wanted:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if wanted is not None and len(frames) == len(wanted):
                break

    arr = np.asarray(frames, dtype=np.float32) / 255.0  # (N, H, W, C)
    per_channel = arr.reshape(-1, arr.shape[-1])
    stats = {
        "min": per_channel.min(0).reshape(3, 1, 1).tolist(),
        "max": per_channel.max(0).reshape(3, 1, 1).tolist(),
        "mean": per_channel.mean(0).reshape(3, 1, 1).tolist(),
        "std": per_channel.std(0).reshape(3, 1, 1).tolist(),
        "count": [len(frames)],
    }
    return stats, len(frames)


def array_stats(values: np.ndarray) -> dict[str, list]:
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [int(values.shape[0])],
    }


def scalar_stats(values: np.ndarray) -> dict[str, list]:
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "collection_dir",
        type=Path,
        nargs="+",
        help="One or more collection runs; a top-up run is merged in order after the first.",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-image-samples", type=int, default=100)
    parser.add_argument("--robot-type", type=str, default="piper_left6")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also export attempts the expert did not finish (off by default).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Keep only the first N successful episodes after merge. 0 keeps all.",
    )
    args = parser.parse_args()

    sources = [p.expanduser().resolve() for p in args.collection_dir]
    out = args.output_dir.expanduser().resolve()

    fps = None
    task_text = None
    kept: list[tuple[Path, dict]] = []
    for src in sources:
        run_meta = json.loads((src / "run_meta.json").read_text(encoding="utf-8"))
        summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
        src_fps = int(run_meta["control_hz"])
        src_task = str(run_meta["instruction"]).strip()
        # Merging runs that disagree on fps or prompt would produce a silently wrong dataset.
        if fps is None:
            fps, task_text = src_fps, src_task
        elif (src_fps, src_task) != (fps, task_text):
            print(
                f"[ERROR] {src} has fps={src_fps} task={src_task!r}, "
                f"incompatible with fps={fps} task={task_text!r}"
            )
            return 2
        attempts = sorted(summary["episode_results"], key=lambda e: e["episode_index"])
        picked = [e for e in attempts if e["success"] or args.include_failed]
        kept.extend((src, e) for e in picked)
        print(f"[export] {src.name}: {len(picked)} of {len(attempts)} attempts kept")

    if args.max_episodes > 0 and len(kept) > args.max_episodes:
        print(f"[export] truncating {len(kept)} -> {args.max_episodes} episodes")
        kept = kept[: args.max_episodes]

    if not kept:
        print("[ERROR] nothing to export")
        return 2

    # An empty target is fine; refuse a populated one rather than mixing two datasets.
    if out.exists() and any(out.iterdir()):
        print(f"[ERROR] output dir exists and is not empty: {out}")
        return 2
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for key in CAMERA_KEYS.values():
        (out / "videos" / "chunk-000" / key).mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    episodes_stats = []
    source_mapping = []
    global_index = 0
    total_frames = 0

    for new_index, (src, attempt) in enumerate(kept):
        old_index = int(attempt["episode_index"])
        npz_path = src / "raw" / f"episode_{old_index:04d}.npz"
        data = np.load(npz_path)
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        length = int(states.shape[0])
        if actions.shape[0] != length:
            print(f"[ERROR] episode {old_index}: {length} states vs {actions.shape[0]} actions")
            return 2

        # A frame-count mismatch would silently misalign images against actions.
        for cam, key in CAMERA_KEYS.items():
            video_src = src / "raw" / f"episode_{old_index:04d}_{cam}.mp4"
            n_frames = video_frame_count(video_src)
            if n_frames != length:
                print(f"[ERROR] episode {old_index} {cam}: {n_frames} frames vs {length} rows")
                return 2
            shutil.copy2(
                video_src, out / "videos" / "chunk-000" / key / f"episode_{new_index:06d}.mp4"
            )

        frame_index = np.arange(length, dtype=np.int64)
        timestamp = (frame_index / float(fps)).astype(np.float32)
        index = np.arange(global_index, global_index + length, dtype=np.int64)
        table = pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 7)),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 7)),
                "timestamp": pa.array(timestamp, type=pa.float32()),
                "frame_index": pa.array(frame_index, type=pa.int64()),
                "episode_index": pa.array(np.full(length, new_index, dtype=np.int64)),
                "index": pa.array(index, type=pa.int64()),
                "task_index": pa.array(np.zeros(length, dtype=np.int64)),
            }
        )
        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet")

        stats = {
            "observation.state": array_stats(states),
            "action": array_stats(actions),
            "timestamp": scalar_stats(timestamp),
            "frame_index": scalar_stats(frame_index.astype(np.float64)),
            "episode_index": scalar_stats(np.full(length, new_index, dtype=np.float64)),
            "index": scalar_stats(index.astype(np.float64)),
            "task_index": scalar_stats(np.zeros(length, dtype=np.float64)),
        }
        for cam, key in CAMERA_KEYS.items():
            video_out = out / "videos" / "chunk-000" / key / f"episode_{new_index:06d}.mp4"
            stats[key], _ = sample_video_stats(video_out, args.max_image_samples)

        episodes_meta.append({"episode_index": new_index, "tasks": [task_text], "length": length})
        episodes_stats.append({"episode_index": new_index, "stats": stats})
        source_mapping.append(
            {
                "episode_index": new_index,
                "source_run": str(src),
                "source_attempt_index": old_index,
                "source_episode_config": f"episode_configs/episode_{old_index:04d}.json",
                "probe_steps": attempt.get("probe_steps"),
                "takeover_steps": (attempt.get("segment_counts") or {}).get("takeover"),
                "success": bool(attempt["success"]),
            }
        )
        global_index += length
        total_frames += length
        if (new_index + 1) % 25 == 0 or new_index + 1 == len(kept):
            print(f"[export] {new_index + 1}/{len(kept)} episodes, {total_frames} frames", flush=True)

    video_info = {
        "video.fps": float(fps),
        "video.height": 224,
        "video.width": 224,
        "video.channels": 3,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    features = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [7], "names": STATE_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in CAMERA_KEYS.values():
        features[key] = {
            "dtype": "video",
            "shape": [224, 224, 3],
            "names": ["height", "width", "channel"],
            "info": dict(video_info),
        }

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": len(kept),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(kept) * len(CAMERA_KEYS),
        "total_chunks": 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(kept)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out / "meta" / "info.json").write_text(
        json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_jsonl(out / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_text}])
    write_jsonl(out / "meta" / "episodes.jsonl", episodes_meta)
    write_jsonl(out / "meta" / "episodes_stats.jsonl", episodes_stats)
    (out / "meta" / "source_mapping.json").write_text(
        json.dumps(
            {
                "note": (
                    "Failed attempts are dropped and episodes renumbered, so episode_index "
                    "here does not match the collector's file names. Use source_attempt_index "
                    "with the collection run's episode_configs/ to rebuild a scene."
                ),
                "collection_runs": [str(p) for p in sources],
                "episodes": source_mapping,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n[export] wrote {out}")
    print(f"[export] {len(kept)} episodes, {total_frames} frames, fps={fps}, task={task_text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
