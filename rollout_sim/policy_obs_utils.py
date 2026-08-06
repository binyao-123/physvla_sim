"""Build Pi05 observations and rollout artifacts (no Isaac imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

POLICY_IMAGE_HEIGHT = 224
POLICY_IMAGE_WIDTH = 224


def expand_arm7_to_dual14(arm7: torch.Tensor) -> torch.Tensor:
    """Left 7-D arm state/action → 14-D (right half zeros)."""

    if arm7.shape[-1] != 7:
        raise ValueError(f"expected last dim 7, got {arm7.shape}")
    batch = arm7.shape[0]
    out = torch.zeros(batch, 14, dtype=arm7.dtype, device=arm7.device)
    out[:, :7] = arm7
    return out


def rgb_to_policy_image(rgb: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Sensor RGB → float CHW [0, 1], directly stretched to 224x224.

    Accepts HWC/CHW, optional alpha (drops 4th channel), and either uint8 or
    float buffers. Floats in [0, 255] are scaled the same as uint8 — Isaac can
    return either depending on annotator/dtype path.
    """

    if rgb is None:
        raise RuntimeError("RGB frame is None")
    x = rgb.detach()
    if x.dim() == 3 and x.shape[-1] in (3, 4):
        x = x[..., :3].permute(2, 0, 1)
    elif x.dim() == 3 and x.shape[0] in (3, 4):
        x = x[:3]
    elif x.dim() == 3:
        raise ValueError(f"Unsupported RGB layout {tuple(x.shape)}; expected HWC/CHW RGB(A).")
    else:
        raise ValueError(f"Expected 3D RGB tensor, got shape {tuple(x.shape)}")

    x = x.unsqueeze(0).to(dtype=torch.float32)
    # Scale any 0..255-like buffer (uint8 or float) into [0, 1].
    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x.clamp(0.0, 1.0).to(device=device)
    if x.shape[-2:] != (POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH):
        x = F.interpolate(
            x,
            size=(POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
    return x


def black_image_like(ref: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Match ref (1,3,H,W) with zeros for right_wrist placeholder."""

    return torch.zeros_like(ref)


def build_state14(
    robot,
    arm_joint_ids: list[int],
    *,
    gripper_open01: float,
    device: torch.device | str,
) -> torch.Tensor:
    """6 arm joints (rad) + gripper 0/1 → (1, 14)."""

    arm = robot.data.joint_pos[:, arm_joint_ids].clone()
    g = torch.tensor([[gripper_open01]], dtype=arm.dtype, device=arm.device)
    arm7 = torch.cat([arm, g], dim=-1)
    return expand_arm7_to_dual14(arm7).to(device=device)


def build_policy_observation(
    *,
    rgb_main: torch.Tensor,
    rgb_wrist: torch.Tensor,
    state14: torch.Tensor,
    task: str,
    device: torch.device | str,
) -> dict[str, Any]:
    head = rgb_to_policy_image(rgb_main, device)
    left_wrist = rgb_to_policy_image(rgb_wrist, device)
    right_wrist = black_image_like(head, device)
    return {
        "observation.images.head": head,
        "observation.images.left_wrist": left_wrist,
        "observation.images.right_wrist": right_wrist,
        "observation.state": state14.to(dtype=torch.float32),
        "task": [task],
    }


def rgb_tensor_to_uint8_hwc(rgb: torch.Tensor) -> np.ndarray:
    """(H,W,3) or (3,H,W) on CPU uint8 for video."""

    x = rgb.detach().cpu()
    if x.dtype != torch.uint8:
        x = (x.clamp(0, 1) * 255.0).byte() if x.max() <= 1.5 else x.byte()
    if x.dim() == 3 and x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    return x.numpy()


def compose_video_frame(
    rgb_main: torch.Tensor | None,
    rgb_wrist: torch.Tensor | None,
    layout: str,
) -> np.ndarray | None:
    if rgb_main is None:
        return None
    main = rgb_tensor_to_uint8_hwc(rgb_main)
    if layout == "head":
        return main
    if rgb_wrist is None:
        return main
    wrist = rgb_tensor_to_uint8_hwc(rgb_wrist)
    if main.shape != wrist.shape:
        import cv2

        wrist = cv2.resize(wrist, (main.shape[1], main.shape[0]))
    return np.concatenate([main, wrist], axis=1)


def write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    try:
        import imageio
    except ImportError as exc:
        raise RuntimeError("imageio required for MP4 export") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)


def write_summary(output_dir: Path, episodes: list[dict[str, Any]]) -> None:
    success = sum(1 for ep in episodes if ep.get("success"))
    total = len(episodes)
    summary = {
        "success_rate": (success / total) if total else 0.0,
        "successes": success,
        "episodes": total,
        "episode_results": episodes,
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {path} (success_rate={summary['success_rate']:.3f})")
