#!/usr/bin/env python3
"""Minimal VLM helper for FITR scale / asset augmentation experiments.

Smoke test (cd physvla_sim/Augment_code; export DASHSCOPE_API_KEY='sk-...'):

  .venv/bin/python vlm_scale.py --image ../datasets/data_normalized/Dishwasher_urdf/11453/images/0.png

Batch absolute-scale / bench experiments: use vlm_batch.py and eval_vlm_bench.py (see their headers).
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

import dashscope
from dashscope import MultiModalConversation

# 华北2（北京）默认 endpoint
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

DEFAULT_MODEL = "qwen3-vl-flash"
DEFAULT_PROMPT = "你看到了什么？请用中文简要描述。"


def image_to_data_uri(path: Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    mime, _ = mimetypes.guess_type(resolved.name)
    if not mime:
        mime = "image/png"
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def ask_image(
    image_path: str | Path,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> str:
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Run: export DASHSCOPE_API_KEY='sk-...'"
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_to_data_uri(image_path)},
                {"text": prompt},
            ],
        }
    ]

    response = MultiModalConversation.call(
        api_key=key,
        model=model,
        messages=messages,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"VLM call failed: status={response.status_code}, "
            f"code={getattr(response, 'code', None)}, "
            f"message={getattr(response, 'message', None)}"
        )

    return response.output.choices[0].message.content[0]["text"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Call Bailian VLM on a local image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "/home/ubuntu/workspace/physvla_sim/datasets/data_normalized/"
            "Dishwasher_urdf/11453/images/0.png"
        ),
        help="Path to a local image file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Question sent to the VLM.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="DashScope vision model name.",
    )
    args = parser.parse_args()

    try:
        answer = ask_image(args.image, prompt=args.prompt, model=args.model)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"image: {args.image}")
    print(f"model: {args.model}")
    print("---")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
