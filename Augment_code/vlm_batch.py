#!/usr/bin/env python3
"""Batch VLM inference for FITR asset augmentation.

Reproduce (cd physvla_sim/Augment_code; export DASHSCOPE_API_KEY='sk-...'):

  # B — pure VLM absolute scale (single 0.png, no bench refs / no pixel calib)
  .venv/bin/python vlm_batch.py --category Dishwasher --bench --force --no-scale-refs --no-pixel-scale --output experiments/vlm_results_baseline.jsonl --sleep 1.0

  # C — Ours: VLM + FITR-Bench scale refs + pixel LOO calibration (also has_handle, door_type)
  .venv/bin/python vlm_batch.py --category Dishwasher --bench --force --sleep 1.0

  # Full FITR-Bench (all categories in fitr_bench.json)
  .venv/bin/python vlm_batch.py --bench --sleep 1.0

  # Dry-run / single asset
  .venv/bin/python vlm_batch.py --category Door --asset-id 9280 --dry-run

Outputs: experiments/vlm_results.jsonl (Ours) or experiments/vlm_results_baseline.jsonl (B).
Then run eval_vlm_bench.py to compare against fitr_bench.json GT.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import dashscope
from dashscope import MultiModalConversation

from experiment_paths import DEFAULT_VLM_RESULTS, DEFAULT_VLM_RESULTS_BASELINE
from prompt.load_prompt import (
    DEFAULT_BENCH_PATH,
    DEFAULT_DATASET_ROOT,
    DEFAULT_VLM_BASE_PATH,
    PromptBundle,
    ScaleReference,
    bench_scale_gt,
    build_prompt,
    build_prompt_for_asset,
    load_calibration_config,
    load_vlm_base,
    parse_prompt_file,
)
from scale.l_norm import compute_l_norm
from scale.pixel_calibrate import (
    MeshScaleRef,
    ScaleRef,
    category_pixel_scale,
    estimate_l_real_cm_mesh,
    resolve_loo_scale,
    resolve_merge_vlm_mode,
    resolve_merge_vlm_scale,
    resolve_scale_method,
)

AUGMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_VLM_RESULTS
DEFAULT_BASELINE_OUTPUT = DEFAULT_VLM_RESULTS_BASELINE
DEFAULT_BENCH_PATH = AUGMENT_ROOT / "fitr_bench.json"
DEFAULT_MODEL = "qwen3-vl-flash"

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_vlm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def vlm_image_paths(bundle: PromptBundle) -> list[Path]:
    paths: list[Path] = []
    if "scale" in bundle.vlm_tasks and bundle.vlm_scale_refs:
        paths.extend(ref.image_path for ref in bundle.vlm_scale_refs)
    paths.append(bundle.image_path)
    return paths


VLM_SCALE_METHODS = frozenset({"vlm_refs", "vlm_single", "vlm_hybrid"})


def single_image_bundle(bundle: PromptBundle) -> PromptBundle:
    """Same asset, prompt without scale refs (baseline-style single image)."""
    sections = parse_prompt_file()
    system, user = build_prompt(
        bundle.category,
        bundle.asset_id,
        vlm_tasks=bundle.vlm_tasks,
        joints=list(bundle.joints),
        scale_refs=bundle.scale_refs,
        vlm_scale_refs=(),
        sections=sections,
    )
    return dataclasses.replace(bundle, system=system, user=user, vlm_scale_refs=())


def merge_hybrid_scale(
    l_single: float | None,
    l_refs: float | None,
    refs: tuple[ScaleReference, ...],
    *,
    span_frac: float = 0.15,
    extrap: str = "single",
    cap_ratio: float = 1.20,
) -> tuple[float | None, str]:
    """Pick refs inside bench span; trust single when target is larger than all refs.

    extrap:
      single — upward extrapolation uses single-image VLM (Faucet)
      anchor — single spike (> ref_max * cap_ratio): refs if in-range else ref-span midpoint (Lamp)
    """
    if l_refs is None:
        return l_single, "single"
    if l_single is None:
        return l_refs, "refs"
    if not refs:
        return l_single, "single"

    ref_max = max(ref.l_real_cm for ref in refs)
    ref_min = min(ref.l_real_cm for ref in refs)
    span = max(ref_max - ref_min, 1.0)

    if extrap == "anchor":
        if l_single > ref_max * cap_ratio:
            if l_refs <= ref_max + span_frac * span:
                return l_refs, "anchor_refs"
            return (ref_min + ref_max) / 2.0, "anchor_mid"
    elif l_single > ref_max + span_frac * span:
        return l_single, "extrap_up"

    if l_refs <= ref_max + span_frac * span:
        if l_single < l_refs:
            if extrap == "anchor" and l_single < ref_min * 0.90:
                return l_refs, "anchor_refs_low"
            return l_single, "single_smaller"
        if l_single <= ref_max * 1.10:
            return l_single, "single_in_band"
        if abs(l_single - l_refs) < 0.2 * span:
            return (l_single + l_refs) / 2.0, "avg"
        return l_refs, "refs"

    return l_single, "single_fallback"


def merge_drawer_vlm_scale(
    pixel_l: float,
    vlm_l: float | None,
    refs: tuple[ScaleReference, ...],
    *,
    target_asset_id: str | None = None,
    desk_min_cm: float = 45.0,
    small_pixel_ratio: float = 1.05,
    vlm_boost_ratio: float = 1.25,
) -> tuple[float, str]:
    """Drawer: pixel base; boost ref_max anchor; trust VLM for large desks only."""
    if vlm_l is None:
        return pixel_l, "pixel"
    if not refs:
        return pixel_l, "pixel"

    ref_min_ref = min(refs, key=lambda ref: ref.l_real_cm)
    ref_max_ref = max(refs, key=lambda ref: ref.l_real_cm)
    ref_min = ref_min_ref.l_real_cm
    ref_max = ref_max_ref.l_real_cm

    if target_asset_id == ref_min_ref.asset_id:
        return pixel_l, "drawer_min_ref"

    if vlm_l > ref_max * 1.05:
        return pixel_l, "pixel_vlm_spike"

    if (
        target_asset_id == ref_max_ref.asset_id
        and pixel_l <= ref_min * small_pixel_ratio
        and ref_max > ref_min
    ):
        return pixel_l * (ref_max / ref_min), "drawer_ref_max"

    if (
        pixel_l <= ref_min * small_pixel_ratio
        and vlm_l >= desk_min_cm
        and vlm_l > pixel_l * vlm_boost_ratio
    ):
        return vlm_l, "vlm_desk"
    return pixel_l, "pixel"


def merge_pixel_vlm_scale(
    pixel_l: float,
    vlm_l: float | None,
    refs: tuple[ScaleReference, ...],
    *,
    spike_ratio: float = 1.05,
    lift_ratio: float = 1.02,
) -> tuple[float, str]:
    """Blend pixel refs with single-image VLM (same call as has_handle)."""
    if vlm_l is None:
        return pixel_l, "pixel"
    if not refs:
        return pixel_l, "pixel"

    ref_max = max(ref.l_real_cm for ref in refs)
    if vlm_l > ref_max * spike_ratio:
        return pixel_l, "pixel_vlm_spike"
    if vlm_l > pixel_l and vlm_l <= ref_max * lift_ratio:
        return vlm_l, "vlm_lift"
    return pixel_l, "pixel"


def resolve_hybrid_settings(
    category: str, calib: dict[str, Any] | None = None
) -> tuple[str, float]:
    calib = calib or load_calibration_config()
    cfg = calib.get("categories", {}).get(category, {})
    mode = str(cfg.get("hybrid_extrap", "single"))
    if mode not in ("single", "anchor"):
        mode = "single"
    cap_ratio = float(cfg.get("hybrid_cap_ratio", 1.20))
    return mode, cap_ratio


def _asset_dir_from_scale_image(image_path: Path) -> Path:
    image_path = Path(image_path)
    if image_path.parent.name in {"images", "parts_render", "parts_render_after_merging"}:
        return image_path.parent.parent
    return image_path.parent


def mesh_scale_estimate(
    bundle: PromptBundle,
    *,
    leave_one_out: bool = True,
) -> dict[str, Any] | None:
    if "scale" not in bundle.vlm_tasks or not bundle.scale_refs:
        return None
    refs: list[MeshScaleRef] = []
    for ref in bundle.scale_refs:
        ref_dir = _asset_dir_from_scale_image(ref.image_path)
        try:
            l_norm = float(compute_l_norm(ref_dir)["L_norm"])
        except (FileNotFoundError, ValueError):
            continue
        refs.append(MeshScaleRef(asset_id=ref.asset_id, l_real_cm=ref.l_real_cm, l_norm=l_norm))
    if not refs:
        return None
    return estimate_l_real_cm_mesh(
        bundle.asset_dir,
        refs,
        target_asset_id=bundle.asset_id,
        leave_one_out=leave_one_out,
    )


def bench_scale_estimate(
    bundle: PromptBundle,
    *,
    leave_one_out: bool = True,
    calib: dict[str, Any] | None = None,
    use_pixel_scale: bool = True,
) -> dict[str, Any] | None:
    calib = calib or load_calibration_config()
    leave_one_out = resolve_loo_scale(bundle.category, calib, cli_leave_one_out=leave_one_out)
    method = resolve_scale_method(bundle.category, calib)
    if method in VLM_SCALE_METHODS:
        return None
    if method == "mesh":
        return mesh_scale_estimate(bundle, leave_one_out=leave_one_out)
    if use_pixel_scale:
        return pixel_scale_estimate(bundle, leave_one_out=leave_one_out, calib=calib)
    return None


def pixel_scale_estimate(
    bundle: PromptBundle,
    *,
    leave_one_out: bool = True,
    calib: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if "scale" not in bundle.vlm_tasks or not bundle.scale_refs:
        return None
    calib = calib or load_calibration_config()
    refs = [
        ScaleRef(asset_id=ref.asset_id, l_real_cm=ref.l_real_cm, image_path=ref.image_path)
        for ref in bundle.scale_refs
    ]
    return category_pixel_scale(
        bundle.image_path,
        refs,
        category=bundle.category,
        target_asset_id=bundle.asset_id,
        leave_one_out=leave_one_out,
        calib=calib,
    )


def scale_only_tasks(bundle: PromptBundle) -> bool:
    return bundle.vlm_tasks == ["scale"]


def image_to_data_uri(path: Path) -> str:
    """Encode a local image as data URI to avoid DashScope OSS re-upload collisions."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    mime, _ = mimetypes.guess_type(resolved.name)
    if not mime:
        mime = "image/png"
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def call_vlm(
    *,
    system: str,
    user: str,
    image_paths: list[Path],
    model: str,
    api_key: str,
) -> str:
    content: list[dict[str, str]] = []
    for path in image_paths:
        content.append({"image": image_to_data_uri(path)})
    content.append({"text": user})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"text": system}]},
        {"role": "user", "content": content},
    ]

    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"VLM call failed: status={response.status_code}, "
            f"code={getattr(response, 'code', None)}, "
            f"message={getattr(response, 'message', None)}"
        )

    message = response.output.choices[0].message.content
    if isinstance(message, list):
        for part in message:
            if isinstance(part, dict) and "text" in part:
                return part["text"]
        raise RuntimeError(f"Unexpected VLM content: {message!r}")
    return str(message)


def normalize_result(parsed: dict[str, Any], bundle: PromptBundle) -> dict[str, Any]:
    result: dict[str, Any] = {
        "longest_edge_cm": parsed.get("longest_edge_cm"),
        "has_handle": parsed.get("has_handle") or {},
        "door_type": parsed.get("door_type"),
        "confidence": parsed.get("confidence"),
    }

    if "has_handle" in bundle.vlm_tasks:
        expected = {j.joint_name for j in bundle.joints}
        has_handle = result["has_handle"]
        if not isinstance(has_handle, dict):
            raise ValueError(f"has_handle must be object, got {type(has_handle).__name__}")
        result["has_handle"] = {
            name: bool(has_handle.get(name, False))
            for name in expected
        }
    else:
        result["has_handle"] = {}

    if "door_type" not in bundle.vlm_tasks:
        result["door_type"] = None

    if result["longest_edge_cm"] is not None:
        result["longest_edge_cm"] = float(result["longest_edge_cm"])

    if result["confidence"] is not None:
        result["confidence"] = float(result["confidence"])

    return result


def process_asset(
    bundle: PromptBundle,
    *,
    model: str,
    api_key: str,
    calib: dict[str, Any] | None = None,
    leave_one_out_scale: bool = True,
    use_pixel_scale: bool = True,
) -> dict[str, Any]:
    calib = calib or load_calibration_config()
    scale_method = resolve_scale_method(bundle.category, calib)
    base: dict[str, Any] = {
        "category": bundle.category,
        "asset_id": bundle.asset_id,
        "model": model,
        "vlm_tasks": bundle.vlm_tasks,
        "scale_ref_asset_ids": [ref.asset_id for ref in bundle.scale_refs],
        "vlm_scale_ref_asset_ids": [ref.asset_id for ref in bundle.vlm_scale_refs],
        "scale_method": scale_method,
        "is_bench_scale_ref": bench_scale_gt(bundle.asset_id, bundle.scale_refs) is not None,
        "timestamp": utc_now(),
    }

    bench_scale = bench_scale_estimate(
        bundle,
        leave_one_out=leave_one_out_scale,
        calib=calib,
        use_pixel_scale=use_pixel_scale,
    )
    need_vlm = (
        not scale_only_tasks(bundle)
        or scale_method in VLM_SCALE_METHODS
        or bench_scale is None
    )

    if not need_vlm and bench_scale is not None:
        result = {
            "longest_edge_cm": float(bench_scale["longest_edge_cm"]),
            "has_handle": {},
            "door_type": None,
            "confidence": 1.0,
        }
        base.update(
            status="ok",
            skipped_vlm=True,
            scale_source=bench_scale["scale_source"],
            pixel_feature=bench_scale.get("pixel_feature"),
            pixel_image=bench_scale.get("pixel_image"),
            pixel_estimates_cm=bench_scale.get("pixel_estimates_cm"),
            mesh_l_norm=bench_scale.get("mesh_l_norm"),
            mesh_estimates_cm=bench_scale.get("mesh_estimates_cm"),
            raw_response=None,
            result=result,
            error=None,
        )
        return base

    raw: str | None = None
    vlm_single_cm: float | None = None
    vlm_refs_cm: float | None = None
    hybrid_merge_mode: str | None = None
    pixel_merge_mode: str | None = None
    try:
        if need_vlm and scale_method == "vlm_hybrid":
            raw_refs = call_vlm(
                system=bundle.system,
                user=bundle.user,
                image_paths=vlm_image_paths(bundle),
                model=model,
                api_key=api_key,
            )
            result_refs = normalize_result(parse_vlm_json(raw_refs), bundle)
            vlm_refs_cm = result_refs.get("longest_edge_cm")

            single = single_image_bundle(bundle)
            raw_single = call_vlm(
                system=single.system,
                user=single.user,
                image_paths=vlm_image_paths(single),
                model=model,
                api_key=api_key,
            )
            result_single = normalize_result(parse_vlm_json(raw_single), single)
            vlm_single_cm = result_single.get("longest_edge_cm")

            extrap_mode, cap_ratio = resolve_hybrid_settings(bundle.category, calib)
            merged_l, hybrid_merge_mode = merge_hybrid_scale(
                vlm_single_cm,
                vlm_refs_cm,
                bundle.vlm_scale_refs,
                extrap=extrap_mode,
                cap_ratio=cap_ratio,
            )
            result = result_refs
            if merged_l is not None:
                result["longest_edge_cm"] = float(merged_l)
            raw = raw_refs
            base_scale_source = f"vlm_hybrid_{hybrid_merge_mode}"
            base_pixel_feature = None
            base_pixel_image = None
            base_pixel_estimates_cm = None
            base_mesh_l_norm = None
            base_mesh_estimates_cm = None
        elif need_vlm:
            raw = call_vlm(
                system=bundle.system,
                user=bundle.user,
                image_paths=vlm_image_paths(bundle),
                model=model,
                api_key=api_key,
            )
            parsed = parse_vlm_json(raw)
            result = normalize_result(parsed, bundle)
        else:
            result = {
                "longest_edge_cm": None,
                "has_handle": {},
                "door_type": None,
                "confidence": None,
            }

        if bench_scale is not None and scale_method not in VLM_SCALE_METHODS:
            pixel_l = float(bench_scale["longest_edge_cm"])
            vlm_l = result.get("longest_edge_cm")
            if (
                scale_method == "pixel"
                and vlm_l is not None
                and resolve_merge_vlm_scale(bundle.category, calib)
            ):
                if resolve_merge_vlm_mode(bundle.category, calib) == "drawer":
                    merged_l, pixel_merge_mode = merge_drawer_vlm_scale(
                        pixel_l,
                        float(vlm_l),
                        bundle.scale_refs,
                        target_asset_id=bundle.asset_id,
                    )
                else:
                    merged_l, pixel_merge_mode = merge_pixel_vlm_scale(
                        pixel_l,
                        float(vlm_l),
                        bundle.scale_refs,
                    )
                result["longest_edge_cm"] = merged_l
                vlm_single_cm = float(vlm_l)
                base_scale_source = (
                    bench_scale["scale_source"]
                    if pixel_merge_mode == "pixel"
                    else f"{bench_scale['scale_source']}_{pixel_merge_mode}"
                )
            else:
                result["longest_edge_cm"] = pixel_l
                base_scale_source = bench_scale["scale_source"]
            base_pixel_feature = bench_scale.get("pixel_feature")
            base_pixel_image = bench_scale.get("pixel_image")
            base_pixel_estimates_cm = bench_scale.get("pixel_estimates_cm")
            base_mesh_l_norm = bench_scale.get("mesh_l_norm")
            base_mesh_estimates_cm = bench_scale.get("mesh_estimates_cm")
        elif scale_method != "vlm_hybrid":
            base_scale_source = scale_method if scale_method in VLM_SCALE_METHODS else "vlm"
            base_pixel_feature = None
            base_pixel_image = None
            base_pixel_estimates_cm = None
            base_mesh_l_norm = None
            base_mesh_estimates_cm = None

        base.update(
            status="ok",
            skipped_vlm=not need_vlm,
            scale_source=base_scale_source,
            pixel_feature=base_pixel_feature,
            pixel_image=base_pixel_image,
            pixel_estimates_cm=base_pixel_estimates_cm,
            mesh_l_norm=base_mesh_l_norm,
            mesh_estimates_cm=base_mesh_estimates_cm,
            vlm_single_cm=vlm_single_cm,
            vlm_refs_cm=vlm_refs_cm,
            hybrid_merge_mode=hybrid_merge_mode,
            pixel_merge_mode=pixel_merge_mode,
            raw_response=raw,
            result=result,
            error=None,
        )
    except Exception as exc:
        if bench_scale is not None:
            result = {
                "longest_edge_cm": float(bench_scale["longest_edge_cm"]),
                "has_handle": {},
                "door_type": None,
                "confidence": 1.0,
            }
            base.update(
                status="ok",
                skipped_vlm=True,
                scale_source=bench_scale["scale_source"],
                pixel_feature=bench_scale.get("pixel_feature"),
                pixel_image=bench_scale.get("pixel_image"),
                pixel_estimates_cm=bench_scale.get("pixel_estimates_cm"),
                mesh_l_norm=bench_scale.get("mesh_l_norm"),
                mesh_estimates_cm=bench_scale.get("mesh_estimates_cm"),
                raw_response=raw,
                result=result,
                error=None,
            )
        else:
            base.update(
                status="error",
                skipped_vlm=False,
                scale_source=None,
                raw_response=raw,
                result=None,
                error=str(exc),
            )
    return base


def iter_asset_ids(
    vlm_base: dict[str, Any],
    *,
    dataset_root: Path,
    categories: list[str] | None = None,
    bench_only: bool = False,
    bench_path: Path = DEFAULT_BENCH_PATH,
) -> Iterator[tuple[str, str]]:
    cat_names = categories or list(vlm_base["categories"])

    if bench_only:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        for cat in cat_names:
            if cat not in bench["categories"]:
                continue
            for asset in bench["categories"][cat]["assets"]:
                yield cat, str(asset["asset_id"])
        return

    for cat in cat_names:
        if cat not in vlm_base["categories"]:
            continue
        cfg = vlm_base["categories"][cat]
        cat_dir = dataset_root / cfg["category_dir"]
        if not cat_dir.is_dir():
            continue
        for asset_dir in sorted(cat_dir.iterdir()):
            if asset_dir.is_dir():
                yield cat, asset_dir.name


def load_done_keys(output_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not output_path.is_file():
        return done
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add((rec["category"], rec["asset_id"]))
    return done


def append_jsonl(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batch VLM inference to vlm_results.jsonl.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--vlm-base", type=Path, default=DEFAULT_VLM_BASE_PATH)
    parser.add_argument("--bench", action="store_true", help="Only process fitr_bench.json assets")
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH_PATH)
    parser.add_argument("--category", action="append", default=None, help="Limit to category (repeatable)")
    parser.add_argument("--asset-id", default=None, help="Process a single asset (requires --category)")
    parser.add_argument("--limit", type=int, default=None, help="Max assets to process this run")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--force", action="store_true", help="Reprocess assets already marked ok")
    parser.add_argument(
        "--no-scale-refs",
        action="store_true",
        help="Pure VLM baseline: no bench scale refs in prompt (single image)",
    )
    parser.add_argument(
        "--no-pixel-scale",
        action="store_true",
        help="Do not override/replace scale with pixel bench calibration",
    )
    parser.add_argument(
        "--no-loo-scale",
        action="store_true",
        help="Do not leave-one-out bench refs when estimating pixel scale",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without calling VLM")
    args = parser.parse_args()

    if args.asset_id and not args.category:
        parser.error("--asset-id requires --category")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not args.dry_run and not api_key:
        print("DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 1

    vlm_base = load_vlm_base(args.vlm_base)
    calib = load_calibration_config()
    done = set() if args.force else load_done_keys(args.output)

    if args.asset_id:
        if not args.category or len(args.category) != 1:
            parser.error("--asset-id requires exactly one --category")
        work = [(args.category[0], args.asset_id)]
    else:
        work = list(
            iter_asset_ids(
                vlm_base,
                dataset_root=args.dataset_root,
                categories=args.category,
                bench_only=args.bench,
                bench_path=args.bench_path,
            )
        )

    pending = [(cat, aid) for cat, aid in work if (cat, aid) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"planned={len(work)} pending={len(pending)} done={len(done)} output={args.output}")
    if not pending:
        return 0

    processed = 0
    errors = 0

    for cat, asset_id in pending:
        try:
            bundle = build_prompt_for_asset(
                cat,
                asset_id,
                dataset_root=args.dataset_root,
                vlm_base=vlm_base,
                bench_path=args.bench_path,
                use_scale_refs=not args.no_scale_refs,
            )
        except Exception as exc:
            record = {
                "category": cat,
                "asset_id": asset_id,
                "status": "error",
                "timestamp": utc_now(),
                "error": f"prompt build failed: {exc}",
            }
            if not args.dry_run:
                append_jsonl(args.output, record)
            print(f"[error] {cat}/{asset_id} prompt: {exc}", file=sys.stderr)
            errors += 1
            continue

        images = vlm_image_paths(bundle)
        bench_scale = None if args.no_pixel_scale else bench_scale_estimate(
            bundle,
            leave_one_out=not args.no_loo_scale,
            calib=calib,
            use_pixel_scale=not args.no_pixel_scale,
        )
        scale_method = resolve_scale_method(cat, calib)
        skip = (
            scale_only_tasks(bundle)
            and scale_method not in VLM_SCALE_METHODS
            and bench_scale is not None
            and not args.no_pixel_scale
        )
        print(
            f"[{'dry' if args.dry_run else 'run'}] {cat}/{asset_id} "
            f"images={len(images)} skip_vlm={skip} method={scale_method} "
            f"bench_scale={None if bench_scale is None else round(float(bench_scale['longest_edge_cm']), 1)} "
            f"bench_refs={len(bundle.scale_refs)} "
            f"mode={'vlm_only' if args.no_scale_refs else 'vlm_ref' if args.no_pixel_scale else 'bench_ref'}"
        )

        if args.dry_run:
            processed += 1
            continue

        record = process_asset(
            bundle,
            model=args.model,
            api_key=api_key,
            calib=calib,
            leave_one_out_scale=not args.no_loo_scale,
            use_pixel_scale=not args.no_pixel_scale,
        )
        append_jsonl(args.output, record)

        if record["status"] == "ok":
            r = record["result"]
            print(
                f"[ok] {cat}/{asset_id} L={r['longest_edge_cm']} "
                f"handle={r['has_handle']} door={r['door_type']} conf={r['confidence']}"
            )
            processed += 1
        else:
            print(f"[error] {cat}/{asset_id}: {record['error']}", file=sys.stderr)
            errors += 1

        if args.sleep > 0 and processed + errors < len(pending):
            time.sleep(args.sleep)

    print(f"finished processed={processed} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
