"""Estimate absolute scale from bench references via pixel geometry or mesh L_norm."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from scale.l_norm import compute_l_norm

FEATURES = ("area", "longest", "h", "w")
SCALE_METHODS = ("pixel", "mesh", "vlm_refs", "vlm_single", "vlm_hybrid")


@dataclass(frozen=True)
class ScaleRef:
    asset_id: str
    l_real_cm: float
    image_path: Path


def find_scale_image(asset_dir: Path) -> Path | None:
    """Prefer preview images that preserve relative framing (not fill-frame renders)."""
    asset_dir = Path(asset_dir)
    images = asset_dir / "images"
    if not images.is_dir():
        return None
    for name in ("0.png", "rendered_0.png"):
        candidate = images / name
        if candidate.is_file():
            return candidate
    matches = sorted(images.glob("*.png"))
    if matches:
        return matches[0]
    matches = sorted(images.glob("*.jpg"))
    return matches[0] if matches else None


def measure_features(image_path: Path) -> dict[str, float]:
    image_path = Path(image_path)
    im = np.array(Image.open(image_path).convert("RGB"))
    corners = np.array([im[0, 0], im[0, -1], im[-1, 0], im[-1, -1]], dtype=float)
    bg = np.median(corners, axis=0)
    mask = np.abs(im.astype(float) - bg).sum(axis=2) > 30
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"No foreground pixels in {image_path}")
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    area = float(mask.sum())
    return {
        "area": area,
        "longest": max(w, h),
        "h": h,
        "w": w,
    }


def _pick_feature(
    ref_features: dict[str, dict[str, float]],
    preferred: str | None,
) -> str | None:
    if preferred and preferred in FEATURES:
        return preferred

    best_name: str | None = None
    best_spread = -1.0
    for name in FEATURES:
        values = [ref_features[aid][name] for aid in ref_features]
        if max(values) <= 0:
            continue
        spread = (max(values) - min(values)) / max(values)
        if spread > best_spread:
            best_spread = spread
            best_name = name
    if best_spread < 0.005:
        return None
    return best_name


def estimate_l_real_cm(
    target_image: Path,
    refs: Iterable[ScaleRef],
    *,
    target_asset_id: str | None = None,
    leave_one_out: bool = False,
    feature: str | None = None,
) -> dict[str, object] | None:
    """LOO-capable scale estimate: L_target = median(L_ref * f_target / f_ref)."""
    target_image = Path(target_image)
    active_refs = [
        ref
        for ref in refs
        if not (leave_one_out and target_asset_id is not None and ref.asset_id == target_asset_id)
    ]
    if not active_refs:
        return None

    try:
        target_feat = measure_features(target_image)
    except ValueError:
        return None

    ref_features: dict[str, dict[str, float]] = {}
    for ref in active_refs:
        try:
            ref_features[ref.asset_id] = measure_features(ref.image_path)
        except ValueError:
            continue
    if not ref_features:
        return None

    feat_name = _pick_feature(ref_features, feature)
    if feat_name is None:
        return None

    target_value = target_feat[feat_name]
    if target_value <= 0:
        return None

    estimates: list[float] = []
    per_ref: dict[str, float] = {}
    for ref in active_refs:
        feats = ref_features.get(ref.asset_id)
        if feats is None:
            continue
        ref_value = feats[feat_name]
        if ref_value <= 0:
            continue
        est = ref.l_real_cm * target_value / ref_value
        estimates.append(est)
        per_ref[ref.asset_id] = est

    if not estimates:
        return None

    return {
        "longest_edge_cm": statistics.median(estimates),
        "scale_source": "ref_pixel",
        "pixel_feature": feat_name,
        "pixel_image": target_image.name,
        "pixel_estimates_cm": per_ref,
        "pixel_ref_asset_ids": [ref.asset_id for ref in active_refs],
    }


def estimate_l_real_cm_aggregate(
    target_image: Path,
    refs: Iterable[ScaleRef],
    features: list[str],
    *,
    target_asset_id: str | None = None,
    leave_one_out: bool = False,
    aggregate: str = "min",
) -> dict[str, object] | None:
    """Run pixel calibration per feature; combine with min/max/median."""
    if not features:
        return None
    if len(features) == 1:
        return estimate_l_real_cm(
            target_image,
            refs,
            target_asset_id=target_asset_id,
            leave_one_out=leave_one_out,
            feature=features[0],
        )

    per_feature: dict[str, float] = {}
    base: dict[str, object] | None = None
    for feat in features:
        est = estimate_l_real_cm(
            target_image,
            refs,
            target_asset_id=target_asset_id,
            leave_one_out=leave_one_out,
            feature=feat,
        )
        if est is None:
            continue
        per_feature[feat] = float(est["longest_edge_cm"])
        if base is None:
            base = est

    if not per_feature or base is None:
        return None

    values = list(per_feature.values())
    if aggregate == "max":
        merged = max(values)
    elif aggregate == "median":
        merged = statistics.median(values)
    else:
        merged = min(values)

    out = dict(base)
    out["longest_edge_cm"] = merged
    out["pixel_feature"] = f"{aggregate}:" + "+".join(per_feature)
    out["pixel_feature_estimates_cm"] = per_feature
    return out


@dataclass(frozen=True)
class MeshScaleRef:
    asset_id: str
    l_real_cm: float
    l_norm: float


def estimate_l_real_cm_mesh(
    target_asset_dir: Path,
    refs: Iterable[MeshScaleRef],
    *,
    target_asset_id: str | None = None,
    leave_one_out: bool = False,
) -> dict[str, object] | None:
    """LOO mesh ratio: L_target = median(L_ref * L_norm_target / L_norm_ref)."""
    target_asset_dir = Path(target_asset_dir)
    try:
        target_l_norm = float(compute_l_norm(target_asset_dir)["L_norm"])
    except (FileNotFoundError, ValueError):
        return None
    if target_l_norm <= 0:
        return None

    active_refs = [
        ref
        for ref in refs
        if not (leave_one_out and target_asset_id is not None and ref.asset_id == target_asset_id)
    ]
    if not active_refs:
        return None

    estimates: list[float] = []
    per_ref: dict[str, float] = {}
    for ref in active_refs:
        if ref.l_norm <= 0:
            continue
        est = ref.l_real_cm * target_l_norm / ref.l_norm
        estimates.append(est)
        per_ref[ref.asset_id] = est

    if not estimates:
        return None

    return {
        "longest_edge_cm": statistics.median(estimates),
        "scale_source": "ref_mesh",
        "mesh_l_norm": target_l_norm,
        "mesh_estimates_cm": per_ref,
        "mesh_ref_asset_ids": [ref.asset_id for ref in active_refs],
    }


def resolve_scale_method(category: str, calib: dict[str, object]) -> str:
    cfg = calib.get("categories", {}).get(category, {})
    method = cfg.get("scale_method", "pixel")
    if method not in SCALE_METHODS:
        return "pixel"
    return str(method)


def resolve_loo_scale(category: str, calib: dict[str, object], *, cli_leave_one_out: bool = True) -> bool:
    cfg = calib.get("categories", {}).get(category, {})
    if "loo_scale" in cfg:
        return bool(cfg["loo_scale"])
    return cli_leave_one_out


def resolve_merge_vlm_scale(category: str, calib: dict[str, object]) -> bool:
    cfg = calib.get("categories", {}).get(category, {})
    return bool(cfg.get("merge_vlm_scale", True))


def resolve_merge_vlm_mode(category: str, calib: dict[str, object]) -> str:
    cfg = calib.get("categories", {}).get(category, {})
    mode = str(cfg.get("merge_vlm_mode", "default"))
    if mode in ("default", "drawer"):
        return mode
    return "default"


def resolve_pixel_features(category: str, calib: dict[str, object]) -> tuple[list[str], str]:
    cfg = calib.get("categories", {}).get(category, {})
    features = cfg.get("pixel_features")
    if features:
        return [str(f) for f in features], str(cfg.get("pixel_aggregate", "min"))
    feature = cfg.get("feature")
    if feature:
        return [str(feature)], "single"
    return [], "single"


def category_pixel_scale(
    target_image: Path,
    refs: Iterable[ScaleRef],
    *,
    category: str,
    target_asset_id: str | None = None,
    leave_one_out: bool = False,
    calib: dict[str, object] | None = None,
) -> dict[str, object] | None:
    calib = calib or {}
    features, aggregate = resolve_pixel_features(category, calib)
    if not features:
        return estimate_l_real_cm(
            target_image,
            refs,
            target_asset_id=target_asset_id,
            leave_one_out=leave_one_out,
            feature=None,
        )
    return estimate_l_real_cm_aggregate(
        target_image,
        refs,
        features,
        target_asset_id=target_asset_id,
        leave_one_out=leave_one_out,
        aggregate=aggregate,
    )
