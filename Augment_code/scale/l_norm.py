"""Compute L_norm (normalized-space bbox longest edge) per FITR 4.1.1 / appendix B."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CATEGORY_DIR = {
    "Laptop": "Laptop_urdf",
    "Display": "Display_urdf",
    "Microwave": "Microwave_urdf",
    "Drawer": "table_urdf",
    "Lamp": "lamp_urdf",
    "Faucet": "Faucet_urdf",
    "Knife": "Knife_urdf",
    "Dishwasher": "Dishwasher_urdf",
    "Door": "Door_urdf",
    "Refrigerator": "Refrigerator_urdf",
    "Scissors": "Scissors_urdf",
    "StorageFurniture": "StorageFurniture_urdf",
}


def load_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices in {path}")
    return np.array(verts, dtype=np.float64)


def bbox_longest_edge(verts: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (L_norm, bbox_dims) for axis-aligned bounding box."""
    dims = verts.max(axis=0) - verts.min(axis=0)
    return float(dims.max()), dims


def collect_asset_vertices(asset_dir: Path) -> tuple[np.ndarray, str]:
    """Load all mesh vertices for an asset. Prefer whole.obj, else PartNet textured_objs."""
    whole = asset_dir / "whole.obj"
    if whole.is_file():
        return load_obj_vertices(whole), "whole.obj"

    textured_dir = asset_dir / "textured_objs"
    if textured_dir.is_dir():
        objs = sorted(textured_dir.glob("*.obj"))
        if objs:
            chunks = [load_obj_vertices(p) for p in objs]
            return np.vstack(chunks), f"textured_objs({len(objs)})"

    objs = sorted(p for p in asset_dir.glob("*.obj") if p.name != "whole.obj")
    if objs:
        chunks = [load_obj_vertices(p) for p in objs]
        return np.vstack(chunks), f"objs({len(objs)})"

    raise FileNotFoundError(f"No mesh found under {asset_dir}")


def compute_l_norm(asset_dir: Path) -> dict:
    verts, source = collect_asset_vertices(asset_dir)
    l_norm, dims = bbox_longest_edge(verts)
    return {
        "L_norm": l_norm,
        "bbox_dims": dims.tolist(),
        "mesh_source": source,
    }


def compute_scale(l_real_cm: float, l_norm: float) -> float:
    """FITR: s = L_real / L_norm (same units; L_real in cm, L_norm dimensionless)."""
    if l_norm <= 0:
        raise ValueError(f"L_norm must be positive, got {l_norm}")
    return l_real_cm / l_norm


def asset_dir_for(category: str, asset_id: str, dataset_root: Path) -> Path:
    if category not in CATEGORY_DIR:
        raise KeyError(f"Unknown category: {category}")
    return dataset_root / CATEGORY_DIR[category] / asset_id
