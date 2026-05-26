"""Mesh loading and point-cloud sampling for contact/grasp candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mesh_vertices_and_normals(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    try:
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh", process=True)
        if not hasattr(mesh, "vertices"):
            raise TypeError(f"Unsupported mesh type at {mesh_path}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        else:
            mesh.fix_normals()
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        return vertices, normals
    except ImportError:
        vertices = _load_obj_vertices(mesh_path)
        normals = _estimate_normals_knn(vertices)
        return vertices, normals


def _load_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices in OBJ: {path}")
    return np.asarray(verts, dtype=np.float64)


def _estimate_normals_knn(points: np.ndarray, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    normals = np.zeros_like(points)
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=min(k, len(points)))
        neighbors = points[np.atleast_1d(idx)]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        n = vh[-1]
        if np.dot(n, p - points.mean(axis=0)) < 0:
            n = -n
        normals[i] = n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(norms, 1e-9, None)


def subsample_points(points: np.ndarray, normals: np.ndarray, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, normals
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], normals[idx]


def distance_to_axis(points: np.ndarray, axis_origin: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    axis_dir = axis_dir / max(np.linalg.norm(axis_dir), 1e-9)
    rel = points - axis_origin.reshape(1, 3)
    along = rel @ axis_dir
    perp = rel - along.reshape(-1, 1) * axis_dir.reshape(1, 3)
    return np.linalg.norm(perp, axis=1)


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    if num_samples >= len(points):
        return np.arange(len(points), dtype=np.int64)
    selected = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    dists = np.full(len(points), np.inf, dtype=np.float64)
    for _ in range(num_samples - 1):
        last = points[selected[-1]]
        dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))
        selected.append(int(np.argmax(dists)))
    return np.asarray(selected, dtype=np.int64)
