"""Load link mesh surfaces from URDF-referenced OBJ files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scale.l_norm import load_obj_vertices

# Skip contact-point derivation on extremely dense contact-link meshes only.
# Meshes above EARLY_SUBSAMPLE_VERT use random subsampling before normal estimation.
DENSE_VERT_THRESHOLD = 30_000
DENSE_FACE_THRESHOLD = 60_000
SKIP_DENSE_VERT_THRESHOLD = 80_000
SKIP_DENSE_FACE_THRESHOLD = 160_000


def count_obj_mesh_size(path: Path) -> tuple[int, int]:
    """Fast line count of OBJ vertices and triangular faces (no numpy load)."""
    verts = faces = 0
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                verts += 1
            elif line.startswith("f "):
                faces += 1
    return verts, faces


def contact_link_mesh_sizes(
    asset_dir: Path,
    *,
    joint_name: str,
    link_name: str,
) -> list[tuple[str, int, int]]:
    """Return (mesh_filename, n_verts, n_faces) for the affordance child link."""
    from affordance.contact_point import _resolve_movable_joint
    from affordance.urdf_kinematics import load_urdf_root, movable_joints, parse_joints

    asset_dir = Path(asset_dir)
    root = load_urdf_root(asset_dir)
    joints = parse_joints(root)
    movable = movable_joints(joints)
    joint = _resolve_movable_joint(
        asset_dir,
        movable,
        root,
        joint_name=joint_name,
        link_name=link_name,
    )
    link_el = next((el for el in root.findall("link") if el.get("name") == joint.child_link), None)
    if link_el is None:
        return []
    sizes: list[tuple[str, int, int]] = []
    for inst in load_mesh_instances_for_link(link_el):
        mesh_path = asset_dir / inst.filename
        if not mesh_path.is_file():
            mesh_path = asset_dir / Path(inst.filename).name
        if not mesh_path.is_file():
            continue
        v, f = count_obj_mesh_size(mesh_path)
        sizes.append((mesh_path.name, v, f))
    return sizes


def is_dense_contact_link_mesh(
    asset_dir: Path,
    *,
    joint_name: str,
    link_name: str,
    vert_threshold: int = SKIP_DENSE_VERT_THRESHOLD,
    face_threshold: int = SKIP_DENSE_FACE_THRESHOLD,
) -> tuple[bool, str]:
    """True when any contact-link mesh exceeds density thresholds."""
    worst: tuple[str, int, int] | None = None
    for name, v, f in contact_link_mesh_sizes(
        asset_dir, joint_name=joint_name, link_name=link_name
    ):
        if v > vert_threshold or f > face_threshold:
            if worst is None or v > worst[1]:
                worst = (name, v, f)
    if worst is None:
        return False, ""
    name, v, f = worst
    return True, f"{name} verts={v} faces={f}"


@dataclass(frozen=True)
class MeshInstance:
    filename: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray


def _parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _parse_rpy(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _load_obj_faces(path: Path) -> np.ndarray:
    faces: list[list[int]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("f "):
                continue
            idx: list[int] = []
            for token in line.split()[1:]:
                idx.append(int(token.split("/")[0]) - 1)
            if len(idx) >= 3:
                faces.append(idx[:3])
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def _estimate_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return _estimate_normals_knn(vertices)
    normals = np.zeros_like(vertices)
    counts = np.zeros(len(vertices), dtype=np.float64)
    for i0, i1, i2 in faces:
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        n = np.cross(v1 - v0, v2 - v0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue
        n = n / norm
        for idx in (i0, i1, i2):
            normals[idx] += n
            counts[idx] += 1.0
    mask = counts > 0
    normals[mask] /= counts[mask, None]
    bad = ~mask
    if np.any(bad):
        normals[bad] = _estimate_normals_knn(vertices[bad], reference=vertices)
    return normals


def _estimate_normals_knn(
    points: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    k: int = 12,
) -> np.ndarray:
    ref = reference if reference is not None else points
    centroid = ref.mean(axis=0)
    if len(points) == 1:
        n = points[0] - centroid
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        return (n / norm).reshape(1, 3)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        n = points - centroid
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.clip(norms, 1e-9, None)

    tree = cKDTree(ref)
    normals = np.zeros_like(points)
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=min(k, len(ref)))
        neighbors = ref[np.atleast_1d(idx)]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        n = vh[-1]
        if np.dot(n, p - centroid) < 0:
            n = -n
        normals[i] = n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(norms, 1e-9, None)


def load_mesh_instances_for_link(link_el) -> list[MeshInstance]:
    instances: list[MeshInstance] = []
    for tag in ("visual", "collision"):
        for block in link_el.findall(tag):
            geom = block.find("geometry/mesh")
            if geom is None:
                continue
            filename = geom.get("filename")
            if not filename:
                continue
            origin = block.find("origin")
            xyz = _parse_xyz(origin.get("xyz") if origin is not None else None)
            rpy = _parse_rpy(origin.get("rpy") if origin is not None else None)
            instances.append(MeshInstance(filename=filename, origin_xyz=xyz, origin_rpy=rpy))
    dedup: dict[tuple[str, tuple[float, ...], tuple[float, ...]], MeshInstance] = {}
    for inst in instances:
        key = (
            inst.filename,
            tuple(round(float(x), 8) for x in inst.origin_xyz),
            tuple(round(float(x), 8) for x in inst.origin_rpy),
        )
        dedup[key] = inst
    return list(dedup.values())


def load_link_surface_in_link_frame(
    asset_dir: Path,
    instances: list[MeshInstance],
    *,
    max_points: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and outward-oriented normals in the child link frame."""
    asset_dir = Path(asset_dir)
    chunks_v: list[np.ndarray] = []
    chunks_n: list[np.ndarray] = []
    for inst in instances:
        mesh_path = asset_dir / inst.filename
        if not mesh_path.is_file():
            mesh_path = asset_dir / Path(inst.filename).name
        if not mesh_path.is_file():
            continue
        verts = load_obj_vertices(mesh_path)
        faces = _load_obj_faces(mesh_path)
        if len(verts) > DENSE_VERT_THRESHOLD:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(verts), size=max_points, replace=False)
            verts = verts[idx]
            normals = _estimate_normals_knn(verts)
        elif len(faces):
            normals = _estimate_vertex_normals(verts, faces)
        else:
            normals = _estimate_normals_knn(verts)
        if np.linalg.norm(inst.origin_xyz) > 1e-9 or np.linalg.norm(inst.origin_rpy) > 1e-9:
            from affordance.urdf_kinematics import origin_to_matrix, transform_points

            mat = origin_to_matrix(inst.origin_xyz, inst.origin_rpy)
            rot = mat[:3, :3]
            trans = mat[:3, 3]
            verts = transform_points(verts, rot, trans)
            normals = transform_points(normals, rot, np.zeros(3))
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.clip(norms, 1e-9, None)
        chunks_v.append(verts)
        chunks_n.append(normals)

    if not chunks_v:
        link_name = instances[0].filename if instances else "?"
        raise FileNotFoundError(f"No mesh loaded for link meshes under {asset_dir} ({link_name})")

    points = np.vstack(chunks_v)
    normals = np.vstack(chunks_n)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        normals = normals[idx]
    return points, normals