"""FITR 4.1.2 contact point derivation (Phi_aff)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affordance.contact_axis import (
    CATEGORY_DEFAULT_AXIS,
    XY_PLANE_CATEGORIES,
    contact_axis_for_prismatic,
    contact_axis_for_revolute,
)
from affordance.mesh_surface import load_link_surface_in_link_frame, load_mesh_instances_for_link
from affordance.urdf_kinematics import (
    JointRecord,
    joint_axis_in_base,
    link_pose_in_base,
    load_urdf_root,
    movable_joints,
    ordered_movable_joints,
    parse_joints,
    transform_direction,
    transform_points,
)
from prompt.load_prompt import list_movable_joints


@dataclass(frozen=True)
class ContactPointResult:
    position_xyz: list[float]
    contact_axis_xyz: list[float]
    source: str
    lever_arm: float | None = None


def _normalize(v: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        if fallback is not None:
            return _normalize(fallback)
        raise ValueError("zero-length vector")
    return v / n


def _round_vec(v: np.ndarray, ndigits: int = 6) -> list[float]:
    return [round(float(x), ndigits) for x in v.tolist()]


def _orient_normal_outward(normal: np.ndarray, point: np.ndarray, reference: np.ndarray) -> np.ndarray:
    outward = point - reference
    if float(np.linalg.norm(normal)) < 1e-9:
        return _normalize(outward)
    n = _normalize(normal)
    if np.dot(n, outward) < 0:
        n = -n
    return n


def _project_contact_axis(axis: np.ndarray, category: str, motion_type: str, hinge_axis: np.ndarray) -> np.ndarray:
    axis = _normalize(axis)
    if motion_type == "prismatic":
        return contact_axis_for_prismatic(axis, category)
    if category in XY_PLANE_CATEGORIES:
        xy = np.array([axis[0], axis[1], 0.0], dtype=np.float64)
        if float(np.linalg.norm(xy)) > 1e-6:
            return _normalize(xy)
    return contact_axis_for_revolute(hinge_axis, category)


def _lever_arm(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = _normalize(axis)
    rel = points - origin.reshape(1, 3)
    along = rel @ axis
    perp = rel - along.reshape(-1, 1) * axis.reshape(1, 3)
    return np.linalg.norm(perp, axis=1)


def _filter_graspable_surface(
    points: np.ndarray,
    normals: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outward = points - reference.reshape(1, 3)
    mask = np.array([np.dot(normals[i], outward[i]) > 0.0 for i in range(len(points))], dtype=bool)
    if not np.any(mask):
        centroid = points.mean(axis=0)
        mask = np.array([np.dot(normals[i], points[i] - centroid) > 0.0 for i in range(len(points))], dtype=bool)
    if not np.any(mask):
        return points, normals
    return points[mask], normals[mask]


# Revolute panels (monitor/laptop screen): contact at top-center of front face, not max lever.
PANEL_TILT_CATEGORIES = {"Display", "Laptop"}


def _panel_tilt_contact(
    points_base: np.ndarray,
    normals_base: np.ndarray,
    category: str,
) -> ContactPointResult:
    """Top-center of outward front face for tiltable screens."""
    default_axis = np.array(CATEGORY_DEFAULT_AXIS[category], dtype=np.float64)
    normal_scores = normals_base @ _normalize(default_axis)
    front_mask = normal_scores > 0.5
    if not np.any(front_mask):
        front_mask = normal_scores > 0.0
    if not np.any(front_mask):
        front_mask = np.ones(len(points_base), dtype=bool)

    front_points = points_base[front_mask]
    x_mid = 0.5 * (float(front_points[:, 0].min()) + float(front_points[:, 0].max()))
    scores = front_points[:, 2] - 0.5 * np.abs(front_points[:, 0] - x_mid)
    idx = int(np.argmax(scores))
    position = front_points[idx]
    contact_axis = _normalize(default_axis)
    return ContactPointResult(
        position_xyz=_round_vec(position),
        contact_axis_xyz=_round_vec(contact_axis),
        source="panel_tilt_top_center",
        lever_arm=None,
    )

def _door_frame(
    points: np.ndarray,
    origin: np.ndarray,
    hinge_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Hinge-aligned frame: hinge_axis (vertical), width_dir (swing), height_mid."""
    hinge = _normalize(hinge_axis)
    rel = points - origin.reshape(1, 3)
    along = rel @ hinge
    perp = rel - along.reshape(-1, 1) * hinge.reshape(1, 3)
    if len(perp) < 3:
        width = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(width, hinge))) > 0.9:
            width = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        width = _normalize(width - np.dot(width, hinge) * hinge)
        height_mid = float(along.mean())
        return hinge, width, height_mid

    cov = np.cov(perp.T)
    evals, evecs = np.linalg.eigh(cov)
    width = _normalize(evecs[:, int(np.argmax(evals))])
    height_mid = 0.5 * (float(along.min()) + float(along.max()))
    return hinge, width, height_mid


def _door_edge_handle_score(
    points: np.ndarray,
    indices: np.ndarray,
    origin: np.ndarray,
    hinge_axis: np.ndarray,
) -> float:
    """Higher on edges with handle-like geometry (variation along hinge / depth)."""
    if len(indices) == 0:
        return -1.0
    p = points[indices]
    hinge = _normalize(hinge_axis)
    along = (p - origin.reshape(1, 3)) @ hinge
    return float(np.std(along) + np.std(p[:, 2]) + 0.5 * np.std(p[:, 1]))


def _door_revolute_contact(
    points_base: np.ndarray,
    normals_base: np.ndarray,
    origin: np.ndarray,
    hinge_axis: np.ndarray,
    category: str,
) -> ContactPointResult:
    """
    Revolute door: max lever on graspable surface, tie-break to
    free edge (farthest from vertical hinge) + height midline along hinge.

    When both lateral edges tie on lever (hinge origin centered on leaf),
    keep points on the edge with higher handle-like geometric variation.
    """
    points, normals = _filter_graspable_surface(points_base, normals_base, origin)
    arms = _lever_arm(points, origin, hinge_axis)
    max_arm = float(np.max(arms))
    tol = max(max_arm * 0.02, 0.01)
    near = np.where(arms >= max_arm - tol)[0]
    hinge_n = _normalize(hinge_axis)
    _, width_dir, height_mid = _door_frame(points, origin, hinge_axis)
    width_proj_signed = (points - origin.reshape(1, 3)) @ width_dir
    pos = near[width_proj_signed[near] >= 0]
    neg = near[width_proj_signed[near] < 0]
    near_work = near
    if len(pos) and len(neg):
        pos_arm = float(np.max(arms[pos]))
        neg_arm = float(np.max(arms[neg]))
        if abs(pos_arm - neg_arm) <= tol:
            pos_score = _door_edge_handle_score(points, pos, origin, hinge_axis)
            neg_score = _door_edge_handle_score(points, neg, origin, hinge_axis)
            near_work = pos if pos_score >= neg_score else neg

    if len(near_work) <= 1:
        idx = int(np.argmax(arms))
    else:
        rel = points[near_work] - origin.reshape(1, 3)
        width_proj = np.abs(rel @ width_dir)
        max_width = float(np.max(width_proj))
        edge = near_work[width_proj >= max_width - max(max_width * 0.02, 0.01)]
        if len(edge) == 0:
            edge = near_work
        along_edge = (points[edge] - origin.reshape(1, 3)) @ hinge_n
        idx = int(edge[np.argmin(np.abs(along_edge - height_mid))])

    position = points[idx]
    normal = _orient_normal_outward(normals[idx], position, origin)
    contact_axis = _project_contact_axis(normal, category, "revolute", hinge_axis)
    return ContactPointResult(
        position_xyz=_round_vec(position),
        contact_axis_xyz=_round_vec(contact_axis),
        source="door_hinge_max_lever",
        lever_arm=float(arms[idx]),
    )


def _revolute_contact(
    points_base: np.ndarray,
    normals_base: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    category: str,
) -> ContactPointResult:
    points, normals = _filter_graspable_surface(points_base, normals_base, origin)
    arms = _lever_arm(points, origin, axis)
    max_arm = float(np.max(arms))
    tol = max(max_arm * 0.02, 0.01)
    near = np.where(arms >= max_arm - tol)[0]
    if len(near) > 1:
        # Prefer top edge + lateral center (handle strip) over corner vertices.
        scores = points[near, 2] - 0.35 * np.abs(points[near, 1])
        idx = int(near[np.argmax(scores)])
    else:
        idx = int(np.argmax(arms))
    position = points[idx]
    normal = _orient_normal_outward(normals[idx], position, origin)
    contact_axis = _project_contact_axis(normal, category, "revolute", axis)
    return ContactPointResult(
        position_xyz=_round_vec(position),
        contact_axis_xyz=_round_vec(contact_axis),
        source="revolute_max_lever",
        lever_arm=float(arms[idx]),
    )


def _prismatic_contact(
    points_base: np.ndarray,
    normals_base: np.ndarray,
    axis: np.ndarray,
    category: str,
) -> ContactPointResult:
    axis = _normalize(axis)
    centroid = points_base.mean(axis=0)
    proj = (points_base - centroid.reshape(1, 3)) @ axis
    half_extent = float((proj.max() - proj.min()) * 0.5)
    position = centroid + half_extent * axis

    front_mask = proj >= float(np.percentile(proj, 75))
    if np.any(front_mask):
        front_points = points_base[front_mask]
        front_normals = normals_base[front_mask]
    else:
        front_points = points_base
        front_normals = normals_base

    normal_scores = front_normals @ axis
    idx = int(np.argmax(normal_scores))
    normal = _orient_normal_outward(front_normals[idx], front_points[idx], centroid)
    if float(np.linalg.norm(normal)) < 1e-9:
        normal = axis.copy()
    if float(np.dot(normal, axis)) < 0:
        normal = -normal
    contact_axis = _project_contact_axis(normal, category, "prismatic", axis)
    return ContactPointResult(
        position_xyz=_round_vec(position),
        contact_axis_xyz=_round_vec(contact_axis),
        source="prismatic_front_center",
        lever_arm=half_extent,
    )


def _resolve_movable_joint(
    asset_dir: Path,
    movable: dict[str, JointRecord],
    root,
    *,
    joint_name: str,
    link_name: str,
) -> JointRecord:
    """Match URDF joint; info1 link_name may be semantic (mobility_v2) not URDF child id."""
    joint = movable.get(joint_name)
    if joint is not None:
        return joint

    specs = list_movable_joints(asset_dir)
    ordered = ordered_movable_joints(root)
    for i, spec in enumerate(specs):
        if spec.joint_name == joint_name or (link_name and spec.link_name == link_name):
            if i < len(ordered):
                return ordered[i]

    if link_name:
        matched = [j for j in movable.values() if j.child_link == link_name]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            for j in matched:
                if j.name == joint_name:
                    return j
    raise KeyError(f"Joint {joint_name} / link {link_name} not found in URDF movable joints")


def compute_contact_point(
    asset_dir: Path,
    *,
    joint_name: str,
    link_name: str,
    motion_type: str,
    category: str,
) -> ContactPointResult:
    """
    Derive contact point and axis per FITR 4.1.2 in URDF base (calibration) frame.

    Revolute: argmax lever arm on outward graspable surface.
    Prismatic: front-face center along joint axis.
    """
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

    link_el = None
    for el in root.findall("link"):
        if el.get("name") == joint.child_link:
            link_el = el
            break
    if link_el is None:
        raise KeyError(f"Link {joint.child_link} not found in URDF")

    instances = load_mesh_instances_for_link(link_el)
    points_link, normals_link = load_link_surface_in_link_frame(asset_dir, instances)

    child_pose = link_pose_in_base(root, joints, joint.child_link)
    rot = child_pose[:3, :3]
    trans = child_pose[:3, 3]
    points_base = transform_points(points_link, rot, trans)
    normals_base = transform_points(normals_link, rot, np.zeros(3))
    norms = np.linalg.norm(normals_base, axis=1, keepdims=True)
    normals_base = normals_base / np.clip(norms, 1e-9, None)

    origin, axis = joint_axis_in_base(joint, root, joints)
    motion = motion_type or joint.motion_type
    if motion == "prismatic":
        return _prismatic_contact(points_base, normals_base, axis, category)
    if category in PANEL_TILT_CATEGORIES:
        return _panel_tilt_contact(points_base, normals_base, category)
    if category == "Door":
        return _door_revolute_contact(points_base, normals_base, origin, axis, category)
    return _revolute_contact(points_base, normals_base, origin, axis, category)


def compute_contact_points_for_asset(
    asset_dir: Path,
    *,
    category: str,
    joints: list[dict[str, str]],
) -> dict[str, ContactPointResult]:
    out: dict[str, ContactPointResult] = {}
    for spec in joints:
        joint_name = spec["joint_name"]
        link_name = spec.get("link_name", "")
        motion_type = spec.get("motion_type", "revolute")
        out[joint_name] = compute_contact_point(
            asset_dir,
            joint_name=joint_name,
            link_name=link_name,
            motion_type=motion_type,
            category=category,
        )
    return out
