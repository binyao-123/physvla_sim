"""ArticuBot-style contact/grasp candidate sampling on articulated link meshes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from mesh_utils import (
    distance_to_axis,
    farthest_point_sampling,
    load_mesh_vertices_and_normals,
    subsample_points,
)
from reference.grasping_utils import align_gripper_z_with_normal, rotation_matrix_to_wxyz
from reference.opening_kinematics import compose_pose
from task_registry import PHYSVLA_ASSETS_DIR


@dataclass(frozen=True)
class HingeSpec:
    """Hinge axis defined in movable link local frame (matches URDF joint child frame)."""

    origin: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class SamplingConfig:
    asset_subdir: str
    mesh_filename: str
    movable_link: str
    base_link: str
    hinge: HingeSpec
    mesh_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    num_fps_points: int = 15
    num_yaw_perturbations: int = 8
    max_yaw_perturb_deg: float = 30.0
    approach_offset_m: float = 0.04
    contact_offset_m: float = 0.02
    min_hinge_distance_percentile: float = 60.0
    horizontal_grasp: bool = True
    max_surface_points: int = 4000
    # When set, keep candidates whose EE->contact direction aligns with this world vector.
    approach_direction_world: tuple[float, float, float] | None = None
    min_approach_direction_dot: float = 0.05
    # Prefer live EE->link direction from sim (scene.usd layout) over yaml vector.
    use_scene_approach_direction: bool = True
    # Workspace filters in world frame (derived from open_laptop scene.usd).
    max_contact_world_y_abs_m: float | None = None
    min_contact_world_x_m: float | None = None
    max_contact_world_x_m: float | None = None
    min_contact_world_z_m: float | None = None
    max_contact_world_z_m: float | None = None
    max_contact_dist_from_link_m: float | None = None
    # Push tasks: robot-side lid face sits near link origin in Y (not far from hinge).
    max_contact_link_local_y_m: float | None = None
    max_contact_delta_y_from_link_m: float | None = None
    push_anchor_dist_m: float = 0.10
    # Lid push face in link_1 frame (from scene.usd + keyboard ref); not EE->hinge ray.
    push_contact_offset_link: tuple[float, float, float] | None = None
    # Gripper orientation at contact, link_1 frame (w,x,y,z). Required for yaml_handle push.
    contact_quat_link: tuple[float, float, float, float] | None = None
    use_push_anchor_fallback: bool = True
    # Optional demo contact — tie-break only, not a hard filter.
    reference_contact_world: tuple[float, float, float] | None = None
    reference_contact_max_dist_m: float | None = None
    # Optional two-point hinge-arc fit for yaml_handle when joint_1 starts away from
    # the reference calibration angle. Missing values keep the legacy link-local path.
    push_contact_reference_joint_deg: float = 15.0
    push_contact_joint_fit_deg: float | None = None
    push_contact_joint_fit_world: tuple[float, float, float] | None = None
    push_contact_joint_fit_range_deg: tuple[float, float] | None = None
    push_contact_joint_arc_points: tuple[tuple[float, float, float, float], ...] = ()
    # Optional push arc endpoint below/above calibrated samples (e.g. -30° for extra close lead).
    push_contact_joint_arc_target_deg: float | None = None
    # Optional yaw anchors [yaw_deg, x, y, z, qw, qx, qy, qz] for calibrated EE/handle contact.
    yaw_contact_anchors: tuple[tuple[float, float, float, float, float, float, float, float], ...] = ()


@dataclass(frozen=True)
class ContactCandidate:
    contact_pos_link: np.ndarray
    approach_pos_link: np.ndarray
    quat_wxyz_link: tuple[float, float, float, float]
    surface_normal_link: np.ndarray
    fps_index: int
    yaw_index: int


def resolve_asset_mesh_path(config: SamplingConfig) -> Path:
    return (PHYSVLA_ASSETS_DIR / config.asset_subdir / config.mesh_filename).resolve()


def load_link_surface(config: SamplingConfig) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = resolve_asset_mesh_path(config)
    points, normals = load_mesh_vertices_and_normals(mesh_path)
    origin = np.asarray(config.mesh_origin, dtype=np.float64)
    if np.linalg.norm(origin) > 1e-9:
        points = points + origin.reshape(1, 3)
    return subsample_points(points, normals, max_points=config.max_surface_points)


def _wxyz_to_rot(quat_wxyz: tuple[float, float, float, float]):
    from scipy.spatial.transform import Rotation as R

    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w])


def link_local_to_world_position(
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    local_pos: np.ndarray,
) -> np.ndarray:
    """Map a point from link-local frame to world (matches USD link prim frame)."""
    rot = _wxyz_to_rot(link_quat_wxyz)
    return np.asarray(link_pos_world, dtype=np.float64) + rot.apply(np.asarray(local_pos, dtype=np.float64))


def link_local_to_world_direction(
    link_quat_wxyz: tuple[float, float, float, float],
    local_dir: np.ndarray,
) -> np.ndarray:
    """Map a direction from link-local frame to world (rotation only)."""
    rot = _wxyz_to_rot(link_quat_wxyz)
    vec = rot.apply(np.asarray(local_dir, dtype=np.float64))
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-9)


def link_local_axes_world(
    link_quat_wxyz: tuple[float, float, float, float],
) -> dict[str, np.ndarray]:
    """Unit X/Y/Z axes of link prim in world frame."""
    return {
        "x": link_local_to_world_direction(link_quat_wxyz, np.array([1.0, 0.0, 0.0])),
        "y": link_local_to_world_direction(link_quat_wxyz, np.array([0.0, 1.0, 0.0])),
        "z": link_local_to_world_direction(link_quat_wxyz, np.array([0.0, 0.0, 1.0])),
    }


def compute_link_local_probe_point(
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    local_offset: tuple[float, float, float],
) -> np.ndarray:
    """World position of link origin + local_offset (e.g. Z +0.2 m sanity check)."""
    return link_local_to_world_position(link_pos_world, link_quat_wxyz, np.asarray(local_offset, dtype=np.float64))


def candidate_push_normal_world(
    candidate: ContactCandidate,
    link_quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    """Outward surface normal in world; push direction is opposite."""
    return link_local_to_world_direction(link_quat_wxyz, candidate.surface_normal_link)


def filter_far_from_hinge(
    points: np.ndarray,
    normals: np.ndarray,
    hinge: HingeSpec,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep surface points far from hinge axis (ArticuBot grasp heuristic).

    For push tasks set ``min_hinge_distance_percentile: 0`` — the reachable push
    face on this laptop lid is near the hinge, not on the far edge.
    """
    if percentile <= 0.0:
        return points, normals

    origin = np.asarray(hinge.origin, dtype=np.float64)
    axis = np.asarray(hinge.axis, dtype=np.float64)
    dists = distance_to_axis(points, origin, axis)
    thresh = float(np.percentile(dists, percentile))
    mask = dists >= thresh
    if not np.any(mask):
        mask = dists >= float(np.median(dists))
    return points[mask], normals[mask]


def _orient_normal_outward(normal: np.ndarray, point: np.ndarray, hinge_origin: np.ndarray) -> np.ndarray:
    n = normal / max(np.linalg.norm(normal), 1e-9)
    outward = point - hinge_origin
    if np.dot(n, outward) < 0:
        n = -n
    return n


def sample_contact_candidates(
    config: SamplingConfig,
    rng: np.random.Generator | None = None,
) -> list[ContactCandidate]:
    """Paper-style: FPS m1=15 positions × m2=8 yaw perturbations on link surface."""
    rng = rng or np.random.default_rng()
    points, normals = load_link_surface(config)
    points, normals = filter_far_from_hinge(
        points, normals, config.hinge, config.min_hinge_distance_percentile
    )

    num_fps = min(config.num_fps_points, len(points))
    fps_idx = farthest_point_sampling(points, num_fps)
    fps_points = points[fps_idx]
    fps_normals = normals[fps_idx]

    if config.num_yaw_perturbations <= 1:
        yaw_angles = [0.0]
    else:
        yaw_angles = np.linspace(
            -config.max_yaw_perturb_deg,
            config.max_yaw_perturb_deg,
            config.num_yaw_perturbations,
        )

    hinge_origin = np.asarray(config.hinge.origin, dtype=np.float64)
    candidates: list[ContactCandidate] = []
    for i, (pt, raw_n) in enumerate(zip(fps_points, fps_normals)):
        outward_n = _orient_normal_outward(raw_n, pt, hinge_origin)
        # Push direction: into lid (opposite outward normal).
        push_normal = -outward_n
        for j, yaw_deg in enumerate(yaw_angles):
            rot = align_gripper_z_with_normal(
                push_normal,
                horizontal=config.horizontal_grasp,
                yaw_perturb_deg=float(yaw_deg),
            )
            quat = rotation_matrix_to_wxyz(rot)
            contact = pt + outward_n * config.contact_offset_m
            approach = pt + outward_n * config.approach_offset_m
            candidates.append(
                ContactCandidate(
                    contact_pos_link=contact,
                    approach_pos_link=approach,
                    quat_wxyz_link=quat,
                    surface_normal_link=outward_n,
                    fps_index=i,
                    yaw_index=j,
                )
            )
    rng.shuffle(candidates)
    return candidates


def filter_candidates_by_local_y(
    candidates: list[ContactCandidate],
    config: SamplingConfig,
) -> list[ContactCandidate]:
    """Deprecated: kept for yaml compat; prefer world-frame filters."""
    return candidates


def scene_approach_direction(
    ee_pos_world: np.ndarray,
    link_pos_world: np.ndarray,
    config: SamplingConfig,
) -> np.ndarray:
    """Unit vector EE -> link (matches scene.usd default layout)."""
    if config.use_scene_approach_direction:
        vec = np.asarray(link_pos_world, dtype=np.float64) - np.asarray(ee_pos_world, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            return vec / norm

    direction = config.approach_direction_world
    if direction is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-9)


def push_contact_surface_world(
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    config: SamplingConfig,
    *,
    ee_pos_world: np.ndarray | None = None,
) -> np.ndarray:
    """World position on the robot-side lid push face."""
    if config.push_contact_offset_link is not None:
        return link_local_to_world_position(
            link_pos_world,
            link_quat_wxyz,
            np.asarray(config.push_contact_offset_link, dtype=np.float64),
        )

    approach_dir = scene_approach_direction(
        ee_pos_world if ee_pos_world is not None else link_pos_world,
        link_pos_world,
        config,
    )
    return np.asarray(link_pos_world, dtype=np.float64) + approach_dir * float(config.push_anchor_dist_m)


def push_anchor_world(
    link_pos_world: np.ndarray,
    approach_dir: np.ndarray,
    config: SamplingConfig,
) -> np.ndarray:
    """Deprecated alias — prefer push_contact_surface_world with link-local offset."""
    return np.asarray(link_pos_world, dtype=np.float64) + approach_dir * float(config.push_anchor_dist_m)


def make_push_anchor_candidate(
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    ee_pos_world: np.ndarray,
    config: SamplingConfig,
) -> ContactCandidate:
    """Synthetic contact on the robot-side lid face when mesh FPS misses."""
    link_pos = np.asarray(link_pos_world, dtype=np.float64)
    ee_pos = np.asarray(ee_pos_world, dtype=np.float64)
    rot = _wxyz_to_rot(link_quat_wxyz)

    if config.push_contact_offset_link is not None:
        surface_l = np.asarray(config.push_contact_offset_link, dtype=np.float64)
        surface_w = link_local_to_world_position(link_pos, link_quat_wxyz, surface_l)
    else:
        approach_dir = scene_approach_direction(ee_pos, link_pos, config)
        surface_w = link_pos + approach_dir * float(config.push_anchor_dist_m)
        surface_l = rot.inv().apply(surface_w - link_pos)

    outward_w = ee_pos - surface_w
    outward_w = outward_w / max(float(np.linalg.norm(outward_w)), 1e-9)
    outward_l = rot.inv().apply(outward_w)
    outward_l = outward_l / max(float(np.linalg.norm(outward_l)), 1e-9)
    push_normal = -outward_l
    grip_rot = align_gripper_z_with_normal(
        push_normal,
        horizontal=config.horizontal_grasp,
        yaw_perturb_deg=0.0,
    )
    quat = rotation_matrix_to_wxyz(grip_rot)
    contact = surface_l + outward_l * config.contact_offset_m
    approach = surface_l + outward_l * config.approach_offset_m
    return ContactCandidate(
        contact_pos_link=contact,
        approach_pos_link=approach,
        quat_wxyz_link=quat,
        surface_normal_link=outward_l,
        fps_index=-1,
        yaw_index=0,
    )


def contact_passes_sanity(
    candidate: ContactCandidate,
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    config: SamplingConfig,
) -> bool:
    """Reject contacts on wrong mesh sheet (e.g. Y≈1 m away from link prim)."""
    _, contact_w, _ = link_to_world_candidate(candidate, link_pos_world, link_quat_wxyz)
    link_pos = np.asarray(link_pos_world, dtype=np.float64)

    max_y_l = config.max_contact_link_local_y_m
    if max_y_l is not None and abs(float(candidate.contact_pos_link[1])) > float(max_y_l):
        return False

    max_dy = config.max_contact_delta_y_from_link_m
    if max_dy is not None and abs(float(contact_w[1] - link_pos[1])) > float(max_dy):
        return False

    min_z = config.min_contact_world_z_m
    if min_z is not None and float(contact_w[2]) < float(min_z):
        return False
    max_z = config.max_contact_world_z_m
    if max_z is not None and float(contact_w[2]) > float(max_z):
        return False

    return True


def filter_candidates_workspace(
    candidates: list[ContactCandidate],
    config: SamplingConfig,
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    ee_pos_world: np.ndarray,
) -> list[ContactCandidate]:
    """Keep contacts on robot-reachable push face using world-frame geometry."""
    if not candidates:
        return candidates

    kept = list(candidates)
    max_y = config.max_contact_world_y_abs_m
    min_x = config.min_contact_world_x_m
    max_x = config.max_contact_world_x_m
    min_z = config.min_contact_world_z_m
    max_z = config.max_contact_world_z_m
    max_link_dist = config.max_contact_dist_from_link_m
    link_pos = np.asarray(link_pos_world, dtype=np.float64)

    if max_y is not None:
        kept = [
            c
            for c in kept
            if abs(float(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1][1])) <= float(max_y)
        ]
    if min_x is not None:
        kept = [
            c
            for c in kept
            if float(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1][0]) >= float(min_x)
        ]
    if max_x is not None:
        kept = [
            c
            for c in kept
            if float(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1][0]) <= float(max_x)
        ]
    if min_z is not None:
        kept = [
            c
            for c in kept
            if float(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1][2]) >= float(min_z)
        ]
    if max_z is not None:
        kept = [
            c
            for c in kept
            if float(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1][2]) <= float(max_z)
        ]
    if max_link_dist is not None:
        kept = [
            c
            for c in kept
            if float(
                np.linalg.norm(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1] - link_pos)
            )
            <= float(max_link_dist)
        ]

    max_y_l = config.max_contact_link_local_y_m
    if max_y_l is not None:
        kept = [c for c in kept if abs(float(c.contact_pos_link[1])) <= float(max_y_l)]

    max_dy = config.max_contact_delta_y_from_link_m
    if max_dy is not None:
        kept = [
            c
            for c in kept
            if abs(
                float(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1][1] - link_pos[1])
            )
            <= float(max_dy)
        ]

    ref = (
        np.asarray(config.reference_contact_world, dtype=np.float64)
        if config.reference_contact_world is not None
        else None
    )
    max_ref_dist = config.reference_contact_max_dist_m
    if ref is not None and max_ref_dist is not None:
        kept = [
            c
            for c in kept
            if float(np.linalg.norm(link_to_world_candidate(c, link_pos_world, link_quat_wxyz)[1] - ref))
            <= float(max_ref_dist)
        ]

    approach_dir = scene_approach_direction(ee_pos_world, link_pos_world, config)
    min_dot = float(config.min_approach_direction_dot)
    dir_kept: list[ContactCandidate] = []
    for candidate in kept:
        _, contact_w, _ = link_to_world_candidate(candidate, link_pos_world, link_quat_wxyz)
        vec = contact_w - ee_pos_world
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            continue
        if float(np.dot(vec / dist, approach_dir)) >= min_dot:
            dir_kept.append(candidate)
    kept = dir_kept

    return kept


def filter_candidates_by_approach_direction(
    candidates: list[ContactCandidate],
    config: SamplingConfig,
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    ee_pos_world: np.ndarray,
) -> list[ContactCandidate]:
    """Filter to robot-side push face; return empty if none pass (caller uses anchor fallback)."""
    filtered = filter_candidates_workspace(
        candidates, config, link_pos_world, link_quat_wxyz, ee_pos_world
    )
    if filtered:
        return filtered

    print(
        "[WARN] Contact filter removed all mesh candidates; relaxing min_approach_direction_dot "
        f"from {config.min_approach_direction_dot} to 0.35."
    )
    relaxed = replace(config, min_approach_direction_dot=0.35)
    filtered = filter_candidates_workspace(
        candidates, relaxed, link_pos_world, link_quat_wxyz, ee_pos_world
    )
    if filtered:
        return filtered

    print("[WARN] No mesh contacts passed workspace filter; push-anchor fallback required.")
    return []


def rank_contact_candidates(
    candidates: list[ContactCandidate],
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    ee_pos_world: np.ndarray,
    config: SamplingConfig,
) -> list[ContactCandidate]:
    """Rank for batch sampling: top Z, near push anchor, optional demo tie-break."""
    if not candidates:
        return []

    anchor = push_contact_surface_world(
        link_pos_world, link_quat_wxyz, config, ee_pos_world=ee_pos_world
    )
    ref = (
        np.asarray(config.reference_contact_world, dtype=np.float64)
        if config.reference_contact_world is not None
        else None
    )

    def sort_key(candidate: ContactCandidate) -> tuple[float, float, float, float]:
        _, contact_w, _ = link_to_world_candidate(candidate, link_pos_world, link_quat_wxyz)
        anchor_dist = float(np.linalg.norm(contact_w - anchor))
        ref_dist = float(np.linalg.norm(contact_w - ref)) if ref is not None else 0.0
        ee_dist = float(np.linalg.norm(contact_w - ee_pos_world))
        # Prefer contacts nearest scene push anchor (not highest Z — that picks wrong mesh sheet).
        return (anchor_dist, ref_dist, ee_dist, -float(contact_w[2]))

    ranked = sorted(candidates, key=sort_key)

    # One yaw per FPS point — keep best-ranked per surface site.
    seen_fps: set[int] = set()
    deduped: list[ContactCandidate] = []
    for candidate in ranked:
        if candidate.fps_index in seen_fps:
            continue
        seen_fps.add(candidate.fps_index)
        deduped.append(candidate)
    return deduped


def prepare_ranked_contact_candidates(
    config: SamplingConfig,
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    ee_pos_world: np.ndarray,
    rng: np.random.Generator | None = None,
) -> list[ContactCandidate]:
    """Sample 15×8 mesh contacts, filter by scene workspace, rank for batch try."""
    raw = sample_contact_candidates(config, rng=rng)
    filtered = filter_candidates_by_approach_direction(
        raw, config, link_pos_world, link_quat_wxyz, ee_pos_world
    )
    filtered = [c for c in filtered if contact_passes_sanity(c, link_pos_world, link_quat_wxyz, config)]

    if config.use_push_anchor_fallback:
        anchor = make_push_anchor_candidate(
            link_pos_world, link_quat_wxyz, ee_pos_world, config
        )
        if not filtered:
            print("[INFO] Using scene push-anchor synthetic contact (no valid mesh FPS).")
            filtered = [anchor]
        else:
            filtered = [anchor] + filtered

    return rank_contact_candidates(
        filtered, link_pos_world, link_quat_wxyz, ee_pos_world, config
    )


def select_top_contact_candidate(
    candidates: list[ContactCandidate],
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    config: SamplingConfig,
    ee_pos_world: np.ndarray | None = None,
) -> ContactCandidate | None:
    """Best contact for probe / first batch attempt."""
    if not candidates:
        return None
    if ee_pos_world is None:
        ranked = rank_contact_candidates(
            candidates, link_pos_world, link_quat_wxyz, link_pos_world, config
        )
    else:
        ranked = rank_contact_candidates(
            candidates, link_pos_world, link_quat_wxyz, ee_pos_world, config
        )
    return ranked[0] if ranked else None


def candidate_world_geometry(
    candidate: ContactCandidate,
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
) -> dict[str, np.ndarray | tuple[float, float, float, float]]:
    approach_w, contact_w, quat_w = link_to_world_candidate(candidate, link_pos_world, link_quat_wxyz)
    push_w = -candidate_push_normal_world(candidate, link_quat_wxyz)
    return {
        "approach_w": approach_w,
        "contact_w": contact_w,
        "quat_w": quat_w,
        "push_w": push_w,
    }


def summarize_candidates_world(
    candidates: list[ContactCandidate],
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    *,
    max_rows: int = 12,
) -> list[dict[str, object]]:
    """Return printable world-frame geometry for contact visualization."""
    rows: list[dict[str, object]] = []
    for candidate in candidates[:max_rows]:
        approach_w, contact_w, _ = link_to_world_candidate(candidate, link_pos_world, link_quat_wxyz)
        push_w = -candidate_push_normal_world(candidate, link_quat_wxyz)
        rows.append(
            {
                "fps": candidate.fps_index,
                "yaw": candidate.yaw_index,
                "contact_link": np.round(candidate.contact_pos_link, 4).tolist(),
                "contact_world": np.round(contact_w, 4).tolist(),
                "approach_world": np.round(approach_w, 4).tolist(),
                "push_world": np.round(push_w, 4).tolist(),
            }
        )
    return rows


def link_to_world_candidate(
    candidate: ContactCandidate,
    link_pos_world: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    contact_w, quat_w = compose_pose(
        link_pos_world,
        link_quat_wxyz,
        candidate.contact_pos_link,
        candidate.quat_wxyz_link,
    )
    approach_w, _ = compose_pose(
        link_pos_world,
        link_quat_wxyz,
        candidate.approach_pos_link,
        candidate.quat_wxyz_link,
    )
    return approach_w, contact_w, quat_w
