"""Joint-space linear interpolation for auto trajectory collection MVP."""

from __future__ import annotations


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_joint_rad(
    start: tuple[float, ...],
    end: tuple[float, ...],
    t: float,
) -> tuple[float, ...]:
    if len(start) != len(end):
        raise ValueError(f"Joint dim mismatch: {len(start)} vs {len(end)}")
    return tuple(lerp(float(s), float(e), t) for s, e in zip(start, end))


def interpolate_joint_segment(
    start: tuple[float, ...],
    end: tuple[float, ...],
    num_steps: int,
) -> list[tuple[float, ...]]:
    if num_steps <= 0:
        return [end]
    return [lerp_joint_rad(start, end, (i + 1) / num_steps) for i in range(num_steps)]


def build_joint_trajectory(
    start: tuple[float, ...],
    waypoints: tuple[tuple[float, ...], ...],
    steps_per_segment: int,
) -> list[tuple[float, ...]]:
    trajectory: list[tuple[float, ...]] = []
    current = start
    for waypoint in waypoints:
        segment = interpolate_joint_segment(current, waypoint, steps_per_segment)
        trajectory.extend(segment)
        current = waypoint
    return trajectory


def lerp_gripper(start: float, end: float, t: float) -> float:
    return lerp(float(start), float(end), t)
