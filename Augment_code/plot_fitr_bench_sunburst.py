#!/usr/bin/env python3
"""Draw a two-ring sunburst of FITR-Bench.

Inner ring: 12 categories, 20 assets each.
Outer ring: original attributes (pale yellow) and three FITR attributes (orange).

  .venv/bin/python plot_fitr_bench_sunburst.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.patches import Circle, Wedge

AUGMENT_ROOT = Path(__file__).resolve().parent

# FITR-Bench: 12 categories × 20 assets. Names match the FITR-Assets figure.
CATEGORIES = [
    ("Laptop", 20),
    ("Display", 20),
    ("Microwave", 20),
    ("Drawer", 20),
    ("Lamp", 20),
    ("Faucet", 20),
    ("Knife", 20),
    ("Dishwasher", 20),
    ("Door", 20),
    ("Fridge", 20),
    ("Scissors", 20),
    ("Storage", 20),
]

# Tableau qualitative colors, a common palette in papers. Yellow/orange
# are reserved for the outer attribute ring.
CATEGORY_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#499894",
    "#B07AA1",
    "#E15759",
    "#76B7B2",
    "#9D7660",
    "#8CD17D",
    "#D4A6C8",
    "#A0CBE8",
    "#FF9D9A",
    "#79706E",
]

# URDF-Anything+ §3.1: part mesh X^(k) plus joint J^(k) = (type, origin, axis, limits).
ORIGINAL_ATTRS = [
    ("Part Geometry", "#F0DC86"),
    ("Joint Type", "#EBDA7A"),
    ("Joint Origin", "#E6D46E"),
    ("Joint Axis", "#E1CE62"),
    ("Motion Limits", "#DCC856"),
]
FITR_ATTRS = [
    ("Absolute Scale", "#F4A261"),
    ("Handle Contact", "#E07A3D"),
    ("Joint Dynamics", "#C65D24"),
]
ATTRS = [*ORIGINAL_ATTRS, *FITR_ATTRS]

TOTAL_ASSETS = sum(n for _, n in CATEGORIES)
N_CATEGORIES = len(CATEGORIES)

R_HOLE = 0.38
R_INNER = 0.66
R_OUTER = 1.08
START_ANGLE = 90.0


def _contrast_text(hex_color: str) -> str:
    rgb = tuple(int(to_hex(hex_color)[i : i + 2], 16) for i in (1, 3, 5))
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#F8FAF9" if luminance < 150 else "#1A1A1A"


def _wedge_angles(sizes: list[int], start: float = START_ANGLE) -> list[tuple[float, float, float]]:
    angles = []
    cursor = start
    for size in sizes:
        span = 360.0 * size / TOTAL_ASSETS
        theta1 = cursor
        theta2 = cursor + span
        angles.append((theta1, theta2, theta1 + 0.5 * span))
        cursor = theta2
    return angles


def _polar(radius: float, deg: float) -> tuple[float, float]:
    rad = np.deg2rad(deg)
    return float(radius * np.cos(rad)), float(radius * np.sin(rad))


def _tangential_rotation(deg: float) -> float:
    rot = (deg - 90.0) % 360.0
    if 90.0 < rot < 270.0:
        rot = (rot + 180.0) % 360.0
    return rot


def _radial_rotation(deg: float) -> float:
    wrapped = deg % 360.0
    if 90.0 < wrapped <= 270.0:
        return wrapped + 180.0
    return wrapped


def _tangential_label(
    ax,
    text: str,
    radius: float,
    deg: float,
    *,
    fontsize: float,
    color: str,
    weight: str = "medium",
) -> None:
    ax.text(
        *_polar(radius, deg),
        text,
        ha="center",
        va="center",
        rotation=_tangential_rotation(deg),
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        zorder=5,
    )


def _radial_label(
    ax,
    text: str,
    radius: float,
    deg: float,
    *,
    fontsize: float,
    color: str,
    weight: str = "medium",
) -> None:
    ax.text(
        *_polar(radius, deg),
        text,
        ha="center",
        va="center",
        rotation=_radial_rotation(deg),
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        zorder=5,
    )


def draw_sunburst(ax) -> None:
    sizes = [n for _, n in CATEGORIES]
    inner_angles = _wedge_angles(sizes)

    for (theta1, theta2, mid), (name, count), color in zip(inner_angles, CATEGORIES, CATEGORY_COLORS):
        ax.add_patch(
            Wedge(
                (0, 0),
                R_INNER,
                theta1,
                theta2,
                width=R_INNER - R_HOLE,
                facecolor=color,
                edgecolor="white",
                linewidth=1.6,
                zorder=2,
            )
        )
        _radial_label(
            ax,
            f"{name}",
            0.5 * (R_HOLE + R_INNER),
            mid,
            fontsize=9.2,
            color=_contrast_text(color),
        )

    attr_span = 360.0 / len(ATTRS)
    cursor = START_ANGLE
    for attr_name, attr_color in ATTRS:
        theta1 = cursor
        theta2 = cursor + attr_span
        mid = theta1 + 0.5 * attr_span
        ax.add_patch(
            Wedge(
                (0, 0),
                R_OUTER,
                theta1,
                theta2,
                width=R_OUTER - R_INNER,
                facecolor=attr_color,
                edgecolor="white",
                linewidth=1.6,
                zorder=2,
            )
        )
        _radial_label(
            ax,
            attr_name,
            0.5 * (R_INNER + R_OUTER),
            mid,
            fontsize=8.8,
            color=_contrast_text(attr_color),
            weight="bold",
        )
        cursor = theta2

    ax.add_patch(Circle((0, 0), R_HOLE, facecolor="white", edgecolor="none", zorder=3))
    ax.text(
        0,
        0.10,
        "FITR-Bench",
        ha="center",
        va="center",
        fontsize=15,
        color="#111827",
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        0,
        -0.02,
        f"{TOTAL_ASSETS}",
        ha="center",
        va="center",
        fontsize=22,
        color="#111827",
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        0,
        -0.14,
        "12 × 20",
        ha="center",
        va="center",
        fontsize=10,
        color="#6B7280",
        zorder=4,
    )

    ax.set_xlim(-1.10, 1.10)
    ax.set_ylim(-1.10, 1.10)
    ax.set_aspect("equal")
    ax.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the FITR-Bench category sunburst.")
    parser.add_argument(
        "--out",
        type=Path,
        default=AUGMENT_ROOT / "experiments" / "figures" / "fitr_bench_sunburst.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10.2, 10.2), facecolor="white")
    ax.set_facecolor("white")
    draw_sunburst(ax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out = args.out.with_suffix(".png")
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
