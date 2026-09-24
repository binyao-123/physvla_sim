#!/usr/bin/env python3
"""Draw a two-ring sunburst of FITR-Assets category counts.

Inner ring: source (URDF-Anything+ vs FITR-augmented).
Outer ring: the 12 object categories, sized by asset count.

  .venv/bin/python plot_fitr_assets_sunburst.py
  .venv/bin/python plot_fitr_assets_sunburst.py --lang zh --out experiments/figures/fitr_assets_sunburst.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_hex
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Circle, Wedge

AUGMENT_ROOT = Path(__file__).resolve().parent

# Verified FITR-Assets inventory. Drawer is PartNet-Mobility Table, not "double drawer".
# (english_label, chinese_label, count)
INHERITED = [
    ("Faucet", "水龙头", 655),
    ("Laptop", "笔记本", 470),
    ("Storage", "储物柜", 366),
    ("Display", "显示器", 290),
    ("Microwave", "微波炉", 268),
    ("Fridge", "冰箱", 238),
    ("Dishwasher", "洗碗机", 234),
    ("Door", "门", 47),
    ("Scissors", "剪刀", 31),
    ("Knife", "小刀", 27),
]
AUGMENTED = [
    ("Drawer", "抽屉", 101),
    ("Lamp", "台灯", 45),
]

TOTAL_ASSETS = sum(n for *_, n in INHERITED + AUGMENTED)
N_CATEGORIES = len(INHERITED) + len(AUGMENTED)
N_INHERITED = sum(n for *_, n in INHERITED)
N_AUGMENTED = sum(n for *_, n in AUGMENTED)

# Inner ring: source encoding (light blue = inherited, deep red = FITR-added).
BLUE_INNER = "#7EB6D9"
RED_INNER = "#8B1E1E"
# Outer ring: pastel categorical palette; FITR pair uses deep reds.
OUTER_COLORS = [
    "#A8C5E2",  # Faucet
    "#F5C49A",  # Laptop
    "#F0A8A8",  # Storage
    "#A8D5D0",  # Display
    "#A8D4A0",  # Microwave
    "#F0E0A0",  # Fridge
    "#D4B8D8",  # Dishwasher
    "#F5C8D0",  # Door
    "#D4C0B0",  # Scissors
    "#D8D4D0",  # Knife
    "#8B1E1E",  # Drawer (FITR, deep)
    "#A33A1C",  # Lamp (FITR, deep)
]
HIGHLIGHT_RED = "#8B1E1E"

R_HOLE = 0.38
R_INNER = 0.66
R_OUTER = 1.08
START_ANGLE = 90.0  # 12 o'clock, then counterclockwise
TEXT_WHITE = "#FFFFFF"
TEXT_BLACK = "#1A1A1A"
# Modest radial explode for the FITR sector (matplotlib pie "explode").
FITR_EXPLODE = 0.058
FITR_EDGE = "#FFF7F5"
FITR_EDGE_LW = 2.6
FITR_OUTER_COLORS = set(OUTER_COLORS[-2:])


def _contrast_text(hex_color: str) -> str:
    rgb = tuple(int(to_hex(hex_color)[i : i + 2], 16) for i in (1, 3, 5))
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return TEXT_WHITE if luminance < 155 else TEXT_BLACK


def _pick_cjk_font() -> str | None:
    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "DejaVu Sans",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return None


def _label(en: str, zh: str, lang: str) -> str:
    if lang == "zh":
        return zh
    if lang == "both":
        return f"{zh} {en}"
    return en


def _wedge_angles(sizes: list[int], start: float = START_ANGLE) -> list[tuple[float, float, float]]:
    """Return (theta1, theta2, mid) in matplotlib Wedge degrees (CCW from +x)."""
    angles = []
    cursor = start
    for size in sizes:
        span = 360.0 * size / TOTAL_ASSETS
        theta1 = cursor
        theta2 = cursor + span
        angles.append((theta1, theta2, theta1 + 0.5 * span))
        cursor = theta2
    return angles


def _polar(radius: float, deg: float, origin: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    rad = np.deg2rad(deg)
    return float(origin[0] + radius * np.cos(rad)), float(origin[1] + radius * np.sin(rad))


def _explode_xy(mid_deg: float, amount: float) -> tuple[float, float]:
    """Offset a wedge center along its angular bisector (classic explode)."""
    return _polar(amount, mid_deg)


def _radial_rotation(deg: float) -> float:
    """Rotate a label so it reads along the radius and stays upright."""
    wrapped = deg % 360.0
    if 90.0 < wrapped <= 270.0:
        return wrapped + 180.0
    return wrapped


def _arc_text(
    ax,
    text: str,
    radius: float,
    center_deg: float,
    *,
    fontsize: float,
    color: str,
    fontfamily: str | None,
    weight: str = "medium",
    flip: bool = False,
) -> None:
    """Place characters along a circular arc, tangent to the ring.

    Each glyph sits at its own angle. Default: rotation = angle - 90.
    With flip=True, reverse travel direction and use rotation = angle + 90
    so the string faces the opposite way without mirroring individual glyphs.
    """
    if not text:
        return
    # Angular pitch ≈ character width / radius; keep loose so the string
    # wraps visibly along the arc instead of stacking tightly.
    char_step = 6.2 * (fontsize / 12.0) * (0.51 / radius)
    n = len(text)
    half = 0.5 * (n - 1) * char_step
    if flip:
        start = center_deg - half
        step = char_step
        rot_offset = 90.0
    else:
        start = center_deg + half
        step = -char_step
        rot_offset = -90.0
    for i, ch in enumerate(text):
        deg = start + i * step
        ax.text(
            *_polar(radius, deg),
            ch,
            ha="center",
            va="center",
            rotation=deg + rot_offset,
            fontsize=fontsize,
            color=color,
            fontfamily=fontfamily,
            fontweight=weight,
            zorder=5,
        )


def _mixed_line(
    ax,
    x: float,
    y: float,
    segments: list[tuple[str, str, str]],
    *,
    fontsize: float,
    fontfamily: str | None,
    zorder: int = 4,
) -> None:
    """Centered line assembled from colored text segments."""
    children = []
    for text, color, weight in segments:
        props = {
            "color": color,
            "fontsize": fontsize,
            "fontweight": weight,
            "ha": "left",
            "va": "baseline",
        }
        if fontfamily:
            props["fontfamily"] = fontfamily
        children.append(TextArea(text, textprops=props))
    packed = HPacker(children=children, align="center", pad=0, sep=0)
    ax.add_artist(
        AnnotationBbox(
            packed,
            (x, y),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=zorder,
        )
    )


def _radial_label(
    ax,
    text: str,
    radius: float,
    deg: float,
    *,
    fontsize: float,
    color: str,
    fontfamily: str | None,
    weight: str = "medium",
    origin: tuple[float, float] = (0.0, 0.0),
) -> None:
    """One intact label rotated to follow the slice radius, as in a sunburst."""
    ax.text(
        *_polar(radius, deg, origin),
        text,
        ha="center",
        va="center",
        rotation=_radial_rotation(deg),
        fontsize=fontsize,
        color=color,
        fontfamily=fontfamily,
        fontweight=weight,
        zorder=5,
    )


def draw_sunburst(ax, lang: str, fontfamily: str | None) -> None:
    inherited_sizes = [n for *_, n in INHERITED]
    augmented_sizes = [n for *_, n in AUGMENTED]
    outer_sizes = inherited_sizes + augmented_sizes
    outer_meta = [
        (en, zh, n, c)
        for (en, zh, n), c in zip(INHERITED + AUGMENTED, OUTER_COLORS)
    ]

    inner_angles = _wedge_angles([N_INHERITED, N_AUGMENTED])
    outer_angles = _wedge_angles(outer_sizes)

    inner_wedges = [
        (inner_angles[0], BLUE_INNER, "URDF-Anything+", _contrast_text(BLUE_INNER), 20.0, "bold", False),
        (inner_angles[1], RED_INNER, "FITR", _contrast_text(RED_INNER), 16.0, "bold", True),
    ]
    # Explode the whole FITR band (inner + outer) along the FITR bisector.
    fitr_origin = _explode_xy(inner_angles[1][2], FITR_EXPLODE)

    for (theta1, theta2, mid), color, name, text_color, fontsize, weight, is_fitr in inner_wedges:
        origin = fitr_origin if is_fitr else (0.0, 0.0)
        ax.add_patch(
            Wedge(
                origin,
                R_INNER,
                theta1,
                theta2,
                width=R_INNER - R_HOLE,
                facecolor=color,
                edgecolor=FITR_EDGE if is_fitr else "white",
                linewidth=FITR_EDGE_LW if is_fitr else 1.8,
                zorder=4 if is_fitr else 2,
            )
        )
        if name == "URDF-Anything+":
            # Lower-left of the blue arc; flipped to face the opposite way.
            # Mid-band radius so the label sits in the center of the ring.
            _arc_text(
                ax,
                name,
                0.5 * (R_HOLE + R_INNER),
                215.0,
                fontsize=fontsize,
                color=text_color,
                fontfamily=fontfamily,
                weight=weight,
                flip=True,
            )
        else:
            _radial_label(
                ax,
                name,
                0.5 * (R_HOLE + R_INNER),
                mid,
                fontsize=fontsize,
                color=text_color,
                fontfamily=fontfamily,
                weight=weight,
                origin=origin,
            )

    for (theta1, theta2, mid), (en, zh, count, color) in zip(outer_angles, outer_meta):
        is_fitr = color in FITR_OUTER_COLORS
        origin = fitr_origin if is_fitr else (0.0, 0.0)
        ax.add_patch(
            Wedge(
                origin,
                R_OUTER,
                theta1,
                theta2,
                width=R_OUTER - R_INNER,
                facecolor=color,
                edgecolor=FITR_EDGE if is_fitr else "white",
                linewidth=FITR_EDGE_LW if is_fitr else 1.5,
                zorder=4 if is_fitr else 2,
            )
        )
        name = _label(en, zh, lang)
        label = f"{name} {count}"
        span = (theta2 - theta1) % 360.0
        if span >= 40.0:
            fontsize = 15.0
        elif span >= 18.0:
            fontsize = 13.4
        else:
            fontsize = 12.0
        if is_fitr:
            fontsize += 1.0
        _radial_label(
            ax,
            label,
            0.5 * (R_INNER + R_OUTER),
            mid,
            fontsize=fontsize,
            color=_contrast_text(color),
            fontfamily=fontfamily,
            weight="bold",
            origin=origin,
        )

    ax.add_patch(Circle((0, 0), R_HOLE, facecolor="white", edgecolor="none", zorder=3))
    ax.text(
        0,
        0.11,
        "FITR-Assets",
        ha="center",
        va="center",
        fontsize=17,
        color="#111827",
        fontfamily=fontfamily,
        fontweight="bold",
        zorder=4,
    )
    _mixed_line(
        ax,
        0,
        -0.01,
        [
            (f"{TOTAL_ASSETS:,}", "#1E4D7B", "bold"),
            (f" (+{N_AUGMENTED})", HIGHLIGHT_RED, "bold"),
        ],
        fontsize=16,
        fontfamily=fontfamily,
    )
    _mixed_line(
        ax,
        0,
        -0.13,
        [
            (f"{N_CATEGORIES}", "#1E4D7B", "bold"),
            (f" (+{len(AUGMENTED)})", HIGHLIGHT_RED, "bold"),
            (_label(" Categories", " 类资产", lang), "#6B7280", "medium"),
        ],
        fontsize=10,
        fontfamily=fontfamily,
    )

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the FITR-Assets category sunburst.")
    parser.add_argument(
        "--out",
        type=Path,
        default=AUGMENT_ROOT / "experiments" / "figures" / "fitr_assets_sunburst.png",
        help="Output image path (.png or .pdf).",
    )
    parser.add_argument("--lang", choices=("en", "zh", "both"), default="en")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", action="store_true", help="Add a figure title above the chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fontfamily = _pick_cjk_font()
    if fontfamily:
        plt.rcParams["font.family"] = fontfamily
        plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10.6, 10.2), facecolor="white")
    ax.set_facecolor("white")
    draw_sunburst(ax, args.lang, fontfamily)
    if args.title:
        fig.suptitle(
            _label("FITR-Assets Category Composition", "FITR-Assets 类别构成", args.lang),
            fontsize=16,
            fontweight="bold",
            color="#111827",
            y=0.98,
        )
        fig.subplots_adjust(top=0.93)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out = args.out.with_suffix(".png")
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
