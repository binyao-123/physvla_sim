#!/usr/bin/env python3
"""Plot FITR experiment bar charts in the π-demo visual style.

Reads data.xlsx sheets:
  - BC任务成功率对比  -> Sim-to-Real Transfer + Real-World VLA Comparison
  - RL消融图          -> Effect of RL Post-Training (bars + lines)

No error bars. Missing cells are plotted as 0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DESKTOP = Path("/home/ubuntu/Desktop")
XLSX = Path(
    "/home/ubuntu/Documents/xwechat_files/wxid_gwwnsxbq583y22_2d40/msg/file/2026-09/data.xlsx"
)
RL_XLSX = Path(
    "/home/ubuntu/Documents/xwechat_files/wxid_gwwnsxbq583y22_2d40/msg/file/2026-09/data(1).xlsx"
)
OUT_DIR = DESKTOP

# Seaborn pastel / muted: the palette ICRA, RSS, CoRL bar charts
# actually use (matplotlib defaults, desaturated). Not primary RGB.
BLUE = "#A1C9F4"
ORANGE = "#FFB482"
RED = "#FF9F9B"
LINE_BLUE = "#4878D0"
LINE_RED = "#D65F5F"
SHADOW = "#292929"
GRID = "#D4D4D4"

TASKS_EN = ["Laptop", "Display", "Microwave", "Drawer", "Lamp", "Faucet"]
TASKS_ZH = ["笔记本", "显示器", "微波炉", "抽屉", "台灯", "水龙头"]


def _pick_cjk_font() -> str | None:
    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
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


def _num(v) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_bc(path: Path):
    ws = openpyxl.load_workbook(path, data_only=True)["BC任务成功率对比"]
    sim, real, xvla, prts, pi = [], [], [], [], []
    for r in range(3, 9):
        sim.append(_num(ws.cell(r, 3).value))
        real.append(_num(ws.cell(r, 4).value))
        xvla.append(_num(ws.cell(r, 6).value))
        prts.append(_num(ws.cell(r, 7).value))
        pi.append(_num(ws.cell(r, 8).value))
    return {
        "sim": np.array(sim),
        "real": np.array(real),
        "xvla": np.array(xvla),
        "prts": np.array(prts),
        "pi": np.array(pi),
    }


def load_rl(path: Path):
    ws = openpyxl.load_workbook(path, data_only=True)["RL消融图"]
    cols = {"sim": 3, "sim_rl": 4, "real": 5, "real_rl": 6}
    data = {k: [] for k in cols}
    for r in range(3, 9):
        for k, c in cols.items():
            data[k].append(_num(ws.cell(r, c).value))
    return {k: np.array(v) for k, v in data.items()}


def _style_axes(ax, ylabel: str, ymax: float = 100.0) -> None:
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1e-9, 20))
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.tick_params(axis="x", length=0, labelsize=11)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.set_xlim(-0.6, None)


def _grouped_bars(
    ax,
    labels: list[str],
    series: list[tuple[str, np.ndarray, str]],
    *,
    ylabel: str,
    bar_width: float | None = None,
) -> None:
    n_groups = len(labels)
    n_series = len(series)
    x = np.arange(n_groups, dtype=float)
    width = bar_width if bar_width is not None else min(0.22, 0.72 / max(n_series, 1))
    gap = width * 0.12
    total = n_series * width + (n_series - 1) * gap
    offsets = -0.5 * total + width / 2 + np.arange(n_series) * (width + gap)
    shadow_dx = width * 0.18

    for i, (name, values, color) in enumerate(series):
        xpos = x + offsets[i]
        # Drop shadow (offset bottom-right), then face bar with black edge.
        ax.bar(
            xpos + shadow_dx,
            np.maximum(values, 0),
            width,
            color=SHADOW,
            edgecolor="none",
            zorder=2,
            align="center",
        )
        ax.bar(
            xpos,
            values,
            width,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            label=name,
            zorder=3,
            align="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=11)
    _style_axes(ax, ylabel)
    ax.set_xlim(-0.55, n_groups - 0.45)
    leg = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        loc="upper right",
        prop={"weight": "bold", "size": 11},
    )
    leg.get_frame().set_linewidth(0.8)


def plot_bc_sim_real(data, fontfamily: str | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    if fontfamily:
        plt.rcParams["font.family"] = fontfamily
    _grouped_bars(
        ax,
        TASKS_EN,
        [
            ("Simulation", data["sim"], BLUE),
            ("Real world", data["real"], ORANGE),
        ],
        ylabel="Normalized Joint Completion (%)",
        bar_width=0.34,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_bc_models(data, fontfamily: str | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2), facecolor="white")
    if fontfamily:
        plt.rcParams["font.family"] = fontfamily
    _grouped_bars(
        ax,
        TASKS_EN,
        [
            ("X-VLA", data["xvla"], BLUE),
            ("PRTS", data["prts"], ORANGE),
            (r"$\pi_{0.5}$", data["pi"], RED),
        ],
        ylabel="Normalized Joint Completion (%)",
        bar_width=0.22,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_rl_framework(data, fontfamily: str | None, out: Path) -> None:
    """Bars for Real / Real+RL; lines for Sim / Sim+RL. Empty sheet -> zeros."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4), facecolor="white")
    if fontfamily:
        plt.rcParams["font.family"] = fontfamily

    labels = TASKS_EN
    n = len(labels)
    x = np.arange(n, dtype=float)
    width = 0.32
    gap = 0.06
    off = np.array([-0.5 * (2 * width + gap) + width / 2, 0.5 * (2 * width + gap) - width / 2])
    shadow_dx = width * 0.22

    bar_series = [
        (data["real"], BLUE),
        (data["real_rl"], RED),
    ]
    for i, (values, color) in enumerate(bar_series):
        xpos = x + off[i]
        ax.bar(
            xpos + shadow_dx,
            np.maximum(values, 0),
            width,
            color=SHADOW,
            edgecolor="none",
            zorder=2,
        )
        ax.bar(
            xpos,
            values,
            width,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )

    # Lines: simulation. Bars: real world. Fig. 4(c) compares BC vs RL post-training.
    ax.plot(
        x,
        data["sim"],
        color=LINE_BLUE,
        marker="o",
        markersize=6,
        linewidth=2.0,
        zorder=4,
    )
    ax.plot(
        x,
        data["sim_rl"],
        color=LINE_RED,
        marker="s",
        markersize=5.5,
        linewidth=2.0,
        linestyle="--",
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=11)
    _style_axes(ax, "Normalized Joint Completion (%)")
    ax.set_xlim(-0.55, n - 0.45)
    leg = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=LINE_BLUE,
                marker="o",
                markersize=6,
                linewidth=2.0,
                label="Simulation (BC)",
            ),
            Line2D(
                [0],
                [0],
                color=LINE_RED,
                marker="s",
                markersize=5.5,
                linewidth=2.0,
                linestyle="--",
                label="Simulation (RL post-training)",
            ),
            Patch(
                facecolor=BLUE,
                edgecolor="black",
                linewidth=1.0,
                label="Real world (BC)",
            ),
            Patch(
                facecolor=RED,
                edgecolor="black",
                linewidth=1.0,
                label="Real world (RL post-training)",
            ),
        ],
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        loc="upper right",
        ncol=1,
        prop={"weight": "bold", "size": 11},
    )
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    fontfamily = _pick_cjk_font()
    if fontfamily:
        plt.rcParams["font.family"] = fontfamily
    plt.rcParams["axes.unicode_minus"] = False

    bc = load_bc(XLSX)
    rl = load_rl(RL_XLSX)

    plot_bc_sim_real(bc, fontfamily, OUT_DIR / "bc_sim_real_success.png")
    plot_bc_models(bc, fontfamily, OUT_DIR / "bc_model_success.png")
    plot_rl_framework(rl, fontfamily, OUT_DIR / "rl_ablation.png")


if __name__ == "__main__":
    main()
