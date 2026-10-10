#!/usr/bin/env python3
"""Figures for the revised paper: ZOD real-range scatter, SSD vs envelopes, and the method pipeline."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/geo_range/figures"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#1a1a19", "#5f5e5a"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def ssd_m(v, t=2.8, f=0.35, g=0.0):
    return (1.47 * v * t + v ** 2 / (30 * (f + g))) * 0.3048


def zod_scatter():
    d = pd.read_csv(ROOT / "output/geo_range/zod_rows_hfov70.csv")
    d = d[d.tag == "ungulate"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharex=True, sharey=True, constrained_layout=True)
    for ax, col, color, name in ((axes[0], "pred_gtcenter", ORANGE, "Learned range head"),
                                 (axes[1], "z_g1", BLUE, "Geometry-anchored range (G1)")):
        z, p = d.gt_z.to_numpy(), d[col].to_numpy()
        night = (d.time_of_day != "day").to_numpy()
        lo, hi = 8, 260
        xs = np.array([lo, hi])
        ax.fill_between(xs, xs * 0.8, xs * 1.2, color="#e9e8e2", lw=0)
        ax.plot(xs, xs, color=GRAY, lw=1)
        ax.scatter(z[~night], p[~night], s=16, color=color, edgecolor="white", linewidth=0.6, label="Day", zorder=3)
        ax.scatter(z[night], p[night], s=16, facecolor="white", edgecolor=color, linewidth=1.2, label="Night/twilight", zorder=3)
        el = np.polyfit(np.log(z), np.log(p), 1)[0]
        med = np.median(np.abs(p - z) / z)
        ax.set_title(f"{name}\nslope {el:.2f}, median error {100 * med:.0f}%", fontsize=9, color=INK)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("LiDAR range (m)")
        ticks = [10, 20, 50, 100, 200]
        ax.set_xticks(ticks); ax.set_xticklabels(ticks); ax.set_yticks(ticks); ax.set_yticklabels(ticks)
    axes[0].set_ylabel("Estimated range (m)")
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(OUT / "zod_range_scatter.pdf"); fig.savefig(OUT / "zod_range_scatter.png", dpi=250)
    plt.close(fig)


def ssd_envelopes():
    v = np.linspace(10, 40, 300)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    for g, color, lab in ((0.0, BLUE, "Level"), (-0.05, ORANGE, "$-5\\%$ grade"), (-0.08, AQUA, "$-8\\%$ grade")):
        ax.plot(v, ssd_m(v, g=g), color=color, lw=2, label=lab)
    for r, ls, lab in ((60, (0, (4, 3)), "60 m: rendered-data limit"), (50, "-", "50 m: G1 joint envelope"),
                       (35.5, (0, (1, 2)), "35.5 m: G1 envelope with real-data buffer")):
        ax.axhline(r, color=INK, lw=1, ls=ls)
        ax.text(39.7, r + 1.2, lab, fontsize=7.5, color=INK, ha="right")
    ax.set_xlim(10, 40); ax.set_ylim(0, 110)
    ax.set_xlabel("Speed (mph)"); ax.set_ylabel("Stopping sight distance (m)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", color="#e9e8e2", lw=0.6)
    fig.savefig(OUT / "ssd_envelopes.pdf"); fig.savefig(OUT / "ssd_envelopes.png", dpi=250)
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.set_xlim(-2, 102); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f3ee", ec=MUTED, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.9))

    box(0, 25, 15, 12, "Camera frame\n(undistorted)\nfocal $f$")
    box(19, 25, 14, 12, "Deer detector\nbox height $h$")
    box(37, 32, 16, 8, "Crop resized\nto 64$\\times$64")
    box(37, 19, 16, 8, "HeightNet\nheight $\\hat{H}$ (m)")
    box(57, 24, 16, 13, "$\\hat{Z} = f\\,\\hat{H}\\,/\\,h$\nrange (m)", fc="#dbe8f8", ec=BLUE, bold=True)
    box(77, 24, 23, 13, "Stopping-distance\nscreen + warning\n$\\hat{Z}/(1+q)$ vs. $SSD$")
    box(10, 2, 46, 10, "Geometry model (separate): depth, normals $\\rightarrow$ grade\n(nuScenes test; vehicle pitch suffices)", fc="#ffffff")
    box(77, 3, 23, 9, "Speed, grade, friction,\nreaction time, latency", fc="#ffffff")
    arrow(15.6, 31, 19, 31); arrow(33.6, 34, 37, 36); arrow(45, 32, 45, 27.6); arrow(53.6, 23, 57, 28)
    arrow(33.6, 28, 57, 30.5); arrow(73.6, 30.5, 77, 30.5); arrow(88.5, 12.6, 88.5, 24)
    ax.text(1, 41.5, "All camera dependence enters through $f$; HeightNet sees only a scale-normalized crop.", fontsize=7.2, color=BLUE)
    fig.savefig(OUT / "pipeline_geo.pdf", bbox_inches="tight"); fig.savefig(OUT / "pipeline_geo.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    zod_scatter(); ssd_envelopes(); pipeline()
    print("figures written to", OUT)
