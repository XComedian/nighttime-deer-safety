#!/usr/bin/env python3
"""Qualitative ZOD results: real frames with LiDAR range vs. learned range head, G1, and Metric3D v2 ViT-L,
plus a zoom test (same animal at 54.4 and 70 degrees)."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/geo_range/figures"
ORANGE, BLUE, GRAY, INK, GREEN = "#eb6834", "#2a78d6", "#8a8a85", "#1a1a19", "#1baf7a"
plt.rcParams.update({"font.family": "serif", "font.size": 8})
PANELS = [("029831", "(a) Horse, day"), ("081417", "(b) Reindeer, twilight"), ("065519", "(c) Deer/reindeer, day"),
          ("073867", "(d) Horses, day"), ("056030", "(e) Sheep, day"), ("059465", "(f) Horse, night (failure)")]


def load(view: str) -> pd.DataFrame:
    ev = json.loads((ROOT / "output/depth_baselines/eval_boxes.json").read_text())
    m3 = json.loads((ROOT / "output/depth_baselines/metric3d_l.json").read_text())
    d = pd.DataFrame(ev[f"zod_{view}"])
    d["m3d"] = [r["pred"] for r in m3[f"zod_{view}"]]
    return d[d.tag == "ungulate"]


def brighten(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def region(boxes, W=960, H=540, aspect=4 / 3, min_w=150):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    w = max(min_w, 2.0 * (x1 - x0), 2.6 * (y1 - y0) * aspect); h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx0 = int(np.clip(cx - w / 2, 0, W - w)); ry0 = int(np.clip(cy - h * 0.55, 0, H - h))
    return rx0, ry0, int(w), int(h)


def panel(ax, d, fid, title):
    rows = d[d.frame == fid].sort_values("gt_z")
    img = brighten(cv2.imread(rows.iloc[0].image))
    rx, ry, rw, rh = region(list(rows.box))
    ax.imshow(img[ry:ry + rh, rx:rx + rw], extent=(rx, rx + rw, ry + rh, ry))
    shown = rows.assign(hpx=rows.box.map(lambda b: b[3] - b[1])).nlargest(3, "hpx").sort_values("gt_z")
    for k, r in enumerate(shown.itertuples(), 1):
        x0, y0, x1, y1 = r.box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ffd23f", lw=1.0))
        ax.text(x0, y0 - 1, str(k), color="#ffd23f", fontsize=7, fontweight="bold", va="bottom")
    ax.set_xlim(rx, rx + rw); ax.set_ylim(ry + rh, ry); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    lines = []
    for k, r in enumerate(shown.itertuples(), 1):
        lines.append((k, r.gt_z, r.pred_head, r.z_g1, r.m3d))
    return lines


def table(ax, lines):
    ax.axis("off")
    hdr = ["#", "LiDAR", "Head", "G1", "M3D-L"]
    cols = [INK, INK, ORANGE, BLUE, GREEN]
    xs = [0.02, 0.13, 0.36, 0.57, 0.77]
    for x, h, c in zip(xs, hdr, cols):
        ax.text(x, 0.92, h, color=c, fontsize=7, fontweight="bold", transform=ax.transAxes, va="top")
    for i, (k, z, hd, g, m) in enumerate(lines):
        y = 0.70 - 0.24 * i
        vals = [str(k), f"{z:.0f} m", f"{hd:.0f} m", f"{g:.0f} m", f"{m:.0f} m"]
        for x, v, c in zip(xs, vals, cols):
            ax.text(x, y, v, color=c, fontsize=7, transform=ax.transAxes, va="top")


def main():
    d70, d54 = load("70"), load("54.4")
    OUT.mkdir(parents=True, exist_ok=True)
    # Figure A: six examples
    fig = plt.figure(figsize=(7.0, 6.4))
    gs = fig.add_gridspec(4, 3, height_ratios=[3.0, 1.15, 3.0, 1.15], hspace=0.12, wspace=0.08)
    for i, (fid, title) in enumerate(PANELS):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[2 * r, c]); lines = panel(ax, d70, fid, title)
        table(fig.add_subplot(gs[2 * r + 1, c]), lines)
    fig.savefig(OUT / "zod_qualitative.png", dpi=220, bbox_inches="tight"); fig.savefig(OUT / "zod_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)
    # Figure B: zoom test
    zf = "081417"
    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.2], hspace=0.12, wspace=0.08, width_ratios=[1, 1, 0.9])
    for j, (d, lab) in enumerate(((d54, "(a) 54.4° view"), (d70, "(b) 70° view: same animals, 1.36× smaller"))):
        ax = fig.add_subplot(gs[0, j]); lines = panel(ax, d, zf, lab)
        table(fig.add_subplot(gs[1, j]), lines)
    a = d54[d54.frame == zf]; b = d70[d70.frame == zf]
    ax = fig.add_subplot(gs[:, 2]); ax.axis("off")
    txt = (f"Median of the three reindeer\n(LiDAR {a.gt_z.median():.1f} m):\n\n"
           f"Head   {a.pred_head.median():.1f} → {b.pred_head.median():.1f} m\n"
           f"G1     {a.z_g1.median():.1f} → {b.z_g1.median():.1f} m\n"
           f"M3D-L  {a.m3d.median():.1f} → {b.m3d.median():.1f} m\n\n"
           "Only the lens changes, so each\nestimate should stay the same.\nG1 and M3D-L barely move;\nthe head shifts (35 → 27 m,\n60 → 51 m) yet stays far\nfrom LiDAR for every animal.")
    ax.text(0.02, 0.95, txt, va="top", fontsize=7.2, color=INK, family="monospace", transform=ax.transAxes)
    fig.savefig(OUT / "zod_zoom_test.png", dpi=220, bbox_inches="tight"); fig.savefig(OUT / "zod_zoom_test.pdf", bbox_inches="tight")
    print(OUT / "zod_qualitative.png", OUT / "zod_zoom_test.png")


if __name__ == "__main__":
    main()
