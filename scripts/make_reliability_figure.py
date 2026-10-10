#!/usr/bin/env python3
"""Probability of late warning vs speed (level road) for the reliability-based screen."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "output/reliability_ssd/reliability_results.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
series = [("perfect perception (to 60 m)", "Perfect perception to 60 m", "#1a1a19", (0, (4, 2))),
          ("real G1 + buffer q90", "G1, real error + 90% buffer", "#2a78d6", "-"),
          ("real G1", "G1, real error, no buffer", "#2a78d6", (0, (1, 1.5))),
          ("real species-mean height", "Species-mean height, real error", "#1baf7a", "-"),
          ("real range head", "Learned range head, real error", "#eb6834", "-")]
fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
for key, lab, col, ls in series:
    c = res["curves"][f"{key} | level"]
    v = sorted(int(k) for k in c); p = [max(c[str(k)]["p_nc"], 1e-4) for k in v]
    ax.plot(v, p, color=col, ls=ls, lw=2, label=lab)
for r in (0.01, 0.05, 0.10):
    ax.axhline(r, color="#c9c8c2", lw=0.8, zorder=0); ax.text(15.3, r * 1.08, f"{int(r*100)}%", fontsize=7, color="#5f5e5a")
ax.set_yscale("log"); ax.set_ylim(1e-3, 1); ax.set_xlim(15, 45)
ax.set_xlabel("Speed (mph)"); ax.set_ylabel("Probability of late warning")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
out = ROOT / "output/reliability_ssd"
fig.savefig(out / "reliability_curves.pdf"); fig.savefig(out / "reliability_curves.png", dpi=250)
print(out / "reliability_curves.png")
