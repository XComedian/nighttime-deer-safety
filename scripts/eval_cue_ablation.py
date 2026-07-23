#!/usr/bin/env python3
"""Evaluate ONE checkpoint across the 4 road-furniture variants (cueA-D).

The four scenes are identical except for lane markings and delineator posts, so a
single model evaluated on all four isolates the cue effect from model and scene
variation. Reports per-band detection recall, matched range MAE, and joint success.

Checkpoint: a fold model trained on the full rural family INCLUDING crossing11, so an
unmarked road is not novel to it -- any degradation reflects missing depth cues rather
than unfamiliarity.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from loso_eval import BANDS, TOL, eval_fold  # noqa: E402  (reuses the same matched-eval)

CUES = ["cueA", "cueB", "cueC", "cueD"]
LABELS = {"cueA": "markings + posts (control)", "cueB": "markings only",
          "cueC": "posts only", "cueD": "neither"}
DEFAULT_CKPT = ROOT / "output/loso_asset/c14/weights/best.pt"
OUT = ROOT / "output" / "cue_ablation"


def main() -> None:
    ckpt = Path(os.environ.get("CUE_CKPT", DEFAULT_CKPT))
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {ckpt}")

    results = []
    for tag in CUES:
        r = eval_fold(tag, ckpt)          # same matched detection+range protocol
        r["label"] = LABELS[tag]
        results.append(r)
        b = r["bands"]
        print(f"{tag} ({LABELS[tag]:<26}) "
              + " ".join(f"{k}: R={b[k]['recall']:.3f} MAE={b[k]['mae']:.1f}" for k, _, _ in BANDS))
    (OUT / "cue_results.json").write_text(json.dumps(results, indent=2))

    def row(r, key):
        return " | ".join(f"{r['bands'][b][key]:.3f}" if key != "mae"
                          else f"{r['bands'][b][key]:.2f}" for b, _, _ in BANDS)

    lines = [
        "# Road-Furniture Cue Ablation",
        "",
        f"One checkpoint (`{ckpt.relative_to(ROOT)}`) evaluated zero-shot on four scenes that are",
        "identical in geometry, vegetation, lighting and seed, differing ONLY in whether lane",
        "markings and delineator posts are present. Any difference therefore isolates the cue.",
        "",
        "## Matched range error (m) by distance band",
        "",
        "| variant | markings | posts | 30-40 | 40-50 | 50-60 |",
        "|---|:--:|:--:|---:|---:|---:|",
    ]
    mk = {"cueA": ("yes", "yes"), "cueB": ("yes", "no"), "cueC": ("no", "yes"), "cueD": ("no", "no")}
    for r in results:
        m, p = mk[r["held"]]
        lines.append(f"| {r['held']} {LABELS[r['held']]} | {m} | {p} | " + row(r, "mae") + " |")
    lines += ["", "## Detection recall by band", "",
              "| variant | 30-40 | 40-50 | 50-60 |", "|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['held']} | " + row(r, "recall") + " |")
    lines += ["", f"## Joint success (detected AND |range error| <= {TOL:.0f} m)", "",
              "| variant | 30-40 | 40-50 | 50-60 |", "|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['held']} | " + row(r, "joint") + " |")

    a = next(r for r in results if r["held"] == "cueA")
    d = next(r for r in results if r["held"] == "cueD")
    dm = [d["bands"][b]["mae"] - a["bands"][b]["mae"] for b, _, _ in BANDS]
    lines += ["", "## Effect size (cueD minus cueA, range MAE in m)", "",
              "  " + ", ".join(f"{b}: {v:+.2f}" for (b, _, _), v in zip(BANDS, dm)),
              "",
              "Positive values mean removing both cues WORSENS ranging. Detection recall is",
              "reported alongside so a range change is not confused with a detection change."]
    (OUT / "cue_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"\nwrote {OUT}/cue_ablation.md")


if __name__ == "__main__":
    main()
