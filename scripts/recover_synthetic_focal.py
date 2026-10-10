#!/usr/bin/env python3
"""Recover per-frame focal length for the rendered rural scenes (lens was randomized, not logged).

For a road of constant paved width W seen at depth Z, the surface spans du = f * W / Z pixels along an
image row, so k = du * Z = f * W. The paved-surface instance mask ('AR8_uphill_visible_lane_surface',
present in every rural scene) and the z-depth map give k per row; the frame value is the median over rows
whose road depth is 8-25 m and whose surface boundaries lie inside the image. W is then fixed by the known
lens prior (24-50 mm on a 36 mm sensor -> f in [640, 1333] px at 960 px width), and the recovered spread is
checked against that prior.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import h5py
import numpy as np

SYN = Path(os.environ.get("SYNTHETIC_ROOT", ROOT / "data/synthetic_deer_dataset_raw"))
OUT = Path(__file__).resolve().parents[1] / "output/synthetic_focal"
SCENES = ["crossing8", "crossing9a", "crossing9b", "crossing9c", "crossing10", "crossing11", "crossing12",
          "crossing13", "crossing14", "crossing15"]
SURFACE = "AR8_uphill_visible_lane_surface"


def frame_k(path: Path) -> float | None:
    with h5py.File(path) as f:
        attrs = json.loads(f["instance_attribute_maps"][()])
        ids = [a["idx"] for a in attrs if a["name"] == SURFACE]
        if not ids:
            return None
        inst = f["instance_segmaps"][:]
        depth = f["depth"][:]
    mask = inst == ids[0]
    h, w = mask.shape
    ks = []
    for v in range(h // 2, h):
        cols = np.flatnonzero(mask[v])
        if len(cols) < 10:
            continue
        lo, hi = cols[0], cols[-1]
        if lo <= 1 or hi >= w - 2:             # boundary leaves the image: width not observed
            continue
        if (hi - lo + 1) > 1.15 * len(cols):   # row interrupted (occluder or gap): unreliable
            continue
        z = float(np.median(depth[v, cols]))
        if 8.0 <= z <= 25.0:
            ks.append((hi - lo + 1) * z)
    return float(np.median(ks)) if len(ks) >= 5 else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(sc, p) for sc in SCENES for p in sorted((SYN / sc).glob("*.hdf5"), key=lambda q: int(q.stem))]
    with ProcessPoolExecutor(32) as ex:
        ks = list(ex.map(frame_k, [p for _, p in jobs], chunksize=20))
    k = np.array([x for x in ks if x is not None])
    f_lo, f_hi = 24 / 36 * 960, 50 / 36 * 960
    width = float(np.median(k)) / ((f_lo + f_hi) / 2)          # uniform lens prior -> median f = midpoint
    focal = {f"{sc}/{p.stem}": (kk / width if kk is not None else None) for (sc, p), kk in zip(jobs, ks)}
    fv = np.array([v for v in focal.values() if v is not None])
    q = np.percentile(fv, [1, 5, 50, 95, 99])
    hist = np.histogram(fv, bins=8, range=(f_lo, f_hi))[0]
    report = {"frames": len(jobs), "recovered": int(len(fv)), "road_width_m": width,
              "focal_px_p1_p5_p50_p95_p99": q.tolist(), "prior_px": [f_lo, f_hi],
              "p99_over_p1": float(q[4] / q[0]), "prior_ratio": f_hi / f_lo,
              "fraction_inside_prior": float(((fv >= 0.95 * f_lo) & (fv <= 1.05 * f_hi)).mean()),
              "histogram_8_bins_over_prior": hist.tolist()}
    (OUT / "focal_px.json").write_text(json.dumps(focal))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
