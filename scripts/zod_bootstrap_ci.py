#!/usr/bin/env python3
"""Frame-clustered bootstrap 95% intervals for the ZOD real-range metrics (animals in one frame are resampled together)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"head": "pred_gtcenter", "G0": "z_g0", "G1": "z_g1", "size_oracle": "base_size_oracle"}


def metrics(g, col):
    rel = np.abs(g[col] - g.gt_z) / g.gt_z
    return np.median(rel), np.polyfit(np.log(g.gt_z), np.log(g[col]), 1)[0]


def main(n_boot=10000):
    rng = np.random.default_rng(0)
    out = {}
    for fov in ("54.4", "70"):
        d = pd.read_csv(ROOT / f"output/geo_range/zod_rows_hfov{fov}.csv")
        for name, g in (("hoofed", d[d.tag == "ungulate"]), ("deer_like", d[d.species.isin(["reindeer", "deer_or_reindeer"])])):
            frames = g.frame.unique()
            groups = {f: g[g.frame == f] for f in frames}
            res = {}
            for m, col in METHODS.items():
                pt = metrics(g, col)
                bs = []
                for _ in range(n_boot):
                    s = pd.concat([groups[f] for f in rng.choice(frames, len(frames))])
                    if s.gt_z.nunique() > 2:
                        bs.append(metrics(s, col))
                bs = np.array(bs)
                res[m] = {"median_rel": float(pt[0]), "median_rel_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                          "slope": float(pt[1]), "slope_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist()}
            out[f"{name}_{fov}"] = {"n": int(len(g)), "frames": int(len(frames)), **res}
            print(fov, name, len(g), "animals", len(frames), "frames |",
                  " ".join(f"{m}: {100*r['median_rel']:.0f}% [{100*r['median_rel_ci'][0]:.0f}-{100*r['median_rel_ci'][1]:.0f}] "
                           f"slope {r['slope']:.2f} [{r['slope_ci'][0]:.2f},{r['slope_ci'][1]:.2f}]" for m, r in res.items() if m in ("head", "G1")))
    (ROOT / "output/geo_range/zod_bootstrap_ci.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
