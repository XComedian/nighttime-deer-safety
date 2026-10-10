#!/usr/bin/env python3
"""Species height priors vs learned height (G1) for geometry-anchored range on ZOD hoofed animals.

Z = f * H / h with H from: G0 constant 1.41 m; G1 HeightNet; species prior from literature (fixed in advance);
species prior from data (mean LiDAR height of the same species, leaving out the test frame); a combination
(geometric mean of G1 and the species prior); and the per-animal size oracle. Species is assumed known
(upper bound for a species-aware detector). Frame-clustered bootstrap intervals.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Typical standing box height (ground to top of head, m), set from typical adult sizes before evaluation.
LITERATURE = {"sheep": 0.75, "horse": 1.60, "cattle": 1.40, "reindeer": 1.05, "deer_or_reindeer": 1.05}
rng = np.random.default_rng(0)


def stats(z, p):
    rel = np.abs(p - z) / z
    return float(np.median(rel)), float(np.polyfit(np.log(z), np.log(p), 1)[0]), float((rel <= 0.10).mean())


def ci(df, col, n=4000):
    frames = df.frame.unique(); g = {f: df[df.frame == f] for f in frames}; out = []
    for _ in range(n):
        s = pd.concat([g[f] for f in rng.choice(frames, len(frames))])
        if s.gt_z.nunique() > 2:
            out.append(stats(s.gt_z.to_numpy(), s[col].to_numpy())[0])
    return [float(q) for q in np.percentile(out, [2.5, 97.5])]


def main():
    res = {}
    for fov in ("54.4", "70"):
        d = pd.read_csv(ROOT / f"output/geo_range/zod_rows_hfov{fov}.csv")
        d = d[d.tag == "ungulate"].copy()
        d["species_grp"] = d.species.replace({"deer_or_reindeer": "reindeer"})
        # leave-one-frame-out species mean of LiDAR height
        lofo = []
        for _, r in d.iterrows():
            pool = d[(d.species_grp == r.species_grp) & (d.frame != r.frame)]
            lofo.append(pool.hgt.mean() if len(pool) else np.nan)
        d["H_sp_data"] = lofo
        d["H_sp_lit"] = d.species.map(LITERATURE)
        f_over_h = d.f / d.box_h
        cols = {"G0 constant 1.41 m": d.z_g0, "G1 HeightNet": d.z_g1,
                "Species prior (literature)": f_over_h * d.H_sp_lit,
                "Species prior (data, leave-frame-out)": f_over_h * d.H_sp_data,
                "G1 x species prior (literature, geometric mean)": f_over_h * np.sqrt(d.H_g1 * d.H_sp_lit),
                "Size-oracle (true height)": d.base_size_oracle}
        out = {}
        for name, z in cols.items():
            d["_z"] = z
            ok = d[np.isfinite(d._z)]
            med, sl, w10 = stats(ok.gt_z.to_numpy(), ok._z.to_numpy())
            per = {sp: round(float(np.median(np.abs(g._z - g.gt_z) / g.gt_z)), 3) for sp, g in ok.groupby("species_grp")}
            out[name] = {"n": int(len(ok)), "median_rel": med, "median_rel_ci": ci(ok, "_z"), "slope": sl, "within_10pct": w10, "per_species": per}
        res[fov] = out
        print(f"== ZOD hoofed, {fov} deg view (n={len(d)}, {d.frame.nunique()} frames)")
        for name, o in out.items():
            print(f"  {name:48s} err {100*o['median_rel']:4.1f}% [{100*o['median_rel_ci'][0]:.0f}-{100*o['median_rel_ci'][1]:.0f}]  slope {o['slope']:.2f}  "
                  f"within10% {o['within_10pct']:.2f}  per-species {o['per_species']}")
    (ROOT / "output/geo_range/species_prior_eval.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
