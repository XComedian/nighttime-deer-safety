#!/usr/bin/env python3
"""Score the HeightNet variants from zod_finetune.py on held-out drives (range Z = f H / h)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/geo_range"
VARIANTS = {"H_rendered_only": "Rendered only (G1)", "H_finetuned": "Rendered -> real fine-tuned", "H_real_only": "Real only (scratch)",
            "H_finetuned_large": "Fine-tuned, large animals only", "H_finetuned_balanced": "Fine-tuned, size-balanced"}
rng = np.random.default_rng(0)


def stats(z, p):
    rel = np.abs(p - z) / z
    return float(np.median(rel)), float(np.polyfit(np.log(z), np.log(p), 1)[0]), float((rel <= 0.10).mean())


def ci(df, col, n=4000):
    fr = df.frame.unique(); g = {f: df[df.frame == f] for f in fr}; v = []
    for _ in range(n):
        s = pd.concat([g[f] for f in rng.choice(fr, len(fr))])
        if s.gt_z.nunique() > 2:
            v.append(stats(s.gt_z.to_numpy(), s[col].to_numpy())[0])
    return [float(q) for q in np.percentile(v, [2.5, 97.5])]


def main():
    res = {}
    for fov in ("70", "54.4"):
        d = pd.read_csv(OUT / f"zod_finetune_rows_hfov{fov}.csv")
        for c in VARIANTS:
            d["Z_" + c] = d.fx * d[c] / d.h_px
        groups = {"hoofed": d[d.tag == "ungulate"], "deer_like": d[d.species.isin(["reindeer", "deer_or_reindeer"])],
                  "dogs": d[d.tag == "dog"], "all_non_statue": d}
        res[fov] = {}
        for gname, g in groups.items():
            e = {"n": int(len(g)), "frames": int(g.frame.nunique())}
            for c, lab in VARIANTS.items():
                med, sl, w10 = stats(g.gt_z.to_numpy(), g["Z_" + c].to_numpy())
                e[lab] = {"median_rel": med, "slope": sl, "within_10pct": w10,
                          "median_rel_ci": ci(g, "Z_" + c) if gname in ("hoofed", "deer_like") else None,
                          "H_median": float(g[c].median())}
            e["H_true_median"] = float(g.H_star.median()) if "H_star" in g else None
            res[fov][gname] = e
    (OUT / "zod_finetune_eval.json").write_text(json.dumps(res, indent=2))
    for fov in res:
        for gname, e in res[fov].items():
            print(f"== {fov} deg, {gname} (n={e['n']}, frames={e['frames']})" + (f", true H* median {e['H_true_median']:.2f}" if e['H_true_median'] else ""))
            for lab in VARIANTS.values():
                r = e[lab]; c = f" [{100*r['median_rel_ci'][0]:.0f}-{100*r['median_rel_ci'][1]:.0f}]" if r["median_rel_ci"] else ""
                print(f"   {lab:30s} err {100*r['median_rel']:5.1f}%{c}  slope {r['slope']:.2f}  within10% {r['within_10pct']:.2f}  H_med {r['H_median']:.2f}")


if __name__ == "__main__":
    main()
