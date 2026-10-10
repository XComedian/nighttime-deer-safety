#!/usr/bin/env python3
"""Compare zero-shot metric-depth baselines with the learned range head and geometry-anchored range (G1)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "output/depth_baselines"
MODELS = ["metric3d_s", "metric3d_l", "unidepth_s", "unidepth_l", "dav2_metric_b"]
LABEL = {"metric3d_s": "Metric3D v2 ViT-S", "metric3d_l": "Metric3D v2 ViT-L", "unidepth_s": "UniDepth v2 ViT-S",
         "unidepth_l": "UniDepth v2 ViT-L", "dav2_metric_b": "Depth Anything V2 Metric-Outdoor-B",
         "head": "Learned range head", "g1": "G1 (geometry-anchored)", "oracle": "Size-oracle"}
BANDS = [(30, 40), (40, 50), (50, 60)]
rng = np.random.default_rng(0)


def med_rel(z, p):
    return float(np.median(np.abs(p - z) / z))


def slope(z, p):
    p = np.clip(p, 0.5, None)
    return float(np.polyfit(np.log(z), np.log(p), 1)[0])


def cluster_ci(df, col, fn, n=4000):
    frames = df.frame.unique()
    groups = {f: df[df.frame == f] for f in frames}
    vals = []
    for _ in range(n):
        s = pd.concat([groups[f] for f in rng.choice(frames, len(frames))])
        if s.gt_z.nunique() > 2:
            vals.append(fn(s.gt_z.to_numpy(), s[col].to_numpy()))
    return [float(q) for q in np.percentile(vals, [2.5, 97.5])]


def main():
    ev = json.loads((DB / "eval_boxes.json").read_text())
    avail = [m for m in MODELS if (DB / f"{m}.json").exists()]
    res = {"models": avail, "zod": {}, "rendered": {}, "dashcam": {}}
    data = {m: json.loads((DB / f"{m}.json").read_text()) for m in avail}
    for fov in ("54.4", "70"):
        base = pd.DataFrame(ev[f"zod_{fov}"])
        base["head"], base["g1"], base["oracle"] = base.pred_head, base.z_g1, base.z_oracle
        for m in avail:
            base[m] = [r["pred"] for r in data[m][f"zod_{fov}"]]
        for gname, g in (("hoofed", base[base.tag == "ungulate"]), ("deer_like", base[base.species.isin(["reindeer", "deer_or_reindeer"])]),
                         ("all", base)):
            entry = {"n": int(len(g)), "frames": int(g.frame.nunique())}
            for col in ["head", "g1", "oracle"] + avail:
                z, p = g.gt_z.to_numpy(), g[col].to_numpy()
                entry[col] = {"median_rel": med_rel(z, p), "slope": slope(z, p)}
                if gname != "all":
                    entry[col]["median_rel_ci"] = cluster_ci(g, col, med_rel)
                    entry[col]["slope_ci"] = cluster_ci(g, col, slope)
            res["zod"][f"{gname}_{fov}"] = entry
    # rendered roads at ground-truth boxes
    rend = {"head": pd.DataFrame(json.loads((DB / "head_gtbox_rendered.json").read_text())),
            "g1": pd.DataFrame([{"tag": r["tag"], "gt_z": r["z"], "pred": r["z_g1"]} for r in json.loads((ROOT / "output/geo_range/loso_rows.json").read_text())])}
    for m in avail:
        rend[m] = pd.DataFrame(data[m]["rendered"])
    for name, d in rend.items():
        e = {"n": int(len(d)), "median_rel": med_rel(d.gt_z.to_numpy(), d.pred.to_numpy()), "slope": slope(d.gt_z.to_numpy(), d.pred.to_numpy())}
        for lo, hi in BANDS:
            b = d[(d.gt_z >= lo) & (d.gt_z < hi)]
            err = (b.pred - b.gt_z).abs()
            e[f"{lo}-{hi}"] = {"mae": float(err.mean()), "within5": float((err <= 5).mean())}
        res["rendered"][name] = e
    # dashcam scale-free slope (pooled within-track, tracks with box-height ratio >= 1.5)
    for m in avail:
        lz, lh = [], []
        for tr in data[m]["dashcam"]:
            h = np.array(tr["h"], float); p = np.clip(np.array(tr["pred"]), 0.5, None)
            if h.max() / h.min() < 1.5:
                continue
            a, b = np.log(1 / h), np.log(p)
            lz += list(b - b.mean()); lh += list(a - a.mean())
        res["dashcam"][m] = float(np.dot(lh, lz) / np.dot(lh, lh))
    real = json.loads((ROOT / "output/geo_range/real_eval.json").read_text())
    res["dashcam"]["g1"] = real["dashcam"]["median_elasticity"]
    res["dashcam"]["head"] = 0.21
    (DB / "summary.json").write_text(json.dumps(res, indent=2))
    for key in ("hoofed_70", "deer_like_70", "hoofed_54.4"):
        z = res["zod"][key]
        print(f"== ZOD {key} (n={z['n']}, frames={z['frames']})")
        for col in ["head", "g1"] + avail + ["oracle"]:
            r = z[col]
            print(f"   {LABEL[col]:36s} err {100*r['median_rel']:5.1f}% [{100*r['median_rel_ci'][0]:.0f}-{100*r['median_rel_ci'][1]:.0f}]  "
                  f"slope {r['slope']:5.2f} [{r['slope_ci'][0]:.2f},{r['slope_ci'][1]:.2f}]")
    print("== rendered roads (GT boxes)")
    for name, e in res["rendered"].items():
        print(f"   {LABEL[name]:36s} n={e['n']:5d} err {100*e['median_rel']:5.1f}% slope {e['slope']:.2f} | " +
              " ".join(f"{b}: MAE {e[b]['mae']:.2f} w5 {e[b]['within5']:.2f}" for b in ("30-40", "40-50", "50-60")))
    print("== dashcam within-track slope:", {LABEL[k]: round(v, 2) for k, v in res["dashcam"].items()})


if __name__ == "__main__":
    main()
