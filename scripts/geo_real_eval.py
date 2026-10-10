#!/usr/bin/env python3
"""Apply the geometry-anchored range (trained on all ten rendered roads) to real data.

ZOD: virtual pinhole views with known focal length; ground truth = LiDAR camera-frame depth.
     Reported by tagged species group, against the learned range head and the size oracle.
Dashcam clips (unknown intrinsics): within-track stability of the estimated height H (scale-free).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from geo_range import OUT, HeightNet, crop_box, predict_height  # noqa: E402
from zod_range_eval import ZOD, animals, virtual_camera  # noqa: E402
from e5_real_consistency import build_tracks, FRAMES, LABELS  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def stats(z_hat: np.ndarray, z: np.ndarray) -> dict:
    ok = np.isfinite(z_hat) & (z_hat > 0)
    z_hat, z = z_hat[ok], z[ok]
    e = z_hat - z
    return {"n": int(len(z)), "mae_m": float(np.abs(e).mean()), "bias_m": float(e.mean()),
            "median_abs_rel": float(np.median(np.abs(e) / z)), "within_10pct": float((np.abs(e) / z <= 0.10).mean()),
            "within_20pct": float((np.abs(e) / z <= 0.20).mean()),
            "elasticity": float(np.polyfit(np.log(z), np.log(z_hat), 1)[0])}


def zod_eval(net: HeightNet, h0: float, hfov: float) -> dict:
    rows = pd.read_csv(ROOT / f"output/zod/range_rows_c8_hfov{hfov:g}.csv")
    tags = pd.read_csv(ROOT / "output/zod/species_tags.csv")
    recs = []
    for fid, grp in rows.groupby("frame"):
        fid_s = f"{int(fid):06d}"
        calib = json.loads((ZOD / fid_s / "calibration.json").read_text())
        K, D, P, _, _ = virtual_camera(calib, hfov)
        view = cv2.imread(str(ROOT / f"output/zod/views/hfov{hfov:g}/{fid_s}.jpg"))
        anns = animals(fid_s, calib, K, D, P)
        for _, r in grp.iterrows():
            a = min(anns, key=lambda q: abs(q["gt_z"] - r.gt_z))
            recs.append({**r.to_dict(), "box": a["box"], "crop": crop_box(view, a["box"]), "f": P[0, 0]})
    hh = predict_height(net, [x["crop"] for x in recs], [(x["box"][2] - x["box"][0]) / max(x["box"][3] - x["box"][1], 1e-3) for x in recs])
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "crop"} for x in recs])
    df["H_g1"] = hh
    df["z_g0"] = df.f * h0 / df.box_h
    df["z_g1"] = df.f * df.H_g1 / df.box_h
    tag_key = tags.assign(gt_r=tags.gt_z.round(2))[["frame", "gt_r", "tag", "species"]]
    df = df.assign(gt_r=df.gt_z.round(2)).merge(tag_key, on=["frame", "gt_r"], how="left")
    df.drop(columns=["box"]).to_csv(ROOT / f"output/geo_range/zod_rows_hfov{hfov:g}.csv", index=False)
    groups = {"hoofed (tagged)": df[df.tag == "ungulate"],
              "deer-like (reindeer/deer)": df[df.species.isin(["reindeer", "deer_or_reindeer"])],
              "horse/cattle": df[df.species.isin(["horse", "cattle"])], "sheep": df[df.species == "sheep"],
              "dogs (tagged)": df[df.tag == "dog"], "all animals in view": df}
    out = {}
    for name, g in groups.items():
        z = g.gt_z.to_numpy()
        out[name] = {"head": stats(g.pred_gtcenter.to_numpy(), z), "G0": stats(g.z_g0.to_numpy(), z),
                     "G1": stats(g.z_g1.to_numpy(), z), "size_oracle": stats(g.base_size_oracle.to_numpy(), z),
                     "H_g1_median": float(g.H_g1.median()) if len(g) else None,
                     "H_true_median": float(g.hgt.median()) if len(g) else None}
        b = g[(g.gt_z >= 30) & (g.gt_z < 60)]
        if len(b) >= 3:
            out[name]["30-60m"] = {m: stats(b[c].to_numpy(), b.gt_z.to_numpy()) for m, c in
                                   (("head", "pred_gtcenter"), ("G0", "z_g0"), ("G1", "z_g1"), ("size_oracle", "base_size_oracle"))}
    return out


def dashcam_eval(net: HeightNet) -> dict:
    tracks = build_tracks(json.loads(LABELS.read_text()))
    res = []
    for tr in tracks:
        crops, asp = [], []
        for d in tr:
            img = cv2.imread(str(FRAMES / d["frame"]))
            box = [d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]]
            crops.append(crop_box(img, box)); asp.append(d["w"] / max(d["h"], 1))
        H = predict_height(net, crops, asp)
        h = np.array([d["h"] for d in tr], float)
        z_rel = H / h                                                    # range up to the unknown focal length
        slope = np.polyfit(np.log(1 / h), np.log(z_rel), 1)[0] if np.ptp(h) > 0 else np.nan
        res.append({"n": len(tr), "h_ratio": float(h.max() / h.min()), "H_cv": float(H.std() / H.mean()),
                    "H_median": float(np.median(H)), "elasticity": float(slope)})
    inf = [r for r in res if r["h_ratio"] >= 1.5]
    return {"tracks": res, "informative": len(inf),
            "median_elasticity": float(np.median([r["elasticity"] for r in inf])),
            "median_H_cv": float(np.median([r["H_cv"] for r in inf]))}


def main() -> None:
    net = HeightNet().to(DEVICE); net.load_state_dict(torch.load(OUT / "heightnet_all.pt")); net.eval()
    h0 = json.loads((OUT / "h0_all.json").read_text())["H0"]
    result = {"H0": h0, "zod": {f"hfov{f:g}": zod_eval(net, h0, f) for f in (54.4, 70)}, "dashcam": dashcam_eval(net)}
    (OUT / "real_eval.json").write_text(json.dumps(result, indent=2))
    for fov, groups in result["zod"].items():
        print(f"== ZOD {fov}")
        for name, g in groups.items():
            line = "  ".join(f"{m} MAE {g[m]['mae_m']:.1f} rel {g[m]['median_abs_rel']:.2f} el {g[m]['elasticity']:.2f}"
                             for m in ("head", "G0", "G1", "size_oracle") if g[m]["n"] >= 3)
            print(f"  {name:26s} n={g['G1']['n']:4d} | {line} | H_g1 {g['H_g1_median']:.2f} H_true {g['H_true_median']:.2f}")
    dc = result["dashcam"]
    print(f"== dashcam: informative tracks {dc['informative']}, median elasticity {dc['median_elasticity']:.2f}, median H CV {dc['median_H_cv']:.3f}")


if __name__ == "__main__":
    main()
