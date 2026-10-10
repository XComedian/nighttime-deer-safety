#!/usr/bin/env python3
"""Summarize E5b ZOD range rows: range head vs geometric baselines, by distance band and lighting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output/zod"
BANDS = [(0, 30), (30, 60), (60, 100), (100, 250)]
PRINT_SUBSET = "rural_large"
METHODS = {"range head (GT box)": "pred_gtcenter", "flat ground": "base_flatground",
           "size prior 1.0 m": "base_size_prior1m", "size oracle (LiDAR height)": "base_size_oracle"}


def stats(d: pd.DataFrame, col: str) -> dict:
    s = d[[col, "gt_z"]].dropna()
    if len(s) < 3:
        return {"n": int(len(s))}
    e = s[col] - s.gt_z
    rel = (e.abs() / s.gt_z)
    lg = np.log(s.gt_z.clip(lower=1)); lp = np.log(s[col].clip(lower=0.5))
    return {"n": int(len(s)), "mae_m": float(e.abs().mean()), "bias_m": float(e.mean()),
            "median_abs_rel": float(rel.median()), "within_5m": float((e.abs() <= 5).mean()),
            "within_10pct": float((rel <= 0.10).mean()),
            "elasticity": float(np.polyfit(lg, lp, 1)[0]) if lg.std() > 0 else float("nan")}


def main(tag: str) -> None:
    d = pd.read_csv(OUT / f"range_rows_{tag}.csv")
    large = d[(d.len >= 1.0) | (d.hgt >= 0.8)]
    res = {"tag": tag, "animals_in_view": int(len(d)), "large_in_view": int(len(large))}
    rural_large = large[large.road_type.isin(["smaller-rural", "arterial-rural", "highway"])]
    res["rural_large_in_view"] = int(len(rural_large))
    for subset_name, sub in [("rural_large", rural_large), ("large", large), ("all", d)]:
        block = {}
        for light, ls in [("all_light", sub), ("day", sub[sub.time_of_day == "day"]),
                          ("night_twilight", sub[sub.time_of_day != "day"])]:
            entry = {"overall": {m: stats(ls, c) for m, c in METHODS.items()}}
            entry["overall"]["recall"] = float(ls.detected.mean()) if len(ls) else None
            for lo, hi in BANDS:
                b = ls[(ls.gt_z >= lo) & (ls.gt_z < hi)]
                be = {m: stats(b, c) for m, c in METHODS.items()}
                be["recall"] = float(b.detected.mean()) if len(b) else None
                jd = b[b.detected == 1]
                be["joint_5m"] = float(((jd.pred_detbox - jd.gt_z).abs() <= 5).sum() / len(b)) if len(b) else None
                entry[f"{lo}-{hi}m"] = be
            block[light] = entry
        res[subset_name] = block
    neg = pd.read_csv(OUT / f"negatives_{tag}.csv")
    res["negatives"] = {"frames": int(len(neg)), "frames_with_detection": float((neg.n_det > 0).mean()),
                        "detections_per_frame": float(neg.n_det.mean())}
    (OUT / f"summary_{tag}.json").write_text(json.dumps(res, indent=2))
    # compact console table: large animals, all lighting
    print(f"{tag}: animals in view {len(d)}, large {len(large)}; negatives FP-frame rate "
          f"{res['negatives']['frames_with_detection']:.3f}")
    for light in ("all_light", "day", "night_twilight"):
        e = res[PRINT_SUBSET][light]
        print(f"  [{light}] recall {e['overall']['recall']}")
        for band in ["overall"] + [f"{lo}-{hi}m" for lo, hi in BANDS]:
            row = e[band]
            cells = []
            for m in METHODS:
                st = row[m]
                cells.append(f"{m.split(' (')[0]}: n={st['n']}" + (f" MAE {st['mae_m']:.1f} bias {st['bias_m']:+.1f} "
                             f"el {st['elasticity']:.2f}" if 'mae_m' in st else ""))
            extra = f" | recall {row['recall']:.2f} joint {row['joint_5m']:.2f}" if band != "overall" and row.get('recall') is not None else ""
            print(f"    {band:8s} " + " ; ".join(cells) + extra)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c8_hfov54.4")
