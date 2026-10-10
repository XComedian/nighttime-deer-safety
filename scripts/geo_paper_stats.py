#!/usr/bin/env python3
"""Paper statistics for the range-method comparison on the ten held-out rural roads (reproducible run).

Reads output/geo_range/loso_detected_comparison.json (geo_eval.py) and reports, per band and method:
mean recall / matched MAE / joint success, roads passing the 0.60 joint screen, road-resampling 95% intervals,
joint envelopes per road, and a joint-tolerance sensitivity from the same matched detections.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/geo_range"
BANDS = ["30-40", "40-50", "50-60"]
METHODS = ["head", "g0", "g1"]


def envelope(js: list[float]) -> int:
    env = 0
    for b, j in zip((40, 50, 60), js):
        if j < 0.60:
            break
        env = b
    return env


def main() -> None:
    d = json.loads((OUT / "loso_detected_comparison.json").read_text())["per_fold"]
    folds = list(d)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(folds), size=(100000, len(folds)))
    res = {}
    for m in METHODS:
        res[m] = {}
        for b in BANDS:
            j = np.array([d[t][b][m]["joint"] for t in folds])
            mae = np.array([d[t][b][m]["mae"] for t in folds])
            rec = np.array([d[t][b]["recall"] for t in folds])
            boot = j[idx].mean(1)
            res[m][b] = {"recall": float(rec.mean()), "mae": float(mae.mean()),
                         "mae_ci": [float(q) for q in np.percentile(mae[idx].mean(1), [2.5, 97.5])],
                         "joint": float(j.mean()), "joint_ci": [float(q) for q in np.percentile(boot, [2.5, 97.5])],
                         "roads_pass": int((j >= 0.60).sum())}
        res[m]["per_road_joint_envelope"] = {t: envelope([d[t][b][m]["joint"] for b in BANDS]) for t in folds}
        envs = list(res[m]["per_road_joint_envelope"].values())
        res[m]["joint_envelope_counts"] = {str(e): envs.count(e) for e in (60, 50, 40, 0)}
        res[m]["mean_joint_envelope"] = envelope([res[m][b]["joint"] for b in BANDS])
        res[m]["lower_ci_joint_envelope"] = envelope([res[m][b]["joint_ci"][0] for b in BANDS])
    (OUT / "paper_stats.json").write_text(json.dumps(res, indent=2))
    for m in METHODS:
        print(m, {b: (round(res[m][b]["joint"], 3), [round(x, 3) for x in res[m][b]["joint_ci"]], res[m][b]["roads_pass"],
                      round(res[m][b]["mae"], 2)) for b in BANDS},
              "envelopes", res[m]["joint_envelope_counts"], "mean-env", res[m]["mean_joint_envelope"],
              "lowerCI-env", res[m]["lower_ci_joint_envelope"])


if __name__ == "__main__":
    main()
