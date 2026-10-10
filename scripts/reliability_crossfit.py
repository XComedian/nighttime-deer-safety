#!/usr/bin/env python3
"""Cross-fitted buffer check: q90 estimated on one half of the ZOD frames, error distribution from the other half."""
import json
import numpy as np
import pandas as pd
from reliability_ssd import ErrorModel, GEO, OUT, load_models, simulate

_, recall = load_models()
z = pd.read_csv(GEO / "zod_rows_hfov70.csv"); h = z[z.tag == "ungulate"].copy()
h["e"] = np.log(h.z_g1 / h.gt_z)
frames = np.array(sorted(h.frame.unique())); rng = np.random.default_rng(0)
res = {}
for rep in range(5):
    perm = rng.permutation(frames); A, B = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
    for fit, test in ((A, B), (B, A)):
        q = float(np.quantile(h[h.frame.isin(fit)].e, 0.9))
        err = ErrorModel(pooled=h[h.frame.isin(test)].e.to_numpy())
        for v in (20, 25, 30, 34, 36):
            res.setdefault(v, []).append(simulate(v, 0.0, err, recall, q=q, n=60000, seed=rep)["p_nc"])
summary = {v: {"mean": float(np.mean(p)), "min": float(np.min(p)), "max": float(np.max(p))} for v, p in res.items()}
(OUT / "crossfit_buffer.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
