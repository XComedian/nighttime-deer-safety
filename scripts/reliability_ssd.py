#!/usr/bin/env python3
"""Reliability-based stopping-distance screening for camera-only deer warning (Monte Carlo).

A vehicle at constant speed V approaches a stationary deer in its path; frames arrive at 10 Hz from 150 m.
Per frame the deer is detected with the measured band probability (0 beyond 60 m, the rendered-data extent),
and its range is estimated as Z_hat = D * exp(eps). eps is PERSISTENT per animal (drawn once per trial):
most range error comes from the animal's size, which does not average out over frames.
The system warns at the first detected frame with Z_hat * exp(-q) <= T(V), where T is the design stopping
sight distance (AASHTO: 2.5 s + 0.30 s machine budget, 3.4 m/s^2 i.e. f_r = 0.35, scenario grade).
The driver then responds with perception-brake time t ~ lognormal (mean 1.1 s, 95th pct 2.0 s; Fambro et al. 1998)
plus the 0.30 s machine budget, and brakes at a ~ N(0.55 g, 0.07 g) truncated at 0.25 g (same source).
Non-compliance (late warning) occurs when the actual stopping distance exceeds the distance left at the warning.
P_nc(V) = P(non-compliance); the operating speed at risk level r is the largest V with P_nc(V) <= r.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "output/geo_range"
OUT = ROOT / "output/reliability_ssd"
G = 9.81
FPS, START_M, LAT = 10.0, 150.0, 0.30
BANDS = [(30, 40), (40, 50), (50, 60)]
# lognormal perception-brake time: mean 1.1 s, 95th percentile 2.0 s
PRT_SIGMA = 0.4165
PRT_MU = np.log(1.1) - PRT_SIGMA ** 2 / 2
MPH = 0.44704


def design_ssd(v_mph: float, grade: float) -> float:
    return (1.47 * v_mph * (2.5 + LAT) + v_mph ** 2 / (30 * (0.35 + grade))) * 0.3048


def detection_prob(d: np.ndarray, recall: list[float]) -> np.ndarray:
    p = np.where(d < 40, recall[0], np.where(d < 50, recall[1], recall[2]))
    return np.where(d > 60, 0.0, p)            # no evidence beyond the 60 m rendered extent


class ErrorModel:
    """Persistent per-animal log range error: banded empirical quantiles (rendered) or pooled (real)."""

    def __init__(self, banded: dict[int, np.ndarray] | None = None, pooled: np.ndarray | None = None,
                 absolute: np.ndarray | None = None):
        # absolute: estimates that do not scale with distance (learned head on real images, slope ~0);
        # each animal draws one persistent estimate from the real predictions.
        self.banded, self.pooled, self.absolute = banded, pooled, absolute

    def eps(self, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self.absolute is not None:
            return np.log(np.quantile(self.absolute, u) / d)
        if self.pooled is not None:
            return np.quantile(self.pooled, u)
        bi = np.clip(((d - 30) // 10).astype(int), 0, 2)
        out = np.empty_like(d)
        for b in range(3):
            m = bi == b
            out[m] = np.quantile(self.banded[b], u[m])
        return out


def simulate(v_mph: float, grade: float, err: ErrorModel | None, recall, q: float = 0.0, n: int = 100000, seed: int = 0,
             oracle_detect: bool = False, block: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    v = v_mph * MPH
    T = design_ssd(v_mph, grade)
    dists = np.arange(START_M, 0, -v / FPS)                    # true distance at each frame
    u = rng.uniform(0.005, 0.995, n)                            # per-animal error quantile (persistent)
    warn_d = np.full(n, 0.0)
    pending = np.ones(n, bool)
    draw = rng.random(n)
    for k, d in enumerate(dists):
        if k % block == 0:                                      # detection outcome persists for `block` frames
            draw = rng.random(n)
        det = np.full(n, d <= 60) if oracle_detect else draw < detection_prob(np.array([d]), recall)[0]
        z_hat = d * (np.exp(err.eps(u, np.full(n, d))) if err is not None else 1.0)
        fire = pending & det & (z_hat * np.exp(-q) <= T)
        warn_d[fire] = d
        pending &= ~fire
        if not pending.any():
            break
    t = np.exp(rng.normal(PRT_MU, PRT_SIGMA, n)) + LAT
    a = np.clip(rng.normal(0.55, 0.07, n), 0.25, None) * G + G * grade      # grade < 0 (downgrade) reduces deceleration
    stop = v * t + v ** 2 / (2 * np.clip(a, 0.5, None))
    fail = stop > warn_d                                       # includes never-warned (warn_d = 0)
    early = warn_d - stop
    return {"p_nc": float(fail.mean()), "p_no_warning": float((warn_d == 0).mean()),
            "median_warning_m": float(np.median(warn_d[warn_d > 0])) if (warn_d > 0).any() else 0.0,
            "median_spare_m": float(np.median(early[~fail])) if (~fail).any() else 0.0,
            "p_early_20m": float((early > 20).mean()), "design_ssd_m": T}


def load_models() -> tuple[dict, list[float]]:
    stats = json.loads((GEO / "paper_stats.json").read_text())
    recall = [stats["g1"][f"{lo}-{hi}"]["recall"] for lo, hi in BANDS]
    # rendered G1 / head errors on matched detections, by band (leave-one-road-out)
    banded = {"g1": {b: [] for b in range(3)}, "head": {b: [] for b in range(3)}}
    for raw in sorted((GEO / "e6_raw").glob("*.json")):
        for fr in json.loads(raw.read_text()):
            for dd in fr["dets"]:
                if dd["match"] is None:
                    continue
                z = fr["gt_m"][dd["match"]]
                if not 30 <= z < 60:
                    continue
                b = int((z - 30) // 10)
                banded["g1"][b].append(np.log(dd["g1"] / z)); banded["head"][b].append(np.log(dd["head"] / z))
    models = {"rendered_g1": ErrorModel(banded={b: np.array(v) for b, v in banded["g1"].items()}),
              "rendered_head": ErrorModel(banded={b: np.array(v) for b, v in banded["head"].items()})}
    z = pd.read_csv(GEO / "zod_rows_hfov70.csv"); h = z[z.tag == "ungulate"]
    models["real_g1"] = ErrorModel(pooled=np.log(h.z_g1 / h.gt_z).to_numpy())
    models["real_head"] = ErrorModel(absolute=h.pred_gtcenter.to_numpy())
    sp = h.copy(); sp["sg"] = sp.species.replace({"deer_or_reindeer": "reindeer"})
    Hs = [sp[(sp.sg == r.sg) & (sp.frame != r.frame)].hgt.mean() for r in sp.itertuples()]
    models["real_species"] = ErrorModel(pooled=np.log((sp.f * np.array(Hs) / sp.box_h) / sp.gt_z).to_numpy())
    return models, recall


def operating_speed(curve: dict[float, float], risk: float) -> float | None:
    ok = [v for v, p in sorted(curve.items()) if p <= risk]
    return max(ok) if ok else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models, recall = load_models()
    speeds = list(range(15, 46, 1))
    scenarios = {
        "perfect perception (to 60 m)": dict(err=None, oracle_detect=True, q=0.0, recall=[1, 1, 1]),
        "rendered G1": dict(err=models["rendered_g1"], q=0.0),
        "real G1": dict(err=models["real_g1"], q=0.0),
        "real G1 + buffer q90": dict(err=models["real_g1"], q=float(np.quantile(models["real_g1"].pooled, 0.9))),
        "real species-mean height": dict(err=models["real_species"], q=0.0),
        "real range head": dict(err=models["real_head"], q=0.0),
    }
    scenarios["real G1 + buffer q90, correlated detection (0.5 s)"] = dict(err=models["real_g1"], q=scenarios["real G1 + buffer q90"]["q"], block=5)
    res = {"recall": recall, "prt": {"mu": PRT_MU, "sigma": PRT_SIGMA}, "q90": scenarios["real G1 + buffer q90"]["q"], "curves": {}, "speeds": {}}
    for grade, gname in ((0.0, "level"), (-0.05, "down5"), (-0.08, "down8")):
        for name, cfg in scenarios.items():
            cfg = dict(cfg); rc = cfg.pop("recall", recall)
            curve = {}
            for v in speeds:
                curve[v] = simulate(v, grade, recall=rc, **cfg)
            key = f"{name} | {gname}"
            res["curves"][key] = curve
            pc = {v: c["p_nc"] for v, c in curve.items()}
            res["speeds"][key] = {f"risk_{r}": operating_speed(pc, r) for r in (0.01, 0.05, 0.10)}
            if True:
                pe = {v: c["p_early_20m"] for v, c in curve.items()}
                print(f"{key:45s} P_nc@20/25/30/35 = {pc[20]:.3f}/{pc[25]:.3f}/{pc[30]:.3f}/{pc[35]:.3f} early>20m@25 {pe[25]:.2f} | "
                      f"V(r<=1%) {res['speeds'][key]['risk_0.01']}  V(r<=5%) {res['speeds'][key]['risk_0.05']}  V(r<=10%) {res['speeds'][key]['risk_0.1']}", flush=True)
    (OUT / "reliability_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
