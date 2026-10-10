#!/usr/bin/env python3
"""Write the revised paper's result tables (LaTeX) straight from the result files, so no number is retyped."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "output/geo_range"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/tables"
NAMES = {"c8": "Rural Baseline", "c9a": "Rural Straight", "c9b": "Rural Right Curve", "c9c": "Rural S-Curve",
         "c10": "Rural Crest", "c11": "Rural Farmland", "c12": "Rural Autumn", "c13": "Rural Scrub",
         "c14": "Rural Conifer", "c15": "Rural Shrub"}
BANDS = ["30-40", "40-50", "50-60"]


def ssd_m(v, t=2.8, f=0.35, g=0.0):
    return (1.47 * v * t + v ** 2 / (30 * (f + g))) * 0.3048


def vmax(r, t=2.8, f=0.35, g=0.0):
    lo, hi = 1.0, 120.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ssd_m(mid, t, f, g) <= r else (lo, mid)
    return lo


def f3(x):
    return f"{x:.3f}"


def table_methods(stats):
    rows = []
    for m, name in (("head", "Learned range head"), ("g0", "G0: constant height (1.41 m)"), ("g1", "G1: HeightNet height")):
        cells = []
        for b in BANDS:
            s = stats[m][b]
            cells.append(f"{f3(s['joint'])} ({s['joint_ci'][0]:.2f}--{s['joint_ci'][1]:.2f}) & {s['roads_pass']}/10 & {s['mae']:.2f}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def table_per_road(per_fold):
    rows = []
    for t, name in NAMES.items():
        r = per_fold[t]
        rec = " & ".join(f"{r[b]['recall']:.3f}" for b in BANDS)
        head = " & ".join(f"{r[b]['head']['joint']:.3f}" for b in BANDS)
        g1 = " & ".join(f"{r[b]['g1']['joint']:.3f}" for b in BANDS)
        rows.append(f"{name} & {rec} & {head} & {g1} \\\\")
    return "\n".join(rows)


def table_cue(cue):
    labels = {"cueA": "Markings and posts", "cueB": "Markings only", "cueC": "Posts only", "cueD": "Neither"}
    rows = []
    for k, lab in labels.items():
        r = cue[k]
        rows.append(f"{lab} & {r['30-40']['recall']:.2f} & " +
                    " & ".join(f"{r[b]['head']['mae']:.2f} / {r[b]['head']['joint']:.2f}" for b in BANDS) + " & " +
                    " & ".join(f"{r[b]['g1']['mae']:.2f} / {r[b]['g1']['joint']:.2f}" for b in BANDS) + r" \\")
    return "\n".join(rows)


def table_zod(real):
    rows = []
    groups = [("hoofed (tagged)", "Hoofed animals"), ("deer-like (reindeer/deer)", "Reindeer and deer"),
              ("horse/cattle", "Horses and cattle"), ("sheep", "Sheep"), ("dogs (tagged)", "Dogs")]
    for key, lab in groups:
        a, b = real["zod"]["hfov70"][key], real["zod"]["hfov54.4"][key]
        cell = lambda g, m: f"{100 * g[m]['median_abs_rel']:.0f}\\% / {g[m]['elasticity']:.2f}"  # noqa: E731
        rows.append(f"{lab} & {a['G1']['n']} / {b['G1']['n']} & " +
                    " & ".join(f"{cell(a, m)}" for m in ("head", "G0", "G1", "size_oracle")) + r" \\")
    return "\n".join(rows)


def table_decisions(e6):
    rows = []
    for key, lab in (("point", "Point estimate"), ("conformal80", "Conformal 80\\%"), ("conformal90", "Conformal 90\\%"),
                     ("conformal95", "Conformal 95\\%")):
        cells = []
        for m in ("head", "g1"):
            a = e6[m][key]
            cells.append(f"{a['20mph_level']['miss_rate']:.3f} & {a['20mph_level']['false_alert_rate']:.3f} & "
                         f"{a['25mph_level']['miss_rate']:.3f} & {a['25mph_level']['false_alert_rate']:.3f}")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    a = e6["g1"]["detection_only"]
    rows.append(r"\midrule" + "\n" + f"Detection only (any detection) & \\multicolumn{{4}}{{c}}{{same for both}} & "
                f"{a['20mph_level']['miss_rate']:.3f} & {a['20mph_level']['false_alert_rate']:.3f} & "
                f"{a['25mph_level']['miss_rate']:.3f} & {a['25mph_level']['false_alert_rate']:.3f} \\\\")
    return "\n".join(rows)


def table_speed(r_real):
    rows = []
    for r, lab in ((60, "60 m (rendered-data limit)"), (50, "50 m (G1 joint envelope)"), (40, "40 m (head, lower 95\\% bound)"),
                   (r_real, f"{r_real:.1f} m (G1 envelope with real-data buffer)")):
        rows.append(f"{lab} & {vmax(r):.1f} & {vmax(r, g=-0.05):.1f} & {vmax(r, g=-0.08):.1f} \\\\")
    return "\n".join(rows)


def table_sensitivity(r_real):
    rows = []
    for f in (0.25, 0.30, 0.35, 0.40, 0.45):
        lab = "Friction factor $f$" if f == 0.25 else ""
        val = f"{f:.2f}" + (" (base)" if f == 0.35 else "")
        rows.append(f"{lab} & {val} & {vmax(50, f=f):.1f} & {vmax(50, f=f, g=-0.08):.1f} & {vmax(r_real, f=f):.1f} & {vmax(r_real, f=f, g=-0.08):.1f} \\\\")
    rows.append(r"\midrule")
    for t in (2.5, 2.0, 1.5, 1.0):
        lab = "Driver reaction (s)" if t == 2.5 else ""
        val = f"{t:.1f}" + (" (base)" if t == 2.5 else "")
        rows.append(f"{lab} & {val} & {vmax(50, t=t + 0.3):.1f} & {vmax(50, t=t + 0.3, g=-0.08):.1f} & {vmax(r_real, t=t + 0.3):.1f} & {vmax(r_real, t=t + 0.3, g=-0.08):.1f} \\\\")
    return "\n".join(rows)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    stats = json.loads((GEO / "paper_stats.json").read_text())
    per_fold = json.loads((GEO / "loso_detected_comparison.json").read_text())["per_fold"]
    cue = json.loads((GEO / "cue_ablation_geo.json").read_text())
    real = json.loads((GEO / "real_eval.json").read_text())
    e6 = json.loads((GEO / "e6_geo_summary.json").read_text())
    zstats = json.loads((GEO / "zod_paper_stats.json").read_text())
    q = float(np.mean([zstats["hoofed_54.4"]["G1"]["over_rel_p90"], zstats["hoofed_70"]["G1"]["over_rel_p90"]]))
    r_real = 50 / (1 + q)
    out = {"tab_methods.tex": table_methods(stats), "tab_per_road.tex": table_per_road(per_fold),
           "tab_cue.tex": table_cue(cue), "tab_zod.tex": table_zod(real), "tab_decisions.tex": table_decisions(e6),
           "tab_speed.tex": table_speed(r_real), "tab_sensitivity.tex": table_sensitivity(r_real)}
    for name, body in out.items():
        (DEST / name).write_text(body + "\n")
    from geo_range import HeightNet
    n_params = sum(p.numel() for p in HeightNet().parameters())
    facts = {"q_over_p90_mean": q, "r_real": r_real, "heightnet_params": n_params,
             "speed_r_real_level": vmax(r_real), "speed_r_real_down5": vmax(r_real, g=-0.05), "speed_r_real_down8": vmax(r_real, g=-0.08)}
    (DEST / "facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
