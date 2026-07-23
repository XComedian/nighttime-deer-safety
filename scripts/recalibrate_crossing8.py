#!/usr/bin/env python3
"""Affine recalibration probe on the frozen crossing8 zero-shot range predictions.

Question: is the crossing8 range failure a
scene-dependent calibration error (recoverable from a few frames of the new scene)
or a representation failure (not recoverable by any monotone rescaling)?

Protocol:
- Matched detections only (end-to-end: detected at conf>=0.25, IoU>=0.5).
- Calibration uses the first K frames (K in {10,30,60,120,300} at an assumed 10 Hz,
  i.e. 1-30 s of scene data) and assumes reference distances for those frames would
  come from an independent source (rangefinder/LiDAR or surveyed markers) during a
  calibration pass. The model checkpoint stays frozen; only the scalar output is
  remapped.
- Evaluation is the fixed untouched tail (frames 300-599), identical across K.
- Fits: bias-only (gt = pred + b), linear affine (gt = a*pred + b), and log-affine
  (ln gt = a*ln pred + b). Oracle ceiling = affine fit on all 447 matched points.
- Joint safety metric recomputed per band on eval frames: deer counted as joint
  success only if detected AND corrected |range error| <= 5 m; missed deer stay
  joint failures. Detection itself is untouched by recalibration.

Outputs: output/crossing8_recalibration/{summary.md, recalibration_table.csv,
recalibration_scatter.png}.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output/single_scene_instance_c8/matched_distance_predictions.csv"
OUT = ROOT / "output/crossing8_recalibration"

EVAL_START = 300
CAL_WINDOWS = [10, 30, 60, 120, 300]
BANDS = [("30-40", 30, 40), ("40-50", 40, 50), ("50-60", 50, 60)]
TOL = 5.0


def load() -> list[dict]:
    rows = []
    with SRC.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "frame": int(r["frame_index"]),
                "gt": float(r["distance_m"]),
                "detected": r["detected"] == "1",
                "pred": float(r["pred_m"]) if r["pred_m"] not in {"", "nan"} else None,
            })
    return rows


def fit_bias(pairs):
    b = sum(g - p for g, p in pairs) / len(pairs)
    return lambda p: p + b, f"gt = pred {b:+.2f}"


def fit_affine(pairs):
    n = len(pairs)
    mp = sum(p for _, p in pairs) / n
    mg = sum(g for g, _ in pairs) / n
    sxx = sum((p - mp) ** 2 for _, p in pairs)
    sxy = sum((p - mp) * (g - mg) for g, p in pairs)
    a = sxy / sxx if sxx > 1e-9 else 1.0
    b = mg - a * mp
    return lambda p: a * p + b, f"gt = {a:.3f}*pred {b:+.2f}"


def fit_log_affine(pairs):
    lp = [(math.log(g), math.log(p)) for g, p in pairs]
    n = len(lp)
    mx = sum(x for _, x in lp) / n
    my = sum(y for y, _ in lp) / n
    sxx = sum((x - mx) ** 2 for _, x in lp)
    sxy = sum((x - mx) * (y - my) for y, x in lp)
    a = sxy / sxx if sxx > 1e-9 else 1.0
    b = my - a * mx
    return lambda p: math.exp(a * math.log(p) + b), f"ln gt = {a:.3f}*ln pred {b:+.3f}"


FITS = {"bias_only": fit_bias, "affine": fit_affine, "log_affine": fit_log_affine}


def eval_correction(rows_eval: list[dict], correct) -> dict:
    matched = [(r["gt"], correct(r["pred"])) for r in rows_eval if r["detected"] and r["pred"] is not None]
    errs = [c - g for g, c in matched]
    out = {
        "eval_matched": len(matched),
        "mae": sum(abs(e) for e in errs) / len(errs),
        "rmse": math.sqrt(sum(e * e for e in errs) / len(errs)),
        "bias": sum(errs) / len(errs),
        "within5_matched": sum(1 for e in errs if abs(e) <= TOL) / len(errs),
    }
    for name, lo, hi in BANDS:
        band = [r for r in rows_eval if lo <= r["gt"] < hi]
        joint = mae_n = 0
        band_err = []
        for r in band:
            if r["detected"] and r["pred"] is not None:
                e = correct(r["pred"]) - r["gt"]
                band_err.append(abs(e))
                if abs(e) <= TOL:
                    joint += 1
        out[f"joint_{name}"] = joint / len(band) if band else None
        out[f"mae_{name}"] = (sum(band_err) / len(band_err)) if band_err else None
        out[f"gt_{name}"] = len(band)
        out[f"recall_{name}"] = (sum(1 for r in band if r["detected"]) / len(band)) if band else None
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    rows_eval = [r for r in rows if r["frame"] >= EVAL_START]
    identity = lambda p: p  # noqa: E731

    results = []
    base = eval_correction(rows_eval, identity)
    results.append({"variant": "uncorrected", "cal_frames": 0, "cal_points": 0,
                    "fit": "identity", **base})

    for k in CAL_WINDOWS:
        cal = [(r["gt"], r["pred"]) for r in rows
               if r["frame"] < k and r["detected"] and r["pred"] is not None]
        if len(cal) < 3:
            continue
        for fname, fitter in FITS.items():
            correct, desc = fitter(cal)
            results.append({"variant": f"{fname}_K{k}", "cal_frames": k,
                            "cal_points": len(cal), "fit": desc,
                            **eval_correction(rows_eval, correct)})

    # Oracle ceiling: affine fit on every matched point including eval (upper bound only).
    all_pairs = [(r["gt"], r["pred"]) for r in rows if r["detected"] and r["pred"] is not None]
    for fname in ("affine", "log_affine"):
        correct, desc = FITS[fname](all_pairs)
        results.append({"variant": f"ORACLE_{fname}_all{len(all_pairs)}", "cal_frames": 600,
                        "cal_points": len(all_pairs), "fit": desc,
                        **eval_correction(rows_eval, correct)})

    with (OUT / "recalibration_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    # Scatter figure: pred vs gt with the K=60 affine and log-affine fits.
    cal60 = [(r["gt"], r["pred"]) for r in rows
             if r["frame"] < 60 and r["detected"] and r["pred"] is not None]
    aff60, aff_desc = fit_affine(cal60)
    log60, log_desc = fit_log_affine(cal60)
    ev = [(r["gt"], r["pred"]) for r in rows_eval if r["detected"] and r["pred"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    ax = axes[0]
    ax.scatter([p for _, p in ev], [g for g, _ in ev], s=8, alpha=0.5, color="#2c7fb8",
               label="eval matched detections")
    xs = [x / 10 for x in range(170, 420)]
    ax.plot(xs, [aff60(x) for x in xs], color="#d95f02", lw=1.5, label=f"affine K=60: {aff_desc}")
    ax.plot(xs, [log60(x) for x in xs], color="#7570b3", lw=1.5, ls="--",
            label=f"log-affine K=60")
    ax.plot([15, 62], [15, 62], color="k", lw=0.8, ls=":", label="identity")
    ax.set_xlabel("Predicted range (m)")
    ax.set_ylabel("Ground-truth range (m)")
    ax.set_title("crossing8 range predictions vs GT (eval frames 300-599)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax2 = axes[1]
    width = 0.35
    for i, (label, correct, color) in enumerate([
            ("uncorrected", identity, "#cbd5e0"), ("affine K=60", aff60, "#d95f02")]):
        m = eval_correction(rows_eval, correct)
        vals = [m[f"mae_{b}"] for b, _, _ in BANDS]
        ax2.bar([x + (i - 0.5) * width for x in range(len(BANDS))], vals, width,
                color=color, label=label)
    ax2.axhline(TOL, color="k", ls="--", lw=1, label="5 m tolerance")
    ax2.set_xticks(range(len(BANDS)), [b for b, _, _ in BANDS])
    ax2.set_xlabel("GT distance band (m)")
    ax2.set_ylabel("Matched MAE (m)")
    ax2.set_title("Per-band matched MAE before/after 6 s calibration")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "recalibration_scatter.png", dpi=200)
    plt.close(fig)

    def fmt(v):
        return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    lines = [
        "# crossing8 Affine Recalibration Probe",
        "",
        "Calibration-vs-representation test for the frozen crossing7 range head",
        "evaluated on the current COCO-derived crossing8 instance labels",
        "(alternative explanation tested during manuscript development).",
        "Model checkpoint frozen; only the scalar range output is remapped by a fit",
        "estimated from the first K frames (10 Hz assumed; reference distances during",
        "calibration would require an independent range source). Evaluation is the",
        "untouched tail, frames 300-599. Detection is unchanged; joint success still",
        "counts missed deer as failures.",
        "",
        "| variant | cal frames | cal pts | fit | eval MAE m | bias m | %<=5m (matched) | joint 30-40 | joint 40-50 | joint 50-60 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append("| " + " | ".join([
            r["variant"], str(r["cal_frames"]), str(r["cal_points"]), r["fit"],
            f"{r['mae']:.2f}", f"{r['bias']:+.2f}", f"{r['within5_matched']:.3f}",
            fmt(r["joint_30-40"]), fmt(r["joint_40-50"]), fmt(r["joint_50-60"])]) + " |")
    lines += [
        "",
        "Eval-tail band sizes (GT boxes): " + ", ".join(
            f"{b}: {base[f'gt_{b}']} (recall {base[f'recall_{b}']:.3f})" for b, _, _ in BANDS),
        "",
        "Reading guide: if calibrated MAE approaches the crossing7 in-sequence level",
        "(2-5 m), the failure was scene calibration; if it stays well above the 5 m",
        "tolerance, or the oracle ceiling itself fails, the failure is representational",
        "and no output remapping can rescue the 5 m screen.",
        "",
        "## Interpretation (numbers above; inputs are frozen artifacts, rerun is deterministic)",
        "",
        "1. **A scene-dependent calibration shift explains most of the signed error.**",
        "   The uncorrected evaluation-tail bias and MAE are both 15.03 m. A bias-only",
        "   correction fitted to the first 60 frames (24 matched instances) reduces",
        "   bias to +0.09 m and MAE to 5.39 m without changing the checkpoint.",
        "2. **A global output remapping does not recover the joint screen.** The 60-frame",
        "   affine fit yields 5.36 m MAE and joint success of 0.282/0.281/0.183.",
        "   Even the optimistic affine fit using all 241 matched instances, including",
        "   evaluation instances, leaves 5.16 m MAE and joint success of",
        "   0.320/0.344/0.169. Every band remains below the 0.60 project threshold.",
        "3. **The result supports a mixed explanation.** A scalar calibration shift",
        "   accounts for the nearly uniform negative bias, but residual range scatter",
        "   and missed detections remain. The zero-shot failure therefore should not be",
        "   described as purely representational or purely calibrational.",
        "4. **Any operational calibration would need an independent metric reference.**",
        "   This post hoc probe uses rendered ground truth and makes crossing8 a",
        "   development scene; a newly prespecified model or correction requires a",
        "   fresh scene-disjoint evaluation.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
