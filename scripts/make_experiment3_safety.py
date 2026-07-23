#!/usr/bin/env python3
"""Build Experiment 3 safety-distance table and figure from synthetic eval outputs."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "experiment3_safety"
F_WET = 0.35
T_REACT = 2.5


def ssd_mph(speed_mph: float, grade: float, reaction_s: float = T_REACT, friction: float = F_WET) -> float:
    # AASHTO imperial: SSD(ft)=1.47 V t + V^2/(30(f+G)); convert ft -> m.
    denom = max(0.05, 30.0 * (friction + grade))
    ssd_ft = 1.47 * speed_mph * reaction_s + speed_mph * speed_mph / denom
    return ssd_ft * 0.3048


def read_distance_summary(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["bucket"]] = {
                "n": float(row["n"]),
                "mae_m": float(row["mae_m"]),
                "rmse_m": float(row["rmse_m"]),
                "bias_m": float(row["bias_m"]),
            }
    return rows


def read_detection(path: Path, model: str = "searched_det") -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == model and row["slice"] == "distance":
                rows[row["bucket"]] = {
                    "gt": float(row["gt"]),
                    "recall": float(row["recall_iou50_conf025"]),
                    "ap50": float(row["ap50"]),
                }
    return rows


def reliable_range(det: dict[str, dict[str, float]], dist: dict[str, dict[str, float]],
                   recall_thr: float = 0.60, mae_thr: float = 5.0) -> tuple[float, float]:
    detection_only = 0.0
    combined = 0.0
    bucket_hi = {"0-10": 10, "10-20": 20, "20-30": 30, "30-40": 40, "40-50": 50, "50-60": 60}
    for bucket, hi in bucket_hi.items():
        if det.get(bucket, {}).get("recall", -1) >= recall_thr:
            detection_only = max(detection_only, float(hi))

    seen_combined_pass = False
    for bucket, hi in bucket_hi.items():
        if det.get(bucket, {}).get("recall", -1) >= recall_thr and dist.get(bucket, {}).get("mae_m", 1e9) <= mae_thr:
            seen_combined_pass = True
            combined = max(combined, float(hi))
        elif seen_combined_pass:
            break
    return detection_only, combined


def main() -> None:
    global OUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval", type=Path, default=ROOT / "output" / "synthetic_eval")
    parser.add_argument("--crossing7-eval", type=Path, default=ROOT / "output" / "synthetic_eval_crossing7")
    parser.add_argument("--base-model", default="searched_det")
    parser.add_argument("--crossing7-model", default="searched_det")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    base_eval = args.base_eval
    c7_eval = args.crossing7_eval
    base_dist = read_distance_summary(base_eval / "distance_head_summary.csv")
    base_det = read_detection(base_eval / "detection_slices.csv", args.base_model)
    c7_dist = read_distance_summary(c7_eval / "distance_head_summary.csv")
    c7_det = read_detection(c7_eval / "detection_slices.csv", args.crossing7_model)

    det_base, comb_base = reliable_range(base_det, base_dist)
    det_c7, comb_c7 = reliable_range(c7_det, c7_dist)
    det_range = max(det_base, det_c7)
    comb_range = max(comb_base, comb_c7)

    rows = []
    for speed in [25, 30, 35, 45, 55]:
        for grade_name, grade in [("flat", 0.0), ("-5% downgrade", -0.05), ("-8% downgrade", -0.08)]:
            ssd = ssd_mph(speed, grade)
            rows.append({
                "speed_mph": speed,
                "grade": grade_name,
                "ssd_m": ssd,
                "detection_reliable_m": det_range,
                "combined_reliable_m": comb_range,
                "detection_margin_m": det_range - ssd,
                "combined_margin_m": comb_range - ssd,
            })

    with (OUT / "ssd_safety_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [f"{r['speed_mph']} mph {r['grade']}" for r in rows]
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y, [r["ssd_m"] for r in rows], color="#cbd5e0", label="required SSD")
    ax.axvline(det_range, color="#2b6cb0", lw=2, label=f"detection envelope ({det_range:.0f} m)")
    ax.axvline(comb_range, color="#38a169", lw=2, ls="--", label=f"oracle-localized range screen ({comb_range:.0f} m)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("distance (m)")
    ax.set_title("Experiment 3: Nighttime Deer Detection Range vs Stopping Sight Distance")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ssd_safety_margin.png", dpi=180)
    plt.close(fig)

    md = [
        "# Experiment 3 Safety-Distance Summary",
        "",
        f"Detection envelope: **{det_range:.0f} m** using recall >= 0.60.",
        f"Contiguous oracle-localized range screen: **{comb_range:.0f} m** using range MAE <= 5 m.",
        "",
        "| speed | grade | SSD m | detection margin m | range-screen margin m |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['speed_mph']} | {r['grade']} | {r['ssd_m']:.1f} | "
            f"{r['detection_margin_m']:.1f} | {r['combined_margin_m']:.1f} |"
        )
    md.extend([
        "",
        "Interpretation: crossing7 extends the nighttime synthetic detection validation envelope to 60 m. "
        "The fine-tuned distance head passes the 30-40 m and 50-60 m buckets but misses the 40-50 m "
        "MAE threshold, so the contiguous component-level range screen stops at 40 m. "
        "This is not an end-to-end warning range because the range evaluator samples ground-truth box centers.",
    ])
    (OUT / "experiment3_safety_summary.md").write_text("\n".join(md))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
