#!/usr/bin/env python3
"""Aggregate calibrated LOSO fold results without re-running inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "output/loso_calibrated"
TAGS = ["c8", "c9a", "c9b", "c9c", "c10", "c11", "c12", "c13", "c14", "c15"]
BANDS = ["30-40", "40-50", "50-60"]
SCREEN = 0.60


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "n": len(array),
    }


def main() -> None:
    folds = [json.loads((RESULT_DIR / f"{tag}.json").read_text()) for tag in TAGS]
    summary = {
        "folds": TAGS,
        "screen": SCREEN,
        "protocol": folds[0]["selection_protocol"],
        "bands": {},
    }
    for band in BANDS:
        raw_joint = [fold["raw"][band]["joint_within_5m"] for fold in folds]
        calibrated_joint = [fold["calibrated"][band]["joint_within_5m"] for fold in folds]
        raw_mae = [fold["raw"][band]["matched_mae_m"] for fold in folds]
        calibrated_mae = [fold["calibrated"][band]["matched_mae_m"] for fold in folds]
        recall = [fold["calibrated"][band]["recall"] for fold in folds]
        summary["bands"][band] = {
            "recall": stats(recall),
            "raw_mae_m": stats(raw_mae),
            "calibrated_mae_m": stats(calibrated_mae),
            "raw_joint": stats(raw_joint),
            "calibrated_joint": stats(calibrated_joint),
            "mean_joint_change": float(np.mean(np.asarray(calibrated_joint) - raw_joint)),
            "folds_passing_raw": int(np.sum(np.asarray(raw_joint) >= SCREEN)),
            "folds_passing_calibrated": int(np.sum(np.asarray(calibrated_joint) >= SCREEN)),
            "aggregate_passes": bool(np.mean(calibrated_joint) >= SCREEN),
        }
    contiguous = all(summary["bands"][band]["aggregate_passes"] for band in BANDS)
    summary["calibrated_joint_envelope_m"] = 60 if contiguous else next(
        (int(band.split("-")[0]) for band in BANDS
         if not summary["bands"][band]["aggregate_passes"]),
        60,
    )

    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Training-Scene-Calibrated Leave-One-Scene-Out Evaluation",
        "",
        "For each outer fold, a compact log-range calibrator was selected by",
        "leave-one-training-scene-out validation using only that fold's inner-validation",
        "matches. It was then refit on all inner-validation matches and applied unchanged",
        "to the held-out outer scene. No outer-scene labels were used for calibration.",
        "",
        "| band (m) | recall | raw MAE (m) | calibrated MAE (m) | raw joint <=5 m | calibrated joint <=5 m | passing folds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for band in BANDS:
        row = summary["bands"][band]
        lines.append(
            f"| {band} | {row['recall']['mean']:.3f} +/- {row['recall']['std']:.3f} "
            f"| {row['raw_mae_m']['mean']:.2f} +/- {row['raw_mae_m']['std']:.2f} "
            f"| {row['calibrated_mae_m']['mean']:.2f} +/- {row['calibrated_mae_m']['std']:.2f} "
            f"| {row['raw_joint']['mean']:.3f} +/- {row['raw_joint']['std']:.3f} "
            f"| {row['calibrated_joint']['mean']:.3f} +/- {row['calibrated_joint']['std']:.3f} "
            f"| {row['folds_passing_calibrated']}/{len(TAGS)} |"
        )
    lines += [
        "",
        f"**Aggregate calibrated joint envelope: {summary['calibrated_joint_envelope_m']} m**",
        "",
        "The envelope is an across-fold mean screen. Per-fold heterogeneity remains visible",
        "in the final column and must accompany the aggregate claim.",
    ]
    (RESULT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
