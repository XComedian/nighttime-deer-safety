#!/usr/bin/env python3
"""Road-resampling uncertainty for the rural LOSO detection/range results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BANDS = ("30-40", "40-50", "50-60")
METRICS = {
    "recall": {"threshold": 0.60, "direction": "ge"},
    "mae": {"threshold": 5.0, "direction": "le"},
    "joint": {"threshold": 0.60, "direction": "ge"},
}


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def passes(values: np.ndarray, metric: str) -> np.ndarray:
    spec = METRICS[metric]
    if spec["direction"] == "ge":
        return values >= spec["threshold"]
    return values <= spec["threshold"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    records = json.loads(args.input.read_text())
    rural = [row for row in records if row["held"] != "c7"]
    if len(rural) != 10:
        raise ValueError(f"expected 10 rural roads, found {len(rural)}")
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(rural), size=(args.replicates, len(rural)))
    rows: list[dict] = []

    for band in BANDS:
        for metric in METRICS:
            values = np.asarray([road["bands"][band][metric] for road in rural], dtype=float)
            sampled = values[indices]
            boot_mean = sampled.mean(axis=1)
            road_pass = passes(values, metric)
            boot_pass_rate = road_pass[indices].mean(axis=1)
            mean_low, mean_high = percentile_interval(boot_mean)
            pass_low, pass_high = percentile_interval(boot_pass_rate)
            rows.append({
                "band": band,
                "metric": metric,
                "n_roads": len(rural),
                "observed_mean": float(values.mean()),
                "mean_ci95_low": mean_low,
                "mean_ci95_high": mean_high,
                "threshold": METRICS[metric]["threshold"],
                "threshold_direction": METRICS[metric]["direction"],
                "roads_passing": int(road_pass.sum()),
                "observed_pass_rate": float(road_pass.mean()),
                "pass_rate_ci95_low": pass_low,
                "pass_rate_ci95_high": pass_high,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "road_bootstrap.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "input": str(args.input),
        "sampling_unit": "held-out rural road",
        "n_roads": len(rural),
        "roads": [row["held"] for row in rural],
        "replicates": args.replicates,
        "seed": args.seed,
        "interval": "percentile 95% road-resampling bootstrap",
        "results": rows,
    }
    (args.output_dir / "road_bootstrap.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Rural LOSO Road-Level Bootstrap",
        "",
        f"Sampling unit: held-out rural road (n={len(rural)}). "
        f"Percentile intervals use {args.replicates:,} resamples with seed {args.seed}.",
        "These intervals describe heterogeneity across the represented roads; they do not "
        "include model-training seed variation or support population-wide safety inference.",
        "",
        "| band | metric | mean (95% CI) | roads passing | pass rate (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['band']} | {row['metric']} | {row['observed_mean']:.3f} "
            f"({row['mean_ci95_low']:.3f}--{row['mean_ci95_high']:.3f}) | "
            f"{row['roads_passing']}/{row['n_roads']} | "
            f"{row['observed_pass_rate']:.2f} "
            f"({row['pass_rate_ci95_low']:.2f}--{row['pass_rate_ci95_high']:.2f}) |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_dir} ({len(rows)} band-metric rows)")


if __name__ == "__main__":
    main()
