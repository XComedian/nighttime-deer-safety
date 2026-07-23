#!/usr/bin/env python3
"""Joint-success sensitivity for the corrected crossing8 instance evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "single_scene_instance_c8" / \
    "matched_distance_predictions.csv"
REFERENCE = ROOT / "output" / "single_scene_instance_c8" / "loso_results.json"
OUT = ROOT / "output" / "crossing8_tolerance"
BANDS = [("30-40", 30.0, 40.0), ("40-50", 40.0, 50.0), ("50-60", 50.0, 60.0)]
TOLERANCES = [5.0, 7.5, 10.0]


def main() -> None:
    with SOURCE.open(newline="") as source:
        rows = [
            {
                "distance_m": float(row["distance_m"]),
                "detected": row["detected"] == "1",
                "pred_m": float(row["pred_m"]) if row["pred_m"] else None,
            }
            for row in csv.DictReader(source)
        ]

    records = []
    for tolerance in TOLERANCES:
        for band, lower, upper in BANDS:
            selected = [
                row for row in rows
                if lower <= row["distance_m"] < upper
            ]
            successes = sum(
                row["detected"]
                and row["pred_m"] is not None
                and abs(row["pred_m"] - row["distance_m"]) <= tolerance
                for row in selected
            )
            records.append({
                "tolerance_m": tolerance,
                "band": band,
                "gt_instances": len(selected),
                "joint_successes": successes,
                "joint_rate": successes / len(selected),
            })

    reference = json.loads(REFERENCE.read_text())[0]["bands"]
    for record in records:
        if record["tolerance_m"] == 5.0:
            expected = reference[record["band"]]["joint"]
            if abs(record["joint_rate"] - expected) > 1e-12:
                raise RuntimeError(
                    f"5 m row does not reproduce {record['band']}: "
                    f"{record['joint_rate']} != {expected}"
                )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tolerance_sensitivity.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    with (OUT / "tolerance_sensitivity.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    by_tolerance = {
        tolerance: {
            record["band"]: record for record in records
            if record["tolerance_m"] == tolerance
        }
        for tolerance in TOLERANCES
    }
    lines = [
        "# Corrected crossing8 Joint-Success Tolerance Sensitivity",
        "",
        "Frozen historical single-road checkpoint; current COCO-derived instance labels;",
        "confidence 0.25 and IoU 0.50. Missed deer remain joint failures.",
        "",
        "| tolerance (m) | joint 30-40 | joint 40-50 | joint 50-60 |",
        "|---:|---:|---:|---:|",
    ]
    for tolerance in TOLERANCES:
        values = by_tolerance[tolerance]
        lines.append(
            f"| {tolerance:g} | "
            f"{values['30-40']['joint_rate']:.3f} | "
            f"{values['40-50']['joint_rate']:.3f} | "
            f"{values['50-60']['joint_rate']:.3f} |"
        )
    lines += [
        "",
        "The 5 m row reproduces `output/single_scene_instance_c8/loso_results.json`",
        "exactly. A wider tolerance changes only the range-success condition;",
        "detections, matching, and denominators remain fixed.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
