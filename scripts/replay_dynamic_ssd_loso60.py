#!/usr/bin/env python3
"""Corrected controller replay using the final c8 LOSO predictions and 60 m gate."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_ssd_advisory import (MPH_TO_MPS, AdvisoryConfig, AdvisoryController,
                                  DeerObservation, EgoState, LatencyBudget, Reason)
from replay_dynamic_ssd import coverage_crossover_mph

FPS = 10.0
SPEEDS = (15, 20, 25, 30, 35)
GRADES = (("flat", 0.0), ("down5", -0.05), ("down8", -0.08))
LATENCY = LatencyBudget(2.5, 0.05, 0.05, 0.2)


def load_frames(path: Path) -> tuple[dict[int, list[dict]], int]:
    frames: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frames[int(row["frame_index"])].append({
                "gt_m": float(row["distance_m"]),
                "detected": row["detected"] == "1",
                "confidence": float(row["detection_confidence"])
                if row["detection_confidence"] else None,
                "pred_m": float(row["pred_m"]) if row["pred_m"] else None,
            })
    return frames, max(frames, default=-1) + 1


def observation(mode: str, boxes: list[dict], ego_mps: float) -> DeerObservation | None:
    if mode == "oracle":
        if not boxes:
            return None
        return DeerObservation(min(box["gt_m"] for box in boxes), 0.0, ego_mps, 1.0)
    detections = [box for box in boxes if box["detected"] and box["pred_m"] is not None]
    if not detections:
        return None
    nearest = min(detections, key=lambda box: box["pred_m"])
    return DeerObservation(
        nearest["pred_m"], 0.0, ego_mps,
        nearest["confidence"] if nearest["confidence"] is not None else 0.0,
    )


def replay(frames: dict[int, list[dict]], n_frames: int, mode: str,
           speed_mph: float, grade: float) -> list[dict]:
    config = AdvisoryConfig(
        r_validated_m=60.0,
        min_persistence_frames=1 if mode == "oracle" else 3,
    )
    controller = AdvisoryController(latency=LATENCY, cfg=config)
    ego = EgoState(speed_mph, grade)
    ego_mps = speed_mph * MPH_TO_MPS
    rows = []
    for frame in range(n_frames):
        obs = observation(mode, frames.get(frame, []), ego_mps)
        decision = controller.step(frame / FPS, ego, obs)
        rows.append({
            "t_s": round(frame / FPS, 2),
            "frame": frame,
            "obs_d_m": round(obs.d_hat_m, 3) if obs else "",
            "ssd_m": round(decision.ssd_required_m, 3),
            "d_eff_m": round(decision.d_effective_m, 3)
            if decision.d_effective_m is not None else "",
            "margin_m": round(decision.margin_m, 3)
            if decision.margin_m is not None else "",
            "state": decision.state.value,
            "raw_state": decision.raw_state.value,
            "reason": decision.reason.value,
            "degraded": int(decision.degraded),
        })
    return rows


def transitions_per_minute(rows: list[dict]) -> float:
    transitions = sum(
        first["state"] != second["state"] for first, second in zip(rows, rows[1:])
    )
    return transitions / (len(rows) / FPS / 60.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_transitions(rows: list[dict]) -> tuple[list[str], int]:
    """Accept dwell holds and the documented pre-persistence CAUTION gate."""
    problems = []
    held_frames = 0
    for previous, current in zip(rows, rows[1:]):
        if current["state"] != previous["state"] and current["state"] != current["raw_state"]:
            persistence_gate = (
                current["state"] == "NORMAL_MONITOR"
                and current["raw_state"] == "CAUTION"
                and current["reason"] == Reason.NOT_PERSISTENT.value
            )
            if current["reason"] != Reason.HELD_BY_DWELL.value and not persistence_gate:
                problems.append(
                    f"frame {current['frame']}: decided {current['state']} != raw "
                    f"{current['raw_state']} without a documented gate "
                    f"(reason {current['reason']})"
                )
    for row in rows:
        if row["reason"] == Reason.HELD_BY_DWELL.value:
            held_frames += 1
        if row["state"] == "BRAKE_NOW" and row["reason"] not in (
            Reason.MARGIN_NEGATIVE.value,
            Reason.TTC_CRITICAL.value,
            Reason.HELD_BY_DWELL.value,
        ):
            problems.append(
                f"frame {row['frame']}: BRAKE_NOW with unexpected reason {row['reason']}"
            )
        if (
            row["state"] == "BRAKE_NOW"
            and row["reason"] == Reason.MARGIN_NEGATIVE.value
            and row["margin_m"] != ""
            and float(row["margin_m"]) > 0
        ):
            problems.append(
                f"frame {row['frame']}: BRAKE_NOW margin {row['margin_m']} > 0"
            )
    return problems, held_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames, n_frames = load_frames(args.input)
    if n_frames != 600:
        raise ValueError(f"expected 600 crossing8 frames, inferred {n_frames}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    total_problems = 0
    for speed in SPEEDS:
        for grade_name, grade in GRADES:
            scenario = f"{speed}mph_{grade_name}"
            metrics[scenario] = {}
            for mode in ("oracle", "predicted_point"):
                replay_mode = "oracle" if mode == "oracle" else "predicted"
                rows = replay(frames, n_frames, replay_mode, speed, grade)
                write_csv(args.output_dir / f"timeline_{mode}_{scenario}.csv", rows)
                problems, held_frames = audit_transitions(rows)
                total_problems += len(problems)
                counts = Counter(row["state"] for row in rows)
                metrics[scenario][mode] = {
                    "state_counts": dict(counts),
                    "state_fractions": {
                        state: counts.get(state, 0) / n_frames
                        for state in ("NORMAL_MONITOR", "CAUTION", "SENSOR_LIMITED",
                                      "SLOW_PREPARE", "BRAKE_NOW")
                    },
                    "transitions_per_minute": transitions_per_minute(rows),
                    "held_frames": held_frames,
                    "audit_problems": problems,
                }

    matched = [
        box for boxes in frames.values() for box in boxes
        if box["detected"] and box["pred_m"] is not None
    ]
    errors = [box["pred_m"] - box["gt_m"] for box in matched]
    prediction_summary = {
        "ground_truth_instances": sum(len(boxes) for boxes in frames.values()),
        "matched_detections": len(matched),
        "matched_mae_m": sum(abs(error) for error in errors) / len(errors),
        "matched_bias_m": sum(errors) / len(errors),
        "predicted_range_min_m": min(box["pred_m"] for box in matched),
        "predicted_range_max_m": max(box["pred_m"] for box in matched),
    }
    payload = {
        "input": str(args.input),
        "coverage_gate_m": 60.0,
        "predicted_uncertainty_m": 0.0,
        "uncertainty_note": "point-prediction replay; no calibrated deployment uncertainty is claimed",
        "n_frames": n_frames,
        "prediction_summary": prediction_summary,
        "flat_coverage_crossover_mph_including_latency": coverage_crossover_mph(60.0),
        "transition_audit_problems": total_problems,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Corrected 60 m Controller Replay",
        "",
        "The final crossing8 LOSO checkpoint is replayed with a 60 m coverage gate. "
        "Predicted mode uses point estimates (sigma=0) because no independently calibrated "
        "uncertainty model exists. The artifact contains matched detections only, so nuisance "
        "alarms caused by unmatched false positives are not measured.",
        "",
        f"Instances: {prediction_summary['ground_truth_instances']}; matched detections: "
        f"{prediction_summary['matched_detections']}; matched MAE "
        f"{prediction_summary['matched_mae_m']:.2f} m; bias "
        f"{prediction_summary['matched_bias_m']:.2f} m.",
        f"Flat-road coverage crossover including the 0.30 s machine latency budget: "
        f"{payload['flat_coverage_crossover_mph_including_latency']:.1f} mph.",
        f"Transition audit problems: {total_problems}.",
        "",
        "| scenario | mode | normal | caution | sensor limited | slow | brake | transitions/min |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, scenario_metrics in metrics.items():
        for mode, values in scenario_metrics.items():
            fractions = values["state_fractions"]
            lines.append(
                f"| {scenario} | {mode} | "
                f"{fractions['NORMAL_MONITOR']:.3f} | {fractions['CAUTION']:.3f} | "
                f"{fractions['SENSOR_LIMITED']:.3f} | {fractions['SLOW_PREPARE']:.3f} | "
                f"{fractions['BRAKE_NOW']:.3f} | "
                f"{values['transitions_per_minute']:.1f} |"
            )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_dir}; transition audit problems: {total_problems}")


if __name__ == "__main__":
    main()
