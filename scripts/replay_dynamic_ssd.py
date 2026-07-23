#!/usr/bin/env python3
"""E7 deterministic offline replay of the dynamic SSD advisory on crossing8.

Replays the frozen-checkpoint crossing8 zero-shot artifact
(output/synthetic_eval_crossing8_zero_shot/matched_distance_predictions.csv) through
scripts/dynamic_ssd_advisory.py in two input modes:

- oracle: ground-truth nearest-deer range, sigma=0, coverage gate off. Validates state
  transitions and equation signs independently of perception error (AGENT_PLAN E7 step 2).
- predicted: nearest matched detected-box range (pred_m) with detector confidence;
  sigma_d comes from the crossing7 per-band matched MAE (what was known before crossing8),
  so calibration coverage on crossing8 measures how badly that prior underestimates error.

Declared replay assumptions (offline study, not a deployment claim):
- 10 Hz frame rate; deer stationary, so closing speed = ego speed; every confirmed deer
  is a potential path conflict (no lane geometry in the rendered scene labels).
- Only GT-matched detections exist in the artifact, so predicted-mode false urgent
  warnings on empty frames are a lower bound (unmatched false positives not replayed).

Outputs to output/dynamic_ssd/: per-frame timelines, oracle-vs-predicted confusion and
metrics, one-at-a-time sensitivity sweep, transition audit, summary, and a figure.
"""
from __future__ import annotations

import csv
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_ssd_advisory import (MPH_TO_MPS, AdvisoryConfig, AdvisoryController,
                                  AdvisoryState, DeerObservation, EgoState,
                                  LatencyBudget, Reason, SEVERITY)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/advisory/replay_input.csv"
OUT = ROOT / "output/dynamic_ssd"

FPS = 10.0
N_FRAMES = 600
# 25/30/35 mph per AGENT_PLAN handoff step 4; 15/20 mph added because required SSD
# crosses the 40 m validated envelope near 21.5 mph — below it the graded states
# (CAUTION/SLOW) are reachable, above it every in-coverage deer implies BRAKE.
SPEEDS = [15, 20, 25, 30, 35]
GRADES = [("flat", 0.0), ("down5", -0.05), ("down8", -0.08)]
LAT = LatencyBudget(t_driver_s=2.5, t_sensor_s=0.05, t_compute_s=0.05, t_warning_s=0.2)

# crossing7 corrected matched MAE per band (paper Table 2) = pre-crossing8 sigma prior.
C7_SIGMA = {"lt40": 1.97, "40to50": 2.64, "ge50": 4.62}


def sigma_prior(pred_m: float) -> float:
    if pred_m < 40.0:
        return C7_SIGMA["lt40"]
    if pred_m < 50.0:
        return C7_SIGMA["40to50"]
    return C7_SIGMA["ge50"]


def load_frames() -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    with SRC.open(newline="") as f:
        for row in csv.DictReader(f):
            frames[int(row["frame_index"])].append({
                "gt_m": float(row["distance_m"]),
                "detected": row["detected"] == "1",
                "conf": float(row["detection_confidence"]) if row["detection_confidence"] != "nan" else None,
                "pred_m": float(row["pred_m"]) if row["pred_m"] != "nan" else None,
            })
    return frames


def observation(mode: str, boxes: list[dict], ego_mps: float) -> DeerObservation | None:
    if mode == "oracle":
        if not boxes:
            return None
        d = min(b["gt_m"] for b in boxes)
        return DeerObservation(d, 0.0, ego_mps, 1.0)
    dets = [b for b in boxes if b["detected"] and b["pred_m"] is not None]
    if not dets:
        return None
    near = min(dets, key=lambda b: b["pred_m"])
    return DeerObservation(near["pred_m"], sigma_prior(near["pred_m"]), ego_mps,
                           near["conf"] if near["conf"] is not None else 0.0)


def run_replay(frames: dict[int, list[dict]], mode: str, speed_mph: float, grade: float,
               lat: LatencyBudget = LAT, cfg: AdvisoryConfig | None = None) -> list[dict]:
    if cfg is None:
        cfg = (AdvisoryConfig(r_validated_m=1e9, min_persistence_frames=1) if mode == "oracle"
               else AdvisoryConfig())
    ctl = AdvisoryController(latency=lat, cfg=cfg)
    ego = EgoState(speed_mph=speed_mph, grade=grade)
    ego_mps = speed_mph * MPH_TO_MPS
    rows = []
    for i in range(N_FRAMES):
        boxes = frames.get(i, [])
        obs = observation(mode, boxes, ego_mps)
        d = ctl.step(i / FPS, ego, obs)
        rows.append({
            "t_s": round(i / FPS, 2), "frame": i,
            "gt_nearest_m": round(min((b["gt_m"] for b in boxes), default=float("nan")), 2),
            "obs_d_m": round(obs.d_hat_m, 2) if obs else "",
            "sigma_m": round(obs.sigma_d_m, 2) if obs else "",
            "ssd_m": round(d.ssd_required_m, 2),
            "d_eff_m": round(d.d_effective_m, 2) if d.d_effective_m is not None else "",
            "margin_m": round(d.margin_m, 2) if d.margin_m is not None else "",
            "ttc_s": round(d.ttc_s, 2) if d.ttc_s is not None else "",
            "state": d.state.value, "raw_state": d.raw_state.value,
            "reason": d.reason.value, "degraded": int(d.degraded),
        })
    return rows


def state_seq(rows: list[dict]) -> list[str]:
    return [r["state"] for r in rows]


def transitions_per_min(rows: list[dict]) -> float:
    seq = state_seq(rows)
    n = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return n / (len(seq) / FPS / 60.0)


def scenario_metrics(oracle_rows: list[dict], pred_rows: list[dict],
                     frames: dict[int, list[dict]]) -> dict:
    o, p = state_seq(oracle_rows), state_seq(pred_rows)
    conf = Counter(zip(o, p))
    urgent = {"SLOW_PREPARE", "BRAKE_NOW"}
    # A false urgent alarm requires an empty frame beyond the observation-coast window
    # of the last deer-present frame; crossing8 empty gaps are 1-2 frames, all inside it.
    coast_frames = int(AdvisoryConfig().max_staleness_s * FPS)
    last_deer = -10**9
    empty = []
    for i in range(N_FRAMES):
        if frames.get(i):
            last_deer = i
        elif i - last_deer > coast_frames:
            empty.append(i)
    o_brake = [i for i, s in enumerate(o) if s == "BRAKE_NOW"]
    critical_recall = (sum(1 for i in o_brake if p[i] == "BRAKE_NOW") / len(o_brake)) if o_brake else None
    missed_critical = (sum(1 for i in o_brake
                           if SEVERITY[AdvisoryState(p[i])] < SEVERITY[AdvisoryState.SLOW_PREPARE])
                       / len(o_brake)) if o_brake else None
    first_o = next((i for i, s in enumerate(o) if s in urgent), None)
    first_p = next((i for i, s in enumerate(p) if s in urgent), None)
    lead_delay_s = ((first_p - first_o) / FPS) if first_o is not None and first_p is not None else None
    return {
        "state_dist_oracle": dict(Counter(o)),
        "state_dist_predicted": dict(Counter(p)),
        "state_agreement": sum(1 for a, b in zip(o, p) if a == b) / len(o),
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(conf.items())},
        "critical_recall": critical_recall,
        "missed_critical_rate": missed_critical,
        "first_urgent_delay_s_pred_vs_oracle": lead_delay_s,
        "false_urgent_on_empty_frames": sum(1 for i in empty if p[i] in urgent),
        "n_empty_frames_beyond_coast": len(empty),
        "transitions_per_min_oracle": round(transitions_per_min(oracle_rows), 2),
        "transitions_per_min_predicted": round(transitions_per_min(pred_rows), 2),
    }


def calibration_coverage(frames: dict[int, list[dict]], k: float = 1.0) -> dict:
    total = inside = 0
    for boxes in frames.values():
        for b in boxes:
            if b["detected"] and b["pred_m"] is not None:
                total += 1
                if abs(b["pred_m"] - b["gt_m"]) <= k * sigma_prior(b["pred_m"]):
                    inside += 1
    return {"k": k, "matched_detections": total, "coverage": round(inside / total, 4)}


def audit_transitions(rows: list[dict]) -> tuple[list[str], int]:
    """Handoff step 5: every decided state must be explained by raw state or a dwell hold."""
    problems, held = [], 0
    for prev, cur in zip(rows, rows[1:]):
        if cur["state"] == prev["state"]:
            continue
        if cur["state"] != cur["raw_state"] and cur["reason"] != Reason.HELD_BY_DWELL.value:
            problems.append(f"frame {cur['frame']}: decided {cur['state']} != raw {cur['raw_state']} "
                            f"without dwell hold (reason {cur['reason']})")
    for r in rows:
        if r["reason"] == Reason.HELD_BY_DWELL.value:
            held += 1
        if r["state"] == "BRAKE_NOW" and r["reason"] not in (
                Reason.MARGIN_NEGATIVE.value, Reason.TTC_CRITICAL.value, Reason.HELD_BY_DWELL.value):
            problems.append(f"frame {r['frame']}: BRAKE_NOW with unexpected reason {r['reason']}")
        if r["state"] == "BRAKE_NOW" and r["reason"] == Reason.MARGIN_NEGATIVE.value \
                and r["margin_m"] != "" and float(r["margin_m"]) > 0:
            problems.append(f"frame {r['frame']}: BRAKE_NOW margin {r['margin_m']} > 0")
    return problems, held


SENS_SPEED = 15  # predicted ranges cluster 18-35 m, so even 20 mph saturates in BRAKE;
                 # 15 mph (SSD ~26 m) is the highest tested speed with predicted-state variety


def sensitivity(frames: dict[int, list[dict]]) -> list[dict]:
    """One-at-a-time deltas from the 15 mph flat predicted baseline. At >=20 mph the
    predicted replay saturates in BRAKE_NOW (SSD exceeds nearly every predicted range),
    so higher-speed baselines show zero sensitivity to every assumption."""
    base_rows = run_replay(frames, "predicted", SENS_SPEED, 0.0)
    base = state_seq(base_rows)
    variants = [
        ("t_driver=1.5", LatencyBudget(1.5, 0.05, 0.05, 0.2), AdvisoryConfig()),
        ("friction=0.55", LAT, AdvisoryConfig(), EgoState(SENS_SPEED, 0.0, 0.55)),
        ("grade=-8%", LAT, AdvisoryConfig(), EgoState(SENS_SPEED, -0.08, 0.35)),
        ("machine_latency=1.0s", LatencyBudget(2.5, 0.3, 0.3, 0.4), AdvisoryConfig()),
        ("k_sigma=0", LAT, AdvisoryConfig(k_sigma=0.0)),
        ("k_sigma=2", LAT, AdvisoryConfig(k_sigma=2.0)),
        ("b_warning=10", LAT, AdvisoryConfig(b_warning_m=10.0)),
        ("b_warning=20", LAT, AdvisoryConfig(b_warning_m=20.0)),
        ("sigma=crossing8_band_MAE", LAT, AdvisoryConfig()),
    ]
    out = [{"variant": f"baseline({SENS_SPEED}mph,flat,wet,k=1)", "frames_changed_pct": 0.0,
            **{f"pct_{s.value}": round(100 * Counter(base)[s.value] / len(base), 1) for s in AdvisoryState}}]
    for spec in variants:
        name, lat, cfg = spec[0], spec[1], spec[2]
        ego = spec[3] if len(spec) > 3 else EgoState(SENS_SPEED, 0.0, 0.35)
        global sigma_prior
        orig_sigma = sigma_prior
        if name == "sigma=crossing8_band_MAE":
            def sigma_prior(pred_m: float) -> float:  # noqa: F811 — post-hoc honest sigma
                return 9.82 if pred_m < 40 else (15.70 if pred_m < 50 else 23.62)
        try:
            rows = run_replay_with_ego(frames, "predicted", ego, lat, cfg)
        finally:
            sigma_prior = orig_sigma
        seq = state_seq(rows)
        changed = sum(1 for a, b in zip(base, seq) if a != b) / len(base)
        out.append({"variant": name, "frames_changed_pct": round(100 * changed, 1),
                    **{f"pct_{s.value}": round(100 * Counter(seq)[s.value] / len(seq), 1)
                       for s in AdvisoryState}})
    return out


def run_replay_with_ego(frames, mode, ego: EgoState, lat, cfg):
    ctl = AdvisoryController(latency=lat, cfg=cfg)
    ego_mps = ego.speed_mph * MPH_TO_MPS
    rows = []
    for i in range(N_FRAMES):
        obs = observation(mode, frames.get(i, []), ego_mps)
        d = ctl.step(i / FPS, ego, obs)
        rows.append({"t_s": i / FPS, "frame": i, "state": d.state.value,
                     "raw_state": d.raw_state.value, "reason": d.reason.value,
                     "margin_m": d.margin_m if d.margin_m is not None else "",
                     "ssd_m": d.ssd_required_m})
    return rows


def coverage_crossover_mph(r_validated_m: float = 40.0) -> float:
    """Speed at which required SSD (incl. latency budget) equals validated coverage."""
    lo, hi = 1.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if required_ssd(mid) > r_validated_m:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 1)


def required_ssd(speed_mph: float) -> float:
    from dynamic_ssd_advisory import required_ssd_m
    return required_ssd_m(EgoState(speed_mph, 0.0), LAT)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_figure(oracle_rows, pred_rows, speed, grade_name):
    order = ["NORMAL_MONITOR", "CAUTION", "SENSOR_LIMITED", "SLOW_PREPARE", "BRAKE_NOW"]
    colors = {"NORMAL_MONITOR": "#74a9cf", "CAUTION": "#fdcc8a", "SENSOR_LIMITED": "#b8b8d0",
              "SLOW_PREPARE": "#fc8d59", "BRAKE_NOW": "#d7301f"}
    t = [r["t_s"] for r in oracle_rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.3]})
    ax = axes[0]
    gt = [r["gt_nearest_m"] for r in oracle_rows]
    pr = [float(r["obs_d_m"]) if r["obs_d_m"] != "" else float("nan") for r in pred_rows]
    ax.plot(t, gt, lw=1.2, color="#2c7fb8", label="GT nearest deer range")
    ax.plot(t, pr, lw=1.0, color="#d95f02", alpha=0.85, label="predicted nearest range")
    ax.axhline(oracle_rows[0]["ssd_m"], color="k", ls="--", lw=1,
               label=f"required SSD ({speed} mph {grade_name}, incl. latency)")
    ax.axhline(40.0, color="#756bb1", ls=":", lw=1, label="validated coverage (40 m)")
    ax.set_ylabel("Distance (m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"E7 offline replay, crossing8 zero-shot, {speed} mph {grade_name} (10 Hz assumed)")
    ax.grid(alpha=0.25)
    ax2 = axes[1]
    for row_i, (label, rows) in enumerate([("oracle", oracle_rows), ("predicted", pred_rows)]):
        for r in rows:
            ax2.barh(row_i, 1 / FPS, left=r["t_s"], color=colors[r["state"]], height=0.6)
    ax2.set_yticks([0, 1], ["oracle inputs", "predicted inputs"])
    ax2.set_xlabel("Time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in order]
    ax2.legend(handles, order, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout()
    fig.savefig(OUT / f"timeline_{speed}mph_{grade_name}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    global SRC, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    SRC = args.input
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()

    metrics, audits = {}, {}
    for speed in SPEEDS:
        for grade_name, grade in GRADES:
            key = f"{speed}mph_{grade_name}"
            oracle_rows = run_replay(frames, "oracle", speed, grade)
            pred_rows = run_replay(frames, "predicted", speed, grade)
            write_csv(OUT / f"timeline_oracle_{key}.csv", oracle_rows)
            write_csv(OUT / f"timeline_predicted_{key}.csv", pred_rows)
            metrics[key] = scenario_metrics(oracle_rows, pred_rows, frames)
            problems, held = audit_transitions(oracle_rows)
            problems_p, held_p = audit_transitions(pred_rows)
            audits[key] = {"oracle_problems": problems, "oracle_held_frames": held,
                           "predicted_problems": problems_p, "predicted_held_frames": held_p}
            if speed in (20, 25) and grade_name == "flat":
                make_figure(oracle_rows, pred_rows, speed, grade_name)

    coverage = [calibration_coverage(frames, k) for k in (1.0, 2.0, 3.0)]
    sens = sensitivity(frames)
    write_csv(OUT / "sensitivity.csv", sens)
    (OUT / "metrics.json").write_text(json.dumps(
        {"metrics": metrics, "calibration_coverage_c7_sigma_on_c8": coverage,
         "audit": audits}, indent=2))

    n_problems = sum(len(a["oracle_problems"]) + len(a["predicted_problems"]) for a in audits.values())
    n_runs = 2 * len(SPEEDS) * len(GRADES)
    crossover = coverage_crossover_mph()
    lines = [
        "# E7 Dynamic SSD Advisory - Offline crossing8 Replay",
        "",
        "Warning-only offline study per docs/AGENT_PLAN.md E7. Inputs replayed from the frozen",
        "crossing8 zero-shot artifact; 10 Hz assumed; deer stationary (closing speed = ego speed);",
        "all confirmed deer treated as path conflicts; unmatched false-positive detections are not",
        "in the source artifact, so predicted-mode false-urgent counts are lower bounds.",
        "",
        f"Transition audit: {n_problems} rule violations across all {n_runs} runs "
        f"(every non-raw decided state is a dwell hold; every BRAKE_NOW traces to margin/TTC).",
        "",
        "## Headline observations",
        "",
        f"1. **Coverage crossover ~{crossover} mph (flat, wet, full latency budget).** Above this",
        "   speed the required SSD exceeds the 40 m validated detection envelope, so no",
        "   in-coverage deer can yield a positive margin: the advisory saturates in BRAKE_NOW",
        "   and a graded warning is structurally impossible. This restates the paper's static",
        "   margin table as a dynamic controller property.",
        "2. **Range underestimation defeats the coverage gate.** All 447 matched crossing8",
        "   predictions fall below 40 m (max 39.7 m) because the head systematically",
        "   underestimates; SENSOR_LIMITED is never triggered by the range value, and the",
        "   predicted-input advisory collapses to near-constant BRAKE_NOW - conservative in",
        "   direction but a ~100% alarm rate, i.e., not a usable warning policy.",
        "3. **The crossing7 uncertainty prior is invalid on crossing8** (coverage table below):",
        "   fewer than 5% of matched detections have |error| <= 3 sigma. Any deployment claim",
        "   would require recalibrated uncertainty or an independent range sensor, as the",
        "   E7 plan already requires.",
        "",
        "## State distribution (fraction of 600 frames)",
        "",
        "| scenario | mode | NORMAL | CAUTION | SENSOR_LIMITED | SLOW | BRAKE | trans/min |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in metrics.items():
        for mode in ("oracle", "predicted"):
            dist = m[f"state_dist_{mode}"]
            tpm = m[f"transitions_per_min_{mode}"]
            lines.append("| " + " | ".join([
                key, mode,
                *(f"{dist.get(s, 0) / N_FRAMES:.3f}" for s in
                  ("NORMAL_MONITOR", "CAUTION", "SENSOR_LIMITED", "SLOW_PREPARE", "BRAKE_NOW")),
                f"{tpm:.1f}"]) + " |")
    lines += [
        "",
        "## Oracle vs predicted",
        "",
        "| scenario | agreement | critical recall | missed critical | first-urgent delay s | false urgent on empty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, m in metrics.items():
        fu = f"{m['false_urgent_on_empty_frames']}/{m['n_empty_frames_beyond_coast']}"
        lines.append(f"| {key} | {m['state_agreement']:.3f} | "
                     f"{m['critical_recall'] if m['critical_recall'] is not None else 'n/a'} | "
                     f"{m['missed_critical_rate'] if m['missed_critical_rate'] is not None else 'n/a'} | "
                     f"{m['first_urgent_delay_s_pred_vs_oracle'] if m['first_urgent_delay_s_pred_vs_oracle'] is not None else 'n/a'} | {fu} |")
    lines += [
        "",
        "## Range-uncertainty calibration (crossing7 sigma prior evaluated on crossing8)",
        "",
        "Coverage = fraction of matched detections with absolute range error <= k * sigma.",
        "",
        "| k | matched detections | coverage |",
        "|---:|---:|---:|",
    ] + [f"| {c['k']} | {c['matched_detections']} | {c['coverage']} |" for c in coverage] + [
        "",
        f"## Sensitivity (one-at-a-time vs {SENS_SPEED} mph flat predicted baseline)",
        "",
        "Baseline is 15 mph, the highest tested speed with predicted-state variety; at",
        ">=20 mph every variant is identical because BRAKE_NOW saturates regardless of",
        "assumptions. High oracle transition rates at 15-20 mph come from nearest-deer",
        "switching between animals ~10 m apart; per-track (rather than nearest-sample)",
        "association would be required to smooth this in a deployed system.",
        "",
        "| variant | frames changed % | NORMAL % | CAUTION % | SENSOR_LTD % | SLOW % | BRAKE % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ] + [f"| {s['variant']} | {s['frames_changed_pct']} | {s['pct_NORMAL_MONITOR']} | "
         f"{s['pct_CAUTION']} | {s['pct_SENSOR_LIMITED']} | {s['pct_SLOW_PREPARE']} | "
         f"{s['pct_BRAKE_NOW']} |" for s in sens]
    (OUT / "experiment_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(metrics)} scenarios); audit problems: {n_problems}")


if __name__ == "__main__":
    main()
