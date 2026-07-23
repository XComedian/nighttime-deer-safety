#!/usr/bin/env python3
"""Reaction-time scenario analysis: SSD and envelope margins vs driver reaction time.

Motivation: AASHTO's 2.5 s perception-reaction time is a conservative design value for
unalerted drivers. A condition-aware system that already knows visibility is degraded
(nighttime, fog, deer-activity hours) and pre-alerts the driver may shorten the
effective reaction time. This table shows what the demonstrated 40 m detection-only
envelope and a target 60 m envelope would support at t = 1.0-2.5 s. Scenario analysis
only: no reduced reaction time is validated here.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_experiment3_safety import ssd_mph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "reaction_ssd"

REACTIONS = [2.5, 2.0, 1.5, 1.0]
SPEEDS = [20, 25, 30, 35]
GRADES = [("flat", 0.0), ("-5%", -0.05), ("-8%", -0.08)]
ENVELOPES = [40.0, 60.0]


def max_speed_within(envelope_m: float, t: float, grade: float) -> float:
    lo, hi = 1.0, 80.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ssd_mph(mid, grade, reaction_s=t) > envelope_m:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in REACTIONS:
        for speed in SPEEDS:
            for gname, g in GRADES:
                ssd = ssd_mph(speed, g, reaction_s=t)
                rows.append({
                    "t_reaction_s": t, "speed_mph": speed, "grade": gname,
                    "ssd_m": round(ssd, 1),
                    "margin_40m": round(40.0 - ssd, 1),
                    "margin_60m": round(60.0 - ssd, 1),
                })
    with (OUT / "reaction_ssd_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# SSD vs Driver Reaction Time (scenario analysis)",
        "",
        "AASHTO 2.5 s is the conservative unalerted design value; shorter times model a",
        "driver pre-alerted by a condition-aware system. Margins are envelope minus SSD;",
        "positive means a deer detected at the envelope edge leaves stopping room.",
        "Machine latency is excluded here (consistent with the paper's Table 6) and",
        "would shift margins negative by roughly speed x latency.",
        "",
        "| t (s) | speed (mph) | grade | SSD (m) | margin @40 m | margin @60 m |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['t_reaction_s']} | {r['speed_mph']} | {r['grade']} | "
                     f"{r['ssd_m']} | {r['margin_40m']} | {r['margin_60m']} |")
    lines += [
        "",
        "## Maximum speed with non-negative margin (flat / -5% / -8%)",
        "",
        "| t (s) | 40 m envelope | 60 m envelope |",
        "|---:|---|---|",
    ]
    for t in REACTIONS:
        v40 = " / ".join(f"{max_speed_within(40, t, g):.0f}" for _, g in GRADES)
        v60 = " / ".join(f"{max_speed_within(60, t, g):.0f}" for _, g in GRADES)
        lines.append(f"| {t} | {v40} mph | {v60} mph |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
