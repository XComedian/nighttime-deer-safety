#!/usr/bin/env python3
"""E7 dynamic SSD advisory: pure, deterministic state calculation (no UI, no actuation).

Warning-only forward-collision advisory per docs/AGENT_PLAN.md E7. Combines a tracked
deer observation, ego state, latency budget, and conservative buffers into one of five
states: NORMAL_MONITOR, CAUTION, SLOW_PREPARE, BRAKE_NOW, SENSOR_LIMITED.

Design rules (from AGENT_PLAN.md):
- SSD uses the same AASHTO imperial equation as the paper (make_experiment3_safety.ssd_mph)
  with t_total = t_driver + t_sensor + t_compute + t_warning.
- d_effective = d_hat - k*sigma_d - closing_speed*t_machine - b_range. Machine latency is
  charged in BOTH t_total and d_effective, per the plan spec; the double count is an
  intentional conservatism (warnings earlier, never later).
- R_validated (default 40 m, crossing8 detection-only envelope) is separate from d_hat:
  a range estimate beyond validated coverage yields SENSOR_LIMITED, never "safe".
- Escalation is immediate; de-escalation requires minimum dwell time plus a hysteresis
  margin. Track persistence gates CAUTION/SENSOR_LIMITED only — SLOW_PREPARE and
  BRAKE_NOW fire without waiting for persistence.
- Quality-gated observations (stale/low-confidence/high-sigma) that still imply a
  non-positive margin escalate to BRAKE_NOW with the degraded flag set, except
  beyond-coverage ranges, which stay SENSOR_LIMITED because the estimate itself is
  outside demonstrated sensing coverage.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_experiment3_safety import ssd_mph  # single source of truth for the SSD equation

MPH_TO_MPS = 0.44704


class AdvisoryState(Enum):
    NORMAL_MONITOR = "NORMAL_MONITOR"
    CAUTION = "CAUTION"
    SENSOR_LIMITED = "SENSOR_LIMITED"
    SLOW_PREPARE = "SLOW_PREPARE"
    BRAKE_NOW = "BRAKE_NOW"


SEVERITY = {
    AdvisoryState.NORMAL_MONITOR: 0,
    AdvisoryState.CAUTION: 1,
    AdvisoryState.SENSOR_LIMITED: 1,
    AdvisoryState.SLOW_PREPARE: 2,
    AdvisoryState.BRAKE_NOW: 3,
}


class Reason(Enum):
    NO_TRACK = "NO_TRACK"
    NOT_PERSISTENT = "NOT_PERSISTENT"
    NO_CONFLICT = "NO_CONFLICT"
    BEYOND_VALIDATED_RANGE = "BEYOND_VALIDATED_RANGE"
    RANGE_UNCERTAIN = "RANGE_UNCERTAIN"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STALE_INPUT = "STALE_INPUT"
    MARGIN_OK = "MARGIN_OK"
    MARGIN_LOW = "MARGIN_LOW"
    MARGIN_SHRINKING = "MARGIN_SHRINKING"
    MARGIN_NEGATIVE = "MARGIN_NEGATIVE"
    TTC_CRITICAL = "TTC_CRITICAL"
    HELD_BY_DWELL = "HELD_BY_DWELL"


@dataclass(frozen=True)
class EgoState:
    speed_mph: float
    grade: float = 0.0          # signed decimal, downgrade negative
    friction: float = 0.35      # conservative wet assumption


@dataclass(frozen=True)
class DeerObservation:
    d_hat_m: float              # range estimate to nearest conflicting deer
    sigma_d_m: float            # range uncertainty (1-sigma or MAE proxy)
    closing_speed_mps: float    # positive when the gap is closing
    confidence: float           # detector confidence in [0,1]
    in_conflict: bool = True    # no lane geometry in replay: all deer conservative-conflict
    age_s: float = 0.0          # time since the observation was produced


@dataclass(frozen=True)
class LatencyBudget:
    t_driver_s: float = 2.5
    t_sensor_s: float = 0.05
    t_compute_s: float = 0.05
    t_warning_s: float = 0.2

    @property
    def machine_s(self) -> float:
        return self.t_sensor_s + self.t_compute_s + self.t_warning_s

    @property
    def total_s(self) -> float:
        return self.t_driver_s + self.machine_s


@dataclass(frozen=True)
class AdvisoryConfig:
    k_sigma: float = 1.0
    b_range_m: float = 0.0            # extra range calibration buffer
    b_warning_m: float = 15.0         # early-warning margin buffer
    r_validated_m: float = 40.0       # crossing8 detection-only envelope
    min_confidence: float = 0.25
    max_sigma_m: float = 8.0
    max_staleness_s: float = 0.4
    min_persistence_frames: int = 3
    min_dwell_s: float = 1.0
    hysteresis_m: float = 2.0
    projection_horizon_s: float = 1.0  # SLOW if margin projected <= 0 within this horizon
    ttc_critical_s: float | None = None  # default: latency total


@dataclass(frozen=True)
class Decision:
    state: AdvisoryState
    reason: Reason
    degraded: bool
    ssd_required_m: float
    d_effective_m: float | None
    margin_m: float | None
    ttc_s: float | None
    raw_state: AdvisoryState
    persistence: int


def required_ssd_m(ego: EgoState, latency: LatencyBudget) -> float:
    return ssd_mph(ego.speed_mph, ego.grade, reaction_s=latency.total_s, friction=ego.friction)


def effective_distance_m(obs: DeerObservation, latency: LatencyBudget, cfg: AdvisoryConfig) -> float:
    return (obs.d_hat_m
            - cfg.k_sigma * obs.sigma_d_m
            - max(0.0, obs.closing_speed_mps) * latency.machine_s
            - cfg.b_range_m)


def _quality_reason(obs: DeerObservation, cfg: AdvisoryConfig) -> Reason | None:
    if obs.age_s > cfg.max_staleness_s:
        return Reason.STALE_INPUT
    if obs.d_hat_m > cfg.r_validated_m:
        return Reason.BEYOND_VALIDATED_RANGE
    if obs.sigma_d_m > cfg.max_sigma_m:
        return Reason.RANGE_UNCERTAIN
    if obs.confidence < cfg.min_confidence:
        return Reason.LOW_CONFIDENCE
    return None


def raw_decision(ego: EgoState,
                 obs: DeerObservation | None,
                 latency: LatencyBudget,
                 cfg: AdvisoryConfig,
                 persistence: int,
                 margin_rate_mps: float | None) -> Decision:
    """Stateless single-frame decision before hysteresis/dwell arbitration."""
    ssd = required_ssd_m(ego, latency)

    if obs is None:
        return Decision(AdvisoryState.NORMAL_MONITOR, Reason.NO_TRACK, False,
                        ssd, None, None, None, AdvisoryState.NORMAL_MONITOR, persistence)

    d_eff = effective_distance_m(obs, latency, cfg)
    margin = d_eff - ssd
    closing = max(0.0, obs.closing_speed_mps)
    ttc = (d_eff / closing) if closing > 1e-6 and d_eff > 0 else (0.0 if d_eff <= 0 and closing > 1e-6 else None)
    ttc_critical = cfg.ttc_critical_s if cfg.ttc_critical_s is not None else latency.total_s

    quality = _quality_reason(obs, cfg)
    critical = margin <= 0.0 or (ttc is not None and ttc <= ttc_critical)

    if quality is not None:
        # Degraded input: conservative caution; escalate only when even the degraded
        # estimate is critical AND the estimate lies inside validated coverage.
        if critical and quality != Reason.BEYOND_VALIDATED_RANGE:
            reason = Reason.TTC_CRITICAL if (ttc is not None and ttc <= ttc_critical and margin > 0) else Reason.MARGIN_NEGATIVE
            return Decision(AdvisoryState.BRAKE_NOW, reason, True,
                            ssd, d_eff, margin, ttc, AdvisoryState.BRAKE_NOW, persistence)
        return Decision(AdvisoryState.SENSOR_LIMITED, quality, True,
                        ssd, d_eff, margin, ttc, AdvisoryState.SENSOR_LIMITED, persistence)

    if not obs.in_conflict:
        return Decision(AdvisoryState.NORMAL_MONITOR, Reason.NO_CONFLICT, False,
                        ssd, d_eff, margin, ttc, AdvisoryState.NORMAL_MONITOR, persistence)

    if critical:
        reason = Reason.TTC_CRITICAL if (ttc is not None and ttc <= ttc_critical and margin > 0) else Reason.MARGIN_NEGATIVE
        return Decision(AdvisoryState.BRAKE_NOW, reason, False,
                        ssd, d_eff, margin, ttc, AdvisoryState.BRAKE_NOW, persistence)

    projected = margin + (margin_rate_mps or 0.0) * cfg.projection_horizon_s
    if margin <= cfg.b_warning_m:
        return Decision(AdvisoryState.SLOW_PREPARE, Reason.MARGIN_LOW, False,
                        ssd, d_eff, margin, ttc, AdvisoryState.SLOW_PREPARE, persistence)
    if projected <= 0.0:
        return Decision(AdvisoryState.SLOW_PREPARE, Reason.MARGIN_SHRINKING, False,
                        ssd, d_eff, margin, ttc, AdvisoryState.SLOW_PREPARE, persistence)

    if persistence < cfg.min_persistence_frames:
        return Decision(AdvisoryState.NORMAL_MONITOR, Reason.NOT_PERSISTENT, False,
                        ssd, d_eff, margin, ttc, AdvisoryState.CAUTION, persistence)
    return Decision(AdvisoryState.CAUTION, Reason.MARGIN_OK, False,
                    ssd, d_eff, margin, ttc, AdvisoryState.CAUTION, persistence)


@dataclass
class AdvisoryController:
    """Stateful wrapper: track persistence, observation coasting, dwell, hysteresis."""
    latency: LatencyBudget = field(default_factory=LatencyBudget)
    cfg: AdvisoryConfig = field(default_factory=AdvisoryConfig)

    _state: AdvisoryState = AdvisoryState.NORMAL_MONITOR
    _state_since_s: float = 0.0
    _persistence: int = 0
    _last_obs: DeerObservation | None = None
    _last_obs_time_s: float | None = None
    _last_margin: float | None = None
    _last_time_s: float | None = None

    def step(self, now_s: float, ego: EgoState, obs: DeerObservation | None) -> Decision:
        # Coast the last observation through short dropouts so single-frame detector
        # misses do not immediately drop the track (staleness gate bounds the coast).
        if obs is not None:
            self._persistence += 1
            self._last_obs, self._last_obs_time_s = obs, now_s
        elif self._last_obs is not None and self._last_obs_time_s is not None:
            aged = now_s - self._last_obs_time_s
            if aged <= self.cfg.max_staleness_s:
                obs = DeerObservation(self._last_obs.d_hat_m, self._last_obs.sigma_d_m,
                                      self._last_obs.closing_speed_mps, self._last_obs.confidence,
                                      self._last_obs.in_conflict, age_s=aged)
            else:
                self._persistence = 0
                self._last_obs = None
        else:
            self._persistence = 0

        margin_rate = None
        raw = raw_decision(ego, obs, self.latency, self.cfg, self._persistence, None)
        if raw.margin_m is not None and self._last_margin is not None and self._last_time_s is not None:
            dt = now_s - self._last_time_s
            if dt > 1e-6:
                margin_rate = (raw.margin_m - self._last_margin) / dt
                raw = raw_decision(ego, obs, self.latency, self.cfg, self._persistence, margin_rate)
        self._last_margin, self._last_time_s = raw.margin_m, now_s

        decided = self._arbitrate(now_s, raw)
        return decided

    def _arbitrate(self, now_s: float, raw: Decision) -> Decision:
        cur, new = self._state, raw.state
        if SEVERITY[new] > SEVERITY[cur] or new == cur:
            pass  # escalation (or no change) is immediate
        else:
            dwell = now_s - self._state_since_s
            if dwell < self.cfg.min_dwell_s:
                return self._hold(raw, cur)
            if cur == AdvisoryState.SLOW_PREPARE and new in (AdvisoryState.CAUTION, AdvisoryState.NORMAL_MONITOR):
                if raw.margin_m is not None and raw.margin_m <= self.cfg.b_warning_m + self.cfg.hysteresis_m:
                    return self._hold(raw, cur)
            if cur == AdvisoryState.BRAKE_NOW and raw.margin_m is not None \
                    and raw.margin_m <= self.cfg.hysteresis_m:
                return self._hold(raw, cur)
        if new != cur:
            self._state, self._state_since_s = new, now_s
        return raw

    def _hold(self, raw: Decision, held_state: AdvisoryState) -> Decision:
        return Decision(held_state, Reason.HELD_BY_DWELL, raw.degraded,
                        raw.ssd_required_m, raw.d_effective_m, raw.margin_m, raw.ttc_s,
                        raw.raw_state, raw.persistence)
