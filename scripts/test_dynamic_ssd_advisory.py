#!/usr/bin/env python3
"""Unit tests for the E7 dynamic SSD advisory controller (oracle logic, no perception)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_ssd_advisory import (MPH_TO_MPS, AdvisoryConfig, AdvisoryController,
                                  AdvisoryState, DeerObservation, EgoState,
                                  LatencyBudget, Reason, effective_distance_m,
                                  raw_decision, required_ssd_m)
from make_experiment3_safety import ssd_mph

LAT = LatencyBudget(t_driver_s=2.5, t_sensor_s=0.05, t_compute_s=0.05, t_warning_s=0.2)
CFG = AdvisoryConfig()


def obs(d, sigma=0.0, closing=30 * MPH_TO_MPS, conf=0.9, conflict=True, age=0.0):
    return DeerObservation(d, sigma, closing, conf, conflict, age)


def ego(v=30.0, g=0.0):
    return EgoState(speed_mph=v, grade=g)


# --- SSD equation: signs and cross-module consistency ---

def test_ssd_matches_paper_equation():
    assert required_ssd_m(EgoState(30, 0.0), LatencyBudget(2.5, 0, 0, 0)) == pytest.approx(
        ssd_mph(30, 0.0, reaction_s=2.5), abs=1e-9)


def test_downgrade_increases_ssd_upgrade_decreases():
    flat = required_ssd_m(ego(30, 0.0), LAT)
    down = required_ssd_m(ego(30, -0.05), LAT)
    up = required_ssd_m(ego(30, +0.05), LAT)
    assert down > flat > up


def test_zero_speed_zero_ssd_and_no_brake():
    e = EgoState(0.0, 0.0)
    assert required_ssd_m(e, LAT) == pytest.approx(0.0)
    d = raw_decision(e, obs(20, closing=0.0), LAT, CFG, persistence=5, margin_rate_mps=None)
    assert d.state == AdvisoryState.CAUTION


# --- effective distance conservatism ---

def test_effective_distance_subtracts_uncertainty_latency_buffer():
    cfg = AdvisoryConfig(k_sigma=2.0, b_range_m=3.0)
    o = obs(40.0, sigma=2.0, closing=10.0)
    assert effective_distance_m(o, LAT, cfg) == pytest.approx(40 - 4.0 - 10 * LAT.machine_s - 3.0)


# --- state selection ---

def test_negative_margin_brakes():
    d = raw_decision(ego(30), obs(20), LAT, CFG, 5, None)  # SSD(30 mph) >> 20 m
    assert d.state == AdvisoryState.BRAKE_NOW and d.reason == Reason.MARGIN_NEGATIVE


def test_small_positive_margin_slows():
    e = ego(25)
    ssd = required_ssd_m(e, LAT)
    d = raw_decision(e, obs(ssd + 5.0, sigma=0.0, closing=0.0), LAT,
                     AdvisoryConfig(r_validated_m=1e9), 5, None)
    assert d.state == AdvisoryState.SLOW_PREPARE and d.reason == Reason.MARGIN_LOW


def test_large_margin_cautions_when_persistent():
    cfg = AdvisoryConfig(r_validated_m=1e9)
    e = ego(25)
    far = required_ssd_m(e, LAT) + cfg.b_warning_m + 10
    ok = raw_decision(e, obs(far, closing=0.0), LAT, cfg, persistence=3, margin_rate_mps=None)
    new = raw_decision(e, obs(far, closing=0.0), LAT, cfg, persistence=1, margin_rate_mps=None)
    assert ok.state == AdvisoryState.CAUTION
    assert new.state == AdvisoryState.NORMAL_MONITOR and new.reason == Reason.NOT_PERSISTENT


def test_persistence_never_delays_critical_warning():
    d = raw_decision(ego(30), obs(15), LAT, CFG, persistence=1, margin_rate_mps=None)
    assert d.state == AdvisoryState.BRAKE_NOW


def test_no_conflict_is_normal_but_never_labeled_safe_beyond_coverage():
    d = raw_decision(ego(25), obs(35, conflict=False, closing=0.0), LAT, CFG, 5, None)
    assert d.state == AdvisoryState.NORMAL_MONITOR and d.reason == Reason.NO_CONFLICT
    far = raw_decision(ego(25), obs(55, conflict=False, closing=0.0), LAT, CFG, 5, None)
    assert far.state == AdvisoryState.SENSOR_LIMITED  # coverage gate precedes conflict


# --- SENSOR_LIMITED gates ---

def test_stale_input_sensor_limited():
    d = raw_decision(ego(25), obs(80, age=1.0, closing=0.0), LAT,
                     AdvisoryConfig(r_validated_m=1e9), 5, None)
    assert d.state == AdvisoryState.SENSOR_LIMITED and d.reason == Reason.STALE_INPUT and d.degraded


def test_beyond_validated_range_sensor_limited_even_if_margin_negative():
    d = raw_decision(ego(35), obs(55), LAT, CFG, 5, None)  # SSD(35) ~ 75 m > 55 m
    assert d.state == AdvisoryState.SENSOR_LIMITED
    assert d.reason == Reason.BEYOND_VALIDATED_RANGE


def test_high_sigma_sensor_limited_but_critical_degraded_brake():
    cfg = AdvisoryConfig(max_sigma_m=8.0)
    safe = raw_decision(ego(25), obs(80, sigma=12.0, closing=0.0), LAT,
                        AdvisoryConfig(max_sigma_m=8.0, r_validated_m=1e9), 5, None)
    assert safe.state == AdvisoryState.SENSOR_LIMITED and safe.reason == Reason.RANGE_UNCERTAIN
    crit = raw_decision(ego(30), obs(20, sigma=12.0), LAT, cfg, 5, None)
    assert crit.state == AdvisoryState.BRAKE_NOW and crit.degraded


def test_low_confidence_sensor_limited():
    d = raw_decision(ego(25), obs(80, conf=0.1, closing=0.0), LAT,
                     AdvisoryConfig(r_validated_m=1e9), 5, None)
    assert d.state == AdvisoryState.SENSOR_LIMITED and d.reason == Reason.LOW_CONFIDENCE


# --- monotonic conservatism: worse conditions never lower severity ---

@pytest.mark.parametrize("base,worse", [
    (dict(v=25, g=0.0), dict(v=30, g=0.0)),        # faster
    (dict(v=25, g=0.0), dict(v=25, g=-0.08)),      # downgrade
])
def test_speed_and_downgrade_move_warnings_earlier(base, worse):
    cfg = AdvisoryConfig(r_validated_m=1e9)
    for d_m in [30, 45, 55, 70, 90]:
        o = obs(d_m, closing=0.0)
        s0 = raw_decision(ego(**base), o, LAT, cfg, 5, None)
        s1 = raw_decision(ego(**worse), o, LAT, cfg, 5, None)
        from dynamic_ssd_advisory import SEVERITY
        assert SEVERITY[s1.state] >= SEVERITY[s0.state]


def test_more_uncertainty_and_latency_never_later():
    from dynamic_ssd_advisory import SEVERITY
    cfg0 = AdvisoryConfig(k_sigma=1.0, r_validated_m=1e9)
    cfg1 = AdvisoryConfig(k_sigma=2.0, r_validated_m=1e9)
    lat1 = LatencyBudget(t_driver_s=2.5, t_sensor_s=0.2, t_compute_s=0.2, t_warning_s=0.6)
    for d_m in [40, 55, 70, 90]:
        o = obs(d_m, sigma=4.0)
        s0 = raw_decision(ego(25), o, LAT, cfg0, 5, None)
        s_sig = raw_decision(ego(25), o, LAT, cfg1, 5, None)
        s_lat = raw_decision(ego(25), o, lat1, cfg0, 5, None)
        assert SEVERITY[s_sig.state] >= SEVERITY[s0.state]
        assert SEVERITY[s_lat.state] >= SEVERITY[s0.state]


# --- controller: hysteresis, dwell, flicker, immediate escalation ---

def test_escalation_immediate_deescalation_dwells():
    c = AdvisoryController(latency=LAT, cfg=AdvisoryConfig(min_dwell_s=1.0, r_validated_m=1e9,
                                                           min_persistence_frames=1))
    e = ego(25)
    ssd = required_ssd_m(e, LAT)
    t = 0.0
    d = c.step(t, e, obs(ssd - 5, closing=0.0))          # immediate BRAKE from NORMAL
    assert d.state == AdvisoryState.BRAKE_NOW
    t += 0.1
    d = c.step(t, e, obs(ssd + 40, closing=0.0))          # margin recovered but dwell holds
    assert d.state == AdvisoryState.BRAKE_NOW and d.reason == Reason.HELD_BY_DWELL
    t += 1.1
    d = c.step(t, e, obs(ssd + 40, closing=0.0))          # after dwell, de-escalates
    assert d.state != AdvisoryState.BRAKE_NOW


def test_no_flicker_on_oscillating_margin():
    cfg = AdvisoryConfig(min_dwell_s=1.0, hysteresis_m=2.0, r_validated_m=1e9,
                         min_persistence_frames=1)
    c = AdvisoryController(latency=LAT, cfg=cfg)
    e = ego(25)
    ssd = required_ssd_m(e, LAT)
    states = []
    for i in range(40):                                    # 4 s at 10 Hz
        wobble = 1.0 if i % 2 else -1.0                    # margin oscillates +-1 m around b_warning
        d = c.step(i * 0.1, e, obs(ssd + cfg.b_warning_m + wobble, closing=0.0))
        states.append(d.state)
    transitions = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert transitions <= 2                                # settles instead of flickering


def test_track_coasts_through_single_frame_dropout():
    cfg = AdvisoryConfig(max_staleness_s=0.4, min_persistence_frames=1, r_validated_m=1e9)
    c = AdvisoryController(latency=LAT, cfg=cfg)
    e = ego(30)
    d0 = c.step(0.0, e, obs(20))
    d1 = c.step(0.1, e, None)                              # dropout: coasted, still braking
    assert d0.state == d1.state == AdvisoryState.BRAKE_NOW
    d_lost = c.step(1.0, e, None)                          # beyond staleness: dwell may hold,
    assert d_lost.raw_state == AdvisoryState.NORMAL_MONITOR  # but raw track is gone


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
