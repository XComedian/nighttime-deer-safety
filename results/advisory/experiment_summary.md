# E7 Dynamic SSD Advisory - Offline crossing8 Replay

Warning-only offline study per docs/AGENT_PLAN.md E7. Inputs replayed from the frozen
crossing8 zero-shot artifact; 10 Hz assumed; deer stationary (closing speed = ego speed);
all confirmed deer treated as path conflicts; unmatched false-positive detections are not
in the source artifact, so predicted-mode false-urgent counts are lower bounds.

Transition audit: 0 rule violations across all 30 runs (every non-raw decided state is a dwell hold; every BRAKE_NOW traces to margin/TTC).

## Headline observations

1. **Coverage crossover ~21.3 mph (flat, wet, full latency budget).** Above this
   speed the required SSD exceeds the 40 m validated detection envelope, so no
   in-coverage deer can yield a positive margin: the advisory saturates in BRAKE_NOW
   and a graded warning is structurally impossible. This restates the paper's static
   margin table as a dynamic controller property.
2. **Range underestimation defeats the coverage gate.** All 447 matched crossing8
   predictions fall below 40 m (max 39.7 m) because the head systematically
   underestimates; SENSOR_LIMITED is never triggered by the range value, and the
   predicted-input advisory collapses to near-constant BRAKE_NOW - conservative in
   direction but a ~100% alarm rate, i.e., not a usable warning policy.
3. **The crossing7 uncertainty prior is invalid on crossing8** (coverage table below):
   fewer than 5% of matched detections have |error| <= 3 sigma. Any deployment claim
   would require recalibrated uncertainty or an independent range sensor, as the
   E7 plan already requires.

## State distribution (fraction of 600 frames)

| scenario | mode | NORMAL | CAUTION | SENSOR_LIMITED | SLOW | BRAKE | trans/min |
|---|---|---:|---:|---:|---:|---:|---:|
| 15mph_flat | oracle | 0.000 | 0.110 | 0.000 | 0.890 | 0.000 | 95.0 |
| 15mph_flat | predicted | 0.023 | 0.000 | 0.000 | 0.095 | 0.882 | 55.0 |
| 15mph_down5 | oracle | 0.000 | 0.108 | 0.000 | 0.892 | 0.000 | 95.0 |
| 15mph_down5 | predicted | 0.023 | 0.000 | 0.000 | 0.062 | 0.915 | 51.0 |
| 15mph_down8 | oracle | 0.000 | 0.112 | 0.000 | 0.888 | 0.000 | 93.0 |
| 15mph_down8 | predicted | 0.023 | 0.000 | 0.000 | 0.047 | 0.930 | 41.0 |
| 20mph_flat | oracle | 0.000 | 0.023 | 0.000 | 0.197 | 0.780 | 93.0 |
| 20mph_flat | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 20mph_down5 | oracle | 0.000 | 0.018 | 0.000 | 0.175 | 0.807 | 90.0 |
| 20mph_down5 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 20mph_down8 | oracle | 0.000 | 0.010 | 0.000 | 0.167 | 0.823 | 91.0 |
| 20mph_down8 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 25mph_flat | oracle | 0.000 | 0.000 | 0.000 | 0.060 | 0.940 | 54.0 |
| 25mph_flat | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 25mph_down5 | oracle | 0.000 | 0.000 | 0.000 | 0.042 | 0.958 | 36.0 |
| 25mph_down5 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 25mph_down8 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 25mph_down8 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 30mph_flat | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 30mph_flat | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 30mph_down5 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 30mph_down5 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 30mph_down8 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 30mph_down8 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 35mph_flat | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 35mph_flat | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 35mph_down5 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 35mph_down5 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |
| 35mph_down8 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 35mph_down8 | predicted | 0.027 | 0.000 | 0.000 | 0.000 | 0.973 | 13.0 |

## Oracle vs predicted

| scenario | agreement | critical recall | missed critical | first-urgent delay s | false urgent on empty |
|---|---:|---:|---:|---:|---:|
| 15mph_flat | 0.085 | n/a | n/a | 0.0 | 0/0 |
| 15mph_down5 | 0.058 | n/a | n/a | 0.0 | 0/0 |
| 15mph_down8 | 0.045 | n/a | n/a | 0.0 | 0/0 |
| 20mph_flat | 0.768 | 0.9850427350427351 | 0.014957264957264958 | 0.1 | 0/0 |
| 20mph_down5 | 0.793 | 0.9834710743801653 | 0.01652892561983471 | 0.1 | 0/0 |
| 20mph_down8 | 0.810 | 0.9838056680161943 | 0.016194331983805668 | 0.1 | 0/0 |
| 25mph_flat | 0.915 | 0.973404255319149 | 0.026595744680851064 | 0.1 | 0/0 |
| 25mph_down5 | 0.937 | 0.9773913043478261 | 0.022608695652173914 | 0.1 | 0/0 |
| 25mph_down8 | 0.973 | 0.9733333333333334 | 0.02666666666666667 | 0.1 | 0/0 |
| 30mph_flat | 0.973 | 0.9733333333333334 | 0.02666666666666667 | 0.1 | 0/0 |
| 30mph_down5 | 0.973 | 0.9733333333333334 | 0.02666666666666667 | 0.1 | 0/0 |
| 30mph_down8 | 0.973 | 0.9733333333333334 | 0.02666666666666667 | 0.1 | 0/0 |
| 35mph_flat | 0.973 | 0.9733333333333334 | 0.02666666666666667 | 0.1 | 0/0 |
| 35mph_down5 | 0.973 | 0.9733333333333334 | 0.02666666666666667 | 0.1 | 0/0 |
| 35mph_down8 | 0.973 | 0.9733333333333334 | 0.02666666666666667 | 0.1 | 0/0 |

## Range-uncertainty calibration (crossing7 sigma prior evaluated on crossing8)

Coverage = fraction of matched detections with absolute range error <= k * sigma.

| k | matched detections | coverage |
|---:|---:|---:|
| 1.0 | 447 | 0.0045 |
| 2.0 | 447 | 0.0134 |
| 3.0 | 447 | 0.0403 |

## Sensitivity (one-at-a-time vs 15 mph flat predicted baseline)

Baseline is 15 mph, the highest tested speed with predicted-state variety; at
>=20 mph every variant is identical because BRAKE_NOW saturates regardless of
assumptions. High oracle transition rates at 15-20 mph come from nearest-deer
switching between animals ~10 m apart; per-track (rather than nearest-sample)
association would be required to smooth this in a deployed system.

| variant | frames changed % | NORMAL % | CAUTION % | SENSOR_LTD % | SLOW % | BRAKE % |
|---|---:|---:|---:|---:|---:|---:|
| baseline(15mph,flat,wet,k=1) | 0.0 | 2.3 | 0.0 | 0.0 | 9.5 | 88.2 |
| t_driver=1.5 | 42.7 | 1.5 | 0.0 | 0.0 | 44.2 | 54.3 |
| friction=0.55 | 17.7 | 1.3 | 0.0 | 0.0 | 19.0 | 79.7 |
| grade=-8% | 5.5 | 2.3 | 0.0 | 0.0 | 4.7 | 93.0 |
| machine_latency=1.0s | 9.7 | 2.7 | 0.0 | 0.0 | 0.0 | 97.3 |
| k_sigma=0 | 15.7 | 1.3 | 0.0 | 0.0 | 19.7 | 79.0 |
| k_sigma=2 | 5.5 | 2.3 | 0.0 | 0.0 | 4.7 | 93.0 |
| b_warning=10 | 0.0 | 2.3 | 0.0 | 0.0 | 9.5 | 88.2 |
| b_warning=20 | 0.0 | 2.3 | 0.0 | 0.0 | 9.5 | 88.2 |
| sigma=crossing8_band_MAE | 9.7 | 2.7 | 0.0 | 0.0 | 0.0 | 97.3 |
