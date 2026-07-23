# crossing8 Affine Recalibration Probe

Calibration-vs-representation test for the frozen crossing7 range head
evaluated on the current COCO-derived crossing8 instance labels
(alternative explanation tested during manuscript development).
Model checkpoint frozen; only the scalar range output is remapped by a fit
estimated from the first K frames (10 Hz assumed; reference distances during
calibration would require an independent range source). Evaluation is the
untouched tail, frames 300-599. Detection is unchanged; joint success still
counts missed deer as failures.

| variant | cal frames | cal pts | fit | eval MAE m | bias m | %<=5m (matched) | joint 30-40 | joint 40-50 | joint 50-60 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| uncorrected | 0 | 0 | identity | 15.03 | -15.03 | 0.023 | 0.029 | 0.000 | 0.000 |
| bias_only_K10 | 10 | 6 | gt = pred +17.67 | 5.98 | +2.63 | 0.414 | 0.107 | 0.323 | 0.183 |
| affine_K10 | 10 | 6 | gt = 2.501*pred -25.53 | 6.61 | +1.16 | 0.398 | 0.223 | 0.208 | 0.127 |
| log_affine_K10 | 10 | 6 | ln gt = 1.595*ln pred -1.528 | 6.58 | +1.27 | 0.421 | 0.262 | 0.198 | 0.127 |
| bias_only_K30 | 30 | 13 | gt = pred +16.52 | 5.63 | +1.48 | 0.436 | 0.146 | 0.323 | 0.169 |
| affine_K30 | 30 | 13 | gt = 2.040*pred -13.52 | 5.70 | +0.38 | 0.474 | 0.282 | 0.240 | 0.141 |
| log_affine_K30 | 30 | 13 | ln gt = 1.364*ln pred -0.781 | 5.76 | +0.17 | 0.466 | 0.291 | 0.229 | 0.127 |
| bias_only_K60 | 60 | 24 | gt = pred +15.13 | 5.39 | +0.09 | 0.511 | 0.233 | 0.354 | 0.141 |
| affine_K60 | 60 | 24 | gt = 1.793*pred -7.22 | 5.36 | -0.20 | 0.526 | 0.282 | 0.281 | 0.183 |
| log_affine_K60 | 60 | 24 | ln gt = 1.179*ln pred -0.175 | 5.36 | -0.42 | 0.541 | 0.291 | 0.292 | 0.183 |
| bias_only_K120 | 120 | 44 | gt = pred +14.62 | 5.35 | -0.41 | 0.519 | 0.252 | 0.375 | 0.099 |
| affine_K120 | 120 | 44 | gt = 1.492*pred +0.68 | 5.14 | -0.68 | 0.594 | 0.320 | 0.344 | 0.169 |
| log_affine_K120 | 120 | 44 | ln gt = 0.991*ln pred +0.439 | 5.13 | -0.98 | 0.579 | 0.311 | 0.344 | 0.155 |
| bias_only_K300 | 300 | 108 | gt = pred +14.41 | 5.34 | -0.62 | 0.556 | 0.282 | 0.396 | 0.099 |
| affine_K300 | 300 | 108 | gt = 1.353*pred +4.71 | 5.14 | -0.52 | 0.579 | 0.311 | 0.354 | 0.155 |
| log_affine_K300 | 300 | 108 | ln gt = 0.885*ln pred +0.795 | 5.14 | -0.85 | 0.579 | 0.320 | 0.354 | 0.141 |
| ORACLE_affine_all241 | 600 | 241 | gt = 1.358*pred +4.85 | 5.16 | -0.23 | 0.586 | 0.320 | 0.344 | 0.169 |
| ORACLE_log_affine_all241 | 600 | 241 | ln gt = 0.891*ln pred +0.781 | 5.14 | -0.63 | 0.571 | 0.311 | 0.354 | 0.141 |

Eval-tail band sizes (GT boxes): 30-40: 103 (recall 0.515), 40-50: 96 (recall 0.490), 50-60: 71 (recall 0.451)

Reading guide: if calibrated MAE approaches the crossing7 in-sequence level
(2-5 m), the failure was scene calibration; if it stays well above the 5 m
tolerance, or the oracle ceiling itself fails, the failure is representational
and no output remapping can rescue the 5 m screen.

## Interpretation (numbers above; inputs are frozen artifacts, rerun is deterministic)

1. **A scene-dependent calibration shift explains most of the signed error.**
   The uncorrected evaluation-tail bias and MAE are both 15.03 m. A bias-only
   correction fitted to the first 60 frames (24 matched instances) reduces
   bias to +0.09 m and MAE to 5.39 m without changing the checkpoint.
2. **A global output remapping does not recover the joint screen.** The 60-frame
   affine fit yields 5.36 m MAE and joint success of 0.282/0.281/0.183.
   Even the optimistic affine fit using all 241 matched instances, including
   evaluation instances, leaves 5.16 m MAE and joint success of
   0.320/0.344/0.169. Every band remains below the 0.60 project threshold.
3. **The result supports a mixed explanation.** A scalar calibration shift
   accounts for the nearly uniform negative bias, but residual range scatter
   and missed detections remain. The zero-shot failure therefore should not be
   described as purely representational or purely calibrational.
4. **Any operational calibration would need an independent metric reference.**
   This post hoc probe uses rendered ground truth and makes crossing8 a
   development scene; a newly prespecified model or correction requires a
   fresh scene-disjoint evaluation.
