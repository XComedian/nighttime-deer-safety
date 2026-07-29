# Corrected 60 m Controller Replay

The final crossing8 LOSO checkpoint is replayed with a 60 m coverage gate. Predicted mode uses point estimates (sigma=0) because no independently calibrated uncertainty model exists. The artifact contains matched detections only, so nuisance alarms caused by unmatched false positives are not measured.

Instances: 536; matched detections: 379; matched MAE 3.65 m; bias 1.87 m.
Flat-road coverage crossover including the 0.30 s machine latency budget: 28.7 mph.
Transition audit problems: 0.

| scenario | mode | normal | caution | sensor limited | slow | brake | transitions/min |
|---|---|---:|---:|---:|---:|---:|---:|
| 15mph_flat | oracle | 0.000 | 0.110 | 0.000 | 0.890 | 0.000 | 95.0 |
| 15mph_flat | predicted_point | 0.017 | 0.188 | 0.002 | 0.793 | 0.000 | 89.0 |
| 15mph_down5 | oracle | 0.000 | 0.110 | 0.000 | 0.890 | 0.000 | 95.0 |
| 15mph_down5 | predicted_point | 0.013 | 0.178 | 0.002 | 0.807 | 0.000 | 89.0 |
| 15mph_down8 | oracle | 0.000 | 0.115 | 0.000 | 0.885 | 0.000 | 93.0 |
| 15mph_down8 | predicted_point | 0.017 | 0.160 | 0.002 | 0.822 | 0.000 | 89.0 |
| 20mph_flat | oracle | 0.000 | 0.023 | 0.000 | 0.200 | 0.777 | 93.0 |
| 20mph_flat | predicted_point | 0.002 | 0.007 | 0.002 | 0.407 | 0.583 | 72.0 |
| 20mph_down5 | oracle | 0.000 | 0.018 | 0.000 | 0.187 | 0.795 | 90.0 |
| 20mph_down5 | predicted_point | 0.002 | 0.007 | 0.002 | 0.372 | 0.618 | 75.0 |
| 20mph_down8 | oracle | 0.000 | 0.010 | 0.000 | 0.170 | 0.820 | 91.0 |
| 20mph_down8 | predicted_point | 0.002 | 0.000 | 0.002 | 0.337 | 0.660 | 77.0 |
| 25mph_flat | oracle | 0.000 | 0.000 | 0.000 | 0.067 | 0.933 | 60.0 |
| 25mph_flat | predicted_point | 0.008 | 0.000 | 0.002 | 0.082 | 0.908 | 52.0 |
| 25mph_down5 | oracle | 0.000 | 0.000 | 0.002 | 0.040 | 0.958 | 37.0 |
| 25mph_down5 | predicted_point | 0.010 | 0.000 | 0.002 | 0.017 | 0.972 | 21.0 |
| 25mph_down8 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 25mph_down8 | predicted_point | 0.010 | 0.000 | 0.002 | 0.000 | 0.988 | 10.0 |
| 30mph_flat | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 30mph_flat | predicted_point | 0.010 | 0.000 | 0.000 | 0.000 | 0.990 | 8.0 |
| 30mph_down5 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 30mph_down5 | predicted_point | 0.010 | 0.000 | 0.000 | 0.000 | 0.990 | 8.0 |
| 30mph_down8 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 30mph_down8 | predicted_point | 0.010 | 0.000 | 0.000 | 0.000 | 0.990 | 8.0 |
| 35mph_flat | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 35mph_flat | predicted_point | 0.010 | 0.000 | 0.000 | 0.000 | 0.990 | 8.0 |
| 35mph_down5 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 35mph_down5 | predicted_point | 0.010 | 0.000 | 0.000 | 0.000 | 0.990 | 8.0 |
| 35mph_down8 | oracle | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 |
| 35mph_down8 | predicted_point | 0.010 | 0.000 | 0.000 | 0.000 | 0.990 | 8.0 |
