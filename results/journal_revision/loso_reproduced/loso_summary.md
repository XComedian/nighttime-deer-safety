# Leave-One-Scene-Out Cross-Validation

Folds: ['c7', 'c8', 'c9a', 'c9b', 'c9c', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15']. Each row trains on the other scenes and
tests zero-shot on the held-out scene (conf 0.25, IoU 0.5, joint tol 5 m).

## Per-fold detection-only envelope

| held-out scene | envelope (m) | R 30-40 | R 40-50 | R 50-60 |
|---|---:|---:|---:|---:|
| c7 | 0 | 0.272 | 0.172 | 0.072 |
| c8 | 60 | 0.604 | 0.720 | 0.842 |
| c9a | 60 | 0.979 | 0.913 | 0.902 |
| c9b | 60 | 0.881 | 0.782 | 0.732 |
| c9c | 60 | 0.989 | 0.958 | 0.960 |
| c10 | 50 | 0.826 | 0.714 | 0.512 |
| c11 | 60 | 0.959 | 0.966 | 0.926 |
| c12 | 60 | 0.966 | 0.960 | 0.894 |
| c13 | 60 | 0.996 | 0.957 | 0.899 |
| c14 | 60 | 0.985 | 0.947 | 0.923 |
| c15 | 60 | 0.985 | 0.962 | 0.908 |

**Detection-only envelope across folds: 54 +/- 17 m**

## Per-band aggregates (mean +/- std across folds)

| band | recall | matched MAE (m) | joint <=5 m |
|---|---:|---:|---:|
| 30-40 | 0.858 +/- 0.216 | 3.40 +/- 1.84 | 0.704 +/- 0.277 |
| 40-50 | 0.823 +/- 0.227 | 3.70 +/- 1.50 | 0.626 +/- 0.241 |
| 50-60 | 0.779 +/- 0.254 | 3.97 +/- 1.20 | 0.544 +/- 0.218 |
