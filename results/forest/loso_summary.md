# Leave-One-Scene-Out Cross-Validation

Folds: ['c7', 'f1']. Each row trains on the other scenes and
tests zero-shot on the held-out scene (conf 0.25, IoU 0.5, joint tol 5 m).

## Per-fold detection-only envelope

| held-out scene | envelope (m) | R 30-40 | R 40-50 | R 50-60 |
|---|---:|---:|---:|---:|
| c7 | 0 | 0.503 | 0.595 | 0.627 |
| f1 | 60 | 0.934 | 0.879 | 0.738 |

**Detection-only envelope across folds: 30 +/- 30 m**

## Per-band aggregates (mean +/- std across folds)

| band | recall | matched MAE (m) | joint <=5 m |
|---|---:|---:|---:|
| 30-40 | 0.718 +/- 0.216 | 2.32 +/- 0.02 | 0.641 +/- 0.185 |
| 40-50 | 0.737 +/- 0.142 | 2.57 +/- 0.22 | 0.644 +/- 0.110 |
| 50-60 | 0.683 +/- 0.055 | 3.03 +/- 0.36 | 0.556 +/- 0.072 |
