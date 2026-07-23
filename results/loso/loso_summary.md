# Leave-One-Scene-Out Cross-Validation

Folds: ['c8', 'c9a', 'c9b', 'c9c', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15']. Each row trains on the other scenes and
tests zero-shot on the held-out scene (conf 0.25, IoU 0.5, joint tol 5 m).

## Per-fold detection-only envelope

| held-out scene | envelope (m) | R 30-40 | R 40-50 | R 50-60 |
|---|---:|---:|---:|---:|
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

**Detection-only envelope across folds: 59 +/- 3 m**

## Per-band aggregates (mean +/- std across folds)

| band | recall | matched MAE (m) | joint <=5 m |
|---|---:|---:|---:|
| 30-40 | 0.917 +/- 0.117 | 3.24 +/- 1.87 | 0.747 +/- 0.231 |
| 40-50 | 0.888 +/- 0.100 | 3.58 +/- 1.68 | 0.679 +/- 0.200 |
| 50-60 | 0.850 +/- 0.127 | 3.64 +/- 1.10 | 0.623 +/- 0.153 |

## Held-out forest scene

The rural aggregate above excludes `c7`, the only enclosed-canopy forest scene.
When it is held out and the training set contains no forest scene, recall is
0.272, 0.172, and 0.072 and joint success is 0.133, 0.098, and 0.046 across
the 30--40, 40--50, and 50--60 m bands. The machine-readable record is included
in `loso_results.json`.
