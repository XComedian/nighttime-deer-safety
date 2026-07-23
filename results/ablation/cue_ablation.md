# Road-Furniture Cue Ablation

One checkpoint (`output/loso_asset/c14/weights/best.pt`) evaluated zero-shot on four scenes that are
identical in geometry, vegetation, lighting and seed, differing ONLY in whether lane
markings and delineator posts are present. Any difference therefore isolates the cue.

## Matched range error (m) by distance band

| variant | markings | posts | 30-40 | 40-50 | 50-60 |
|---|:--:|:--:|---:|---:|---:|
| cueA markings + posts (control) | yes | yes | 1.82 | 2.23 | 2.31 |
| cueB markings only | yes | no | 1.41 | 1.92 | 2.05 |
| cueC posts only | no | yes | 3.18 | 3.99 | 2.89 |
| cueD neither | no | no | 3.65 | 3.75 | 3.08 |

## Detection recall by band

| variant | 30-40 | 40-50 | 50-60 |
|---|---:|---:|---:|
| cueA | 0.993 | 0.961 | 0.947 |
| cueB | 0.991 | 0.966 | 0.885 |
| cueC | 0.991 | 0.932 | 0.967 |
| cueD | 1.000 | 0.963 | 0.902 |

## Joint success (detected AND |range error| <= 5 m)

| variant | 30-40 | 40-50 | 50-60 |
|---|---:|---:|---:|
| cueA | 0.964 | 0.891 | 0.870 |
| cueB | 0.974 | 0.918 | 0.835 |
| cueC | 0.786 | 0.649 | 0.802 |
| cueD | 0.731 | 0.676 | 0.714 |

## Effect size (cueD minus cueA, range MAE in m)

  30-40: +1.82, 40-50: +1.52, 50-60: +0.78

Positive values mean removing both cues WORSENS ranging. Detection recall is
reported alongside so a range change is not confused with a detection change.
