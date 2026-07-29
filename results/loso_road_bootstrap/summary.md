# Rural LOSO Road-Level Bootstrap

Sampling unit: held-out rural road (n=10). Percentile intervals use 100,000 resamples with seed 20260728.
These intervals describe heterogeneity across the represented roads; they do not include model-training seed variation or support population-wide safety inference.

| band | metric | mean (95% CI) | roads passing | pass rate (95% CI) |
|---|---|---:|---:|---:|
| 30-40 | recall | 0.917 (0.835--0.976) | 10/10 | 1.00 (1.00--1.00) |
| 30-40 | mae | 3.239 (2.205--4.497) | 9/10 | 0.90 (0.70--1.00) |
| 30-40 | joint | 0.747 (0.600--0.884) | 7/10 | 0.70 (0.40--1.00) |
| 40-50 | recall | 0.888 (0.821--0.945) | 10/10 | 1.00 (1.00--1.00) |
| 40-50 | mae | 3.583 (2.673--4.717) | 8/10 | 0.80 (0.50--1.00) |
| 40-50 | joint | 0.679 (0.553--0.799) | 7/10 | 0.70 (0.40--1.00) |
| 50-60 | recall | 0.850 (0.761--0.914) | 9/10 | 0.90 (0.70--1.00) |
| 50-60 | mae | 3.644 (3.041--4.392) | 9/10 | 0.90 (0.70--1.00) |
| 50-60 | joint | 0.623 (0.525--0.714) | 6/10 | 0.60 (0.30--0.90) |
