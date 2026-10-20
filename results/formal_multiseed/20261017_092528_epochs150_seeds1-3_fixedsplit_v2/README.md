# Completed Protocol-v2 Formal Comparison

24 complete runs: eight schedulers, training seeds 1/2/3, 150 epochs each.
All share split_seed=1, CIFAR-10/ResNet-18, batch 128, SGD momentum 0.9
and weight decay 5e-4. Parameters were frozen after validation-only pilots.

- [Frozen configuration](formal_config.json)
- [All 24 run summaries](tables/all_runs.csv)
- [Mean and sample standard deviation](tables/scheduler_mean_std.csv)
- [Per-seed figures](plots/README.md)
- [Verified Week 3 baseline report](week3_baseline/README.md)
- [Baseline checkpoint checks](week3_baseline/verification.md)

Each scheduler/seed directory contains epoch CSV, summary CSV, metadata and
split indices. Checkpoints remain local and are excluded from Git. Temporary
console logs and the old seed2 queue helper remain local as well.

Accuracy columns are fractions (0-1). Cross-seed SD is computed with ddof=1,
not a confidence interval. Test results are from validation-selected checkpoints;
do not tune schedules on these test numbers. Historical momentum-cycling runs
must not be pooled with protocol v2.

The training commit was 55d4d41 plus uncommitted modifications. Frozen source
snapshots and SHA256 identify the actual version. Later publication commits do
not change that historical training provenance.
