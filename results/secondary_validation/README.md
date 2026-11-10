# Completed CIFAR-100 transfer comparison

Result directory: [20261108_cifar100_frozen_v1_download_retry1](20261108_cifar100_frozen_v1_download_retry1/).

The first attempt `20261108_cifar100_frozen_v1/` stopped during a truncated data download,
before training or a frozen manifest. That directory stays intact locally and is excluded
from Git. The repaired archive was verified against the official checksum. No completed
or partial training was overwritten; frozen scheduler choices were unchanged.

All **24 runs (8 policies × seeds 16/17/18)** completed on 9 November 2026 at 21:13 Beijing
time. Each has 150 epochs, 52,800 logged optimizer updates and six training/schedule figures.
All 24 best-validation checkpoints were locked at 21:12:24 Beijing time, after training
and before the 24 final test evaluations. Checkpoint ties use lower validation loss, then
earlier epoch. No last-epoch test was used for selection.

| Policy | Selected-model test accuracy (%) |
|---|---:|
| Constant LR | 60.87 ± 0.57 |
| StepLR | 75.92 ± 0.19 |
| MultiStepLR | 76.37 ± 0.28 |
| Cosine | 76.94 ± 0.23 |
| Warm Restarts | 76.49 ± 0.70 |
| Warm-up + Cosine | 77.30 ± 0.54 |
| CyclicLR | 68.60 ± 0.28 |
| OneCycleLR | 76.60 ± 0.39 |

Values are mean ± sample SD across three seeds. Warm-up + Cosine has the highest
observed mean; its difference from Cosine changes sign across seeds. OneCycle and
Warm Restarts are close on average. These results do not establish statistical superiority.
Cyclic retains its early-speed/final-accuracy trade-off. Step has the highest full-budget
validation AUC and is earliest to remain ≥70% through epoch 150 (from epoch 51).
Warm Restarts reaches five consecutive ≥70% epochs earlier (from epoch 36), then drops
after restart. These two convergence definitions must remain distinct.

- [Per-seed metrics](20261108_cifar100_frozen_v1_download_retry1/summary/per_seed.csv)
- [Means, sample SD and attainment counts](20261108_cifar100_frozen_v1_download_retry1/summary/summary.csv)
- [Within-seed differences](20261108_cifar100_frozen_v1_download_retry1/summary/paired_differences.csv)
- [Read-only audit](catalog/audit.json) and [artifact inventory](catalog/artifact_manifest.csv)

The audit rechecks 3,600 epoch rows, 1,267,200 optimizer-update records, frozen hashes,
checkpoint selection, lock/test chronology and every saved summary value without reading
test images. The original-machine audit additionally verified all 48 final checkpoint hashes.
The 50 checkpoint binaries, including the pre-resume backup, remain local (about 3.37 GB).

OneCycle seed17 resumed from the preserved epoch-30 checkpoint. Its original data and
dated recovery implementation are preserved under `recovery_20261109/` and
`protocol/recovery_20261109_source/`. The historical pause recorder is archived there
and must not be treated as a reusable stopping command. No completed run was repeated.
Pause time is excluded from its segmented timing; timing comparisons remain secondary
to convergence epochs/updates. Three seeds, one secondary dataset, unequal historical
tuning budgets and potential power/thermal drift limit broader claims.

The maintained runner now refuses an existing root before changing its status and validates
unique checkpoint-lock coverage. These maintenance fixes apply to future invocations;
the frozen original and recovery sources and all recorded experimental outputs remain intact.
