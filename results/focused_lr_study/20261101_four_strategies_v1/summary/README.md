# Focused LR study

Validated completed runs: 35 / at most 35.

## Configuration choices (validation only)

- cosine: cosine_lr0.1; uncertain=True; Paired seed4/5 best-validation accuracy; <=0.2pp mean gap or opposing signs is uncertain; incumbent fallback when available, otherwise seed4 leader.
- onecycle: onecycle_lr0.03; uncertain=True; Paired seed4/5 best-validation accuracy; <=0.2pp mean gap or opposing signs is uncertain; incumbent fallback when available, otherwise seed4 leader.
- cyclic: cyclic_floor0.01; uncertain=True; Paired seed4/5 best-validation accuracy; <=0.2pp mean gap or opposing signs is uncertain; incumbent fallback when available, otherwise seed4 leader.
- warmrestart: warmrestart_T0_50; uncertain=True; Paired seed4/5 best-validation accuracy; <=0.2pp mean gap or opposing signs is uncertain; incumbent fallback when available, otherwise seed4 leader.

## Interpretation and next steps

All bounds are finite search candidates, not globally optimal learning rates.
Tuning seeds are excluded from confirmation. Do not merge historical configurations into these estimates.
Parallel wall times are descriptive; a scheduler speed claim requires serial timing on controlled hardware.
Three confirmation seeds are descriptive evidence, not proof of a universal ranking.
If confirmation reverses tuning rankings, report uncertainty; do not tune on the inspected test scores.
Next: inspect per-seed consistency and accuracy/stability trade-offs; choose one secondary workload only after this comparison.

## Descriptive confirmation results

- cosine: mean test accuracy 94.9900%
- onecycle: mean test accuracy 94.8900%
- warmrestart: mean test accuracy 94.8500%
- cyclic: mean test accuracy 91.6133%