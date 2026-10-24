# Week 4 - Learning-Rate Scheduler Verification

## Status

All Week 4 implementations are present and verified (seven required implementations
when StepLR and MultiStepLR are counted separately, plus one optional schedule):

- Fixed LR
- `StepLR`
- `MultiStepLR`
- `CosineAnnealingLR`
- `CosineAnnealingWarmRestarts`
- `CyclicLR`
- `OneCycleLR`
- Optional linear warm-up + cosine decay

## Implementation

All implementations are centralized in `schedulers.py` and return the scheduler,
the stepping interval, and explicit metadata. Epoch schedulers are stepped after
each validation pass; `CyclicLR` and `OneCycleLR` are stepped after every optimizer
update. The latter two explicitly set `cycle_momentum=False`, so optimizer momentum
remains fixed at 0.9.

| Strategy | Implementation and verified parameters |
| --- | --- |
| Fixed LR | No scheduler; LR remains 0.1. |
| StepLR | `step_size=50`, `gamma=0.1`. |
| MultiStepLR | Milestones 75 and 112, `gamma=0.1`. |
| CosineAnnealingLR | `T_max=150`, `eta_min=0`. |
| Warm Restarts | `T_0=37`, `T_mult=2`, `eta_min=0`; restart after updates 37 and 111. |
| CyclicLR | Batch stepping; base LR 0.01, max LR 0.1, `triangular2`, two-epoch rise, no momentum cycling. |
| OneCycleLR | Batch stepping; max LR 0.1, `pct_start=0.3`, `div_factor=25`, `final_div_factor=10000`, no momentum cycling. |
| Warm-up + cosine | Linear warm-up for 7 epochs from 0.001, then cosine decay to 0. |

## Verification

The frozen 150-epoch schedule CSVs were generated before the major experiments.
The formal run logs independently record the LR used at each epoch. Six automated
tests now verify scheduler intervals, LR metadata, actual trace boundaries, fixed
momentum and weight decay, frozen configuration handling, split invariance, and
validation metric behaviour. The formal run folder contains eight individual plot
files plus `summary_all_schedulers.png`. All six tests passed on 20 October 2026.
An additional submission check compared all 24 formal epoch-start LR logs against
the frozen traces (352 batch updates per epoch for CyclicLR and OneCycleLR).
All 3,600 epoch-start samples matched within numerical tolerance.

The new separate plots are retrospective renderings of the frozen pre-experiment
traces. This does not prove that the new plot layout was reviewed before training.
Metadata `initial_lr` is the configured optimizer reference LR (0.1), whereas
the first LR actually used is 0.01 for CyclicLR, 0.004 for OneCycleLR and 0.001
for warm-up + cosine. Min/max metadata describe configured bounds, not necessarily
the observed extrema of the finite epoch-sampled trace.

For batch-level schedules, formal epoch logs are epoch-start samples. The full
batch-resolution traces remain in the pilot's `formal_schedules/cyclic.csv` and
`onecycle.csv`; their individual plots are downsampled only for visual readability.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python scripts/plot_scheduler_schedules.py results/lr_pilots/20261016_110243_fixedmomentum_split1_seed1/formal_schedules results/formal_multiseed/20261017_092528_epochs150_seeds1-3_fixedsplit_v2/week4_schedules
```

The strategy-specific plots and explanations are included in the Meeting 02
submission. The required baseline plots remain separate from the scheduler plots;
the summary plot is provided only as an overview.
