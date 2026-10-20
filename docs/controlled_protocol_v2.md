# Controlled protocol v2

## Scope

This update covers momentum control, an economical LR pilot, and a frozen
dataset split/main configuration. It does not launch the three-seed main study.
Previous formal and smoke-test results are retained unchanged.

## Corrections

- CyclicLR and OneCycleLR explicitly use `cycle_momentum=False`. SGD momentum
  stays at 0.9 and weight decay stays at 5e-4. Historical runs of these two
  schedulers used momentum cycling and must not be pooled with v2.
- `--split-seed` defaults to 1, independent of `--seed`. The fixed partition
  contains 45,000 training and 5,000 validation images; the official test set
  contains 10,000 images. Each run saves exact split indices and SHA-256.
- Epoch CSV files now record momentum. Metadata and result summaries include
  split seed, split hash, source hash and protocol version. StepLR minimum-LR
  metadata reflects the last learning rate used during training.
- `--validation-only` skips final test evaluation. Empty test metrics in pilot
  CSVs are intentional, not failed evaluation.

## Pilot Design

`scripts/run_lr_pilot.py` performs an exponential training-only LR sweep from
1e-4 to 1 over up to 704 optimizer steps. The range test is diagnostic: training
loss confounds increasing LR with continued learning and is not an optimal-LR
estimator. Two candidates are selected conservatively from the predefined grid
0.01, 0.03, 0.1, 0.3 using the observed smoothed-loss minimum. The minimum bracket
is 0.01/0.03. A detected divergence stops the sweep.

Every scheduler then receives exactly two candidates, each trained for three
epochs on the full training split with identical seed, batch size, augmentation,
optimizer and hardware. Jobs run sequentially. Each candidate starts a new
process, model and optimizer; range-test weights are discarded.

Selection maximizes final validation accuracy; ties prefer lower final
validation loss, then lower LR. Test metrics are never used for selection.
Only the LR amplitude is screened; other scheduler parameters use the existing
rules in `schedulers.py`. CyclicLR uses base LR = amplitude/10 and a two-epoch
rise; OneCycleLR uses maximum LR = amplitude, div_factor=25 and
final_div_factor=10000. The frozen JSON includes full 150-epoch scheduler
metadata and prospective LR traces.

## Limitations

Three epochs are a coarse screening budget. Scheduler durations are compressed
for the pilots and scale with the 150-epoch main budget, so short-run rankings
may not predict long-run rankings. In particular, StepLR and MultiStepLR have
identical decay points at this three-epoch budget; their identical pilot
results do not show that the methods are equivalent at 150 epochs. Warm-up
also occupies a larger fraction of a three-epoch run than a 150-epoch run.
The LR sweep did not necessarily locate an
instability boundary. These settings are reasonable pilot-selected candidates,
not demonstrated optima. Only seed 1 is used for tuning; seeds 1, 2 and 3 are
reserved in the subsequent main protocol. Do not tune further on test accuracy.

## Reproduction

Run from the project directory in the dedicated `ft5-lr` Conda environment:

```powershell
python scripts/run_lr_pilot.py --pilot-epochs 3 --range-steps 704
python -m unittest discover -s tests -v
```

Each completed pilot directory contains protocol/environment JSON, LR sweep CSV
and plot, 16 candidate result folders, pilot and selection tables, source
snapshots, prospective 150-epoch schedules and `formal_config.json`.

The frozen JSON supplies defaults to `train.py --config <path> --scheduler
<name> --seed <seed>`. Explicit CLI overrides are allowed and recorded; an
override changes the protocol and should not be silently pooled with the
frozen comparison. Use a separate output/checkpoint directory per seed and
scheduler. Keep the environment, data, split and source snapshot fixed.

The existing first formal seed-1 experiment remains historical evidence. New
multi-seed comparisons should use one consistent v2 protocol; changing selected
LRs or momentum invalidates direct pooling with earlier runs.
