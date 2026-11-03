# Constant learning-rate baseline: eighth-policy confirmation

Authorization: user requests the remaining strategy from the original eight-policy plan.

## Frozen design

Use `constant` (fixed LR throughout training, not PyTorch's temporary-factor ConstantLR scheduler). Retain LR=0.1 from the original validation-pilot frozen configuration at `results/lr_pilots/20261016_110243_fixedmomentum_split1_seed1/formal_config.json`. No new LR candidates or parameter tuning. This provides the no-scheduling baseline missing from the latest matched-seed comparison; historical seeds1/2/3 are not pooled into it.

Register seeds6/7/8 in advance, all150 epochs, sequential execution (450 epochs total). Copy and hash-check the completed Warm-up + Cosine campaign's frozen training source. Preserve CIFAR-10/ResNet-18, train45000/validation5000/test10000, split seed1 and original split hash, augmentation, normalization, SGD momentum0.9, weight decay0.0005, batch128, workers0, CUDA and CPU threads4. Do not modify previous configurations or result directories.

Checkpoint selection: highest validation accuracy, then lowest validation loss, then earlier epoch. Evaluate only the selected checkpoint once on the test set after training. No test-driven configuration changes, early stopping or additional seeds. Prior other-policy test scores have been inspected; this extension is not a pristine untouched benchmark.

## Implementation checks and output

The legacy schedule-preview helper assumes a scheduler object and cannot handle the existing constant implementation, which correctly returns None. The new queue therefore checks that frozen implementation explicitly and generates the expected constant trace locally. The training implementation is unchanged. Compare all52800 actual optimizer-update LRs to0.1, momentum0.9; check150 epoch rows, exact split, validation-selected best and epoch150 last checkpoints, and six plots per run. Preserve partial runs and fail visibly rather than overwrite or automatically retry.

Outputs include per-seed results and means/sample SD for best/final validation accuracy, selected-checkpoint test accuracy, validation SD over epochs141–150, first80/90/93/94% validation crossings in epochs and cumulative epoch seconds, and training time. Unreached thresholds remain missing and have explicit counts. Generate eight-policy tables for all24 matched-seed confirmation runs and descriptive paired differences against Cosine. Existing results are read only; do not rerun prior policies.

Scope: compare the frozen fixed-LR baseline with the seven scheduled policies. Three seeds, unequal historical tuning budgets, previously inspected benchmark results and differing serial/shared-GPU timing conditions limit conclusions. Do not infer scheduler speedups from raw historical timing differences or claim globally optimal LR. Do not change proposal/PPT or upload to GitHub.

Output directory: `results/focused_lr_study/20261103_constant_confirmation_v1`. Preserve source snapshot, environment, protocol, historical configuration, process/queue logs, per-run artifacts and summary tables. After all three audits pass, mark complete.
