# Meeting 02 Code Audit

Review date: 20 October 2026.

## Scope and fixes

- Reviewed the training/evaluation loop, metrics, seed setup, scheduler control,
  checkpoint selection, frozen configuration loading, plotting and reporting.
- Corrected `test.py`: previously it imported the training entry point and would
  start a new training run. It now only loads an existing checkpoint and evaluates
  the validation split by default; test evaluation requires explicit selection.
- Removed duplicated single-run plotting from the smoke-test runner and reused
  `scripts/plot_training_curves.py` in both plotting workflows. Figures are closed
  after saving to avoid accumulating open matplotlib figures.
- Preserved the frozen training core so the completed campaign's source SHA256
  remains valid. `.gitattributes` preserves Python file bytes across Git checkout.
- Ignored temporary console logs, the abandoned pilot, historical superseded
  formal campaigns and the machine-specific queue helper; preserved these locally.
- Retained smoke tests as a separate script, LR pilots as a separate script,
  and report/plot utilities because they have distinct uses. Historical snapshots
  are provenance records, not active duplicate modules.

## Validation

- Three controlled-protocol tests pass: config overrides, fixed momentum/decay
  over all 150-epoch schedules, and split invariance across training seeds.
- A validation-loop test checks sample-weighted loss, accuracy, eval mode and
  disabled gradients using unequal-sized batches.
- `test.py` evaluated StepLR seed1's best checkpoint on all 5,000 validation
  examples: accuracy 0.9506, loss 0.1921205110; no training was launched.
- Week 3 verification separately reevaluated six baseline checkpoints and
  checked all 900 baseline epoch records, LR traces and source/split hashes.
- Meeting 02 packaging checked all 24 runs have epochs 1-150 and the same
  source/split hashes; per-seed summary tables contain eight schedulers each.

## Remaining limitations

- Evaluation checkpoints omit optimizer/scheduler/RNG state: exact training
  resumption is unsupported. Restart incomplete runs from epoch 1 in a new folder.
- Do not reuse an output directory for a new run: epoch logs are rewritten while
  result summaries append rows. Use a distinct scheduler/seed/run directory.
- `--config` supplies selected defaults; explicit CLI overrides are permitted.
  Full frozen equality is verified by the report, not enforced by the parser.
- Seed setup covers Python/NumPy/PyTorch and deterministic cuDNN, but does not
  enforce every deterministic algorithm. PYTHONHASHSEED set after startup does
  not change the current interpreter's hash seed. Cross-platform bitwise identity
  is not claimed.
- Epoch-start LR logs undersample batch-level cyclic/OneCycle schedules.
- Dependency files have unpinned package versions; the recorded pilot environment
  and source snapshot identify the original campaign more precisely.
- Historical training commit `55d4d41` does not include every training change.
  Use protocol v2's source snapshot plus SHA256, not the commit alone.

No unrelated existing results were reverted. No new training or test-based tuning
was performed during the audit.
