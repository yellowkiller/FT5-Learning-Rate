# Code cleanup and results audit — 3 November 2026

## Cleanup scope

Reviewed training, checkpoint selection, schedule logging, experiment queues, result layout
and publication exclusions. Consolidated duplicated fixed-policy queue code from Constant
and Warm-up + Cosine into `scripts/run_fixed_confirmation.py`, retaining their existing CLI
entry points as small wrappers. The two previous active implementations contained over400 lines;
their original bytes remain in the immutable campaign source snapshots.

The shared queue retains historical fixed choices, validation-only checkpoint selection,
seed registration, source hashes, no partial-run overwrite, an exclusive queue lock and
the existing per-run audits. It is intended for explicitly initialized **new** roots;
existing completed campaigns must continue to use their own frozen snapshots.

Fixed the active general schedule-preview helper to handle the constant policy's `None`
scheduler object. This is a preview fix, not a change to training or recorded schedules.
The existing uncommitted protocol-v3 training improvements are included: per-update LR
logging, finite-value checks, accuracy/loss/earlier-epoch checkpoint ties, and periodic
last checkpoints. No model training, parameter search or test-image evaluation was performed
as part of this cleanup. No frozen source, checkpoint, figure or existing CSV was rewritten.

Retained distinct historical pilot, follow-up, smoke-test and report scripts because they
encode different protocols or remain referenced by historical records. Historical source
snapshots are provenance, so removing them would break traceability. Python caches,
runtime locks and console/process logs are excluded from Git. Dataset binaries and all
checkpoint binaries remain local. Historical result directories are not renamed or moved.

## Result classification and audit

Added `results/README.md` and the regenerable, portable `results/catalog/` tables.
Classification separates candidate tuning, finalist replication, targeted validation,
frozen choices and held-out-training-seed confirmation. Original artifacts remain indexed
by campaign and stage. Catalog paths are relative; original absolute metadata stays intact.
The manifest fingerprints original artifacts, including local-only checkpoints and logs.

All60 recent runs passed read-only checks: 150 ordered epoch rows, exact split hash,
metadata/optimizer conditions, 52,800 actual optimizer-update LRs, absent test scores in
tuning/replication/targeted validation, present test scores in confirmation, validation-selected
best checkpoint, epoch150 last checkpoint with optimizer state, and six per-run plots.
Every registered frozen source SHA-256 also matched. The24 confirmation rows cover exactly
eight schedulers × seeds6/7/8. This validates file consistency, not scientific fairness or
the completeness of every historical test-inspection decision.

## Verification

`python -m unittest discover -s tests -v`:15 tests pass, including independent 150-epoch
trace checks for both fixed-policy queues, initialization/freezing and existing-root refusal,
constant-policy preview, checkpoint ties, update-LR recording and nonfinite-loss failure.

`python scripts/audit_focused_results.py`:60/60 runs and24/24 confirmation records pass.
This command requires local checkpoints; it does not reevaluate any checkpoint on images.

## Remaining limitations

- Historical tuning budgets differ. Existing results must not be advertised as a completed
  equal-budget tuning study or proof of globally optimal configurations.
- The official test set was inspected previously; further selection must use validation only.
- Early campaigns used shared-GPU execution; later fixed-policy jobs were serial. Raw time
  differences do not establish a scheduler speedup.
- Last checkpoints contain recovery state, but a validated exact-resume CLI is not implemented.
- Three seeds and one primary workload give limited descriptive evidence. CIFAR-100 transfer,
  controlled timing and a separately frozen fair-budget study remain future work.
- A Git clone excludes checkpoint binaries and raw runtime logs. The artifact manifest records
  their local paths and hashes, not a downloadable public checkpoint archive.
