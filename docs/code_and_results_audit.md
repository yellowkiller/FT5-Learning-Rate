# Code and Results Audit

Date: 2026-10-24

## Code

- `python -m unittest discover -s tests -v`: 9 tests passed.
- Python compilation completed for the repository sources.
- Scheduler option overrides are limited to documented parameters for
  Warm Restarts, CyclicLR and OneCycleLR.
- The default scheduler traces remain unchanged when no override is supplied.
- The Week 5 runner uses the source snapshot saved with the campaign so that a
  later workspace edit cannot silently change a running experiment.
- Source snapshots under result directories are provenance copies, not runtime
  modules used by the main training command.

## Results

- Historical comparison: 24 complete runs, 8 strategies, 3 seeds, 150 epochs.
- Week 5 pilots: 8 complete validation-only runs.
- Follow-up formal runs: 6 complete runs, 2 selected configurations, 3 seeds,
  150 epochs each.
- Every pilot and follow-up run has a complete epoch CSV, `results.csv`,
  metadata and a checkpoint locally.
- Original and selected parameter versions remain separate in the comparison
  tables. No results are pooled across configurations.

## Git scope

The repository keeps code, configuration, metadata, CSV summaries and plots.
Checkpoint binaries, queue logs, process IDs, Python caches and temporary
runtime files are excluded by `.gitignore`. The dataset remains local.
