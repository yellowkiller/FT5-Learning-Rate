# Week 5 diagnostics and conditional Week 6-7 follow-up

## Scope

Existing 24 formal runs are preserved. This is targeted exploratory diagnosis, not equal-budget retuning of all eight strategies. Prior test scores were visible; new choices use validation only.

## Order

1. Existing log audit and convergence analysis (existing_analysis/).
2. Extended training-only range test (lr_range/); halt if max LR 0.1 fails safety gate.
3. Paired pilots (pilot/): cyclic floor, OneCycle rise fraction, restart length. Fresh state for every run.
4. Select using validation tail-5 mean; retain defaults if inconclusive.
5. Only changed strategies receive new 150-epoch runs on seeds 1/2/3 (formal/). Never pool parameter versions.

## Budget and caveats

420 pilot epochs initially; up to 420 additional epochs for close-pair seed2 checks; up to 1350 formal epochs. GPU runtime can span many hours.
Cyclic pilots retain the formal batch cycle length. OneCycle pilots compress the schedule to 30 epochs, so extrapolation is uncertain. Restart pilots use the full 150 epochs to cover the relevant restart windows.
LR range screening provides safety evidence, not optimal hyperparameters. Threshold 0.2pp is a practical decision rule, not a significance test.

## Files

protocol.json freezes candidates and selection rules. schedules/ contains full traces and pre-training plots. source_snapshot/ and source_sha256.json preserve the runnable code. status.json reports progress/failure. Each run retains logs, metadata, checkpoint and separate plots.

## Completed

See selected_parameters.csv, pilot_results.csv and completion.json.
New formal results, if any, remain separate from original runs; formal_summary.csv reports only changed configurations.
