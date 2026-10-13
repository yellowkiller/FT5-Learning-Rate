# Smoke Test Experiment

This folder contains the 30-epoch seed-1 smoke test comparison for all scheduler methods mentioned in the project plan.

## Parameters

- Epochs: 30
- Seed: 1
- Batch size: 128
- Initial LR: 0.1
- Device: cuda

## Result Summary

| scheduler | best_val_acc | final_val_acc | test_acc | training_time_seconds | scheduler_params |
| --- | --- | --- | --- | --- | --- |
| onecycle | 0.9384 | 0.9384 | 0.9358 | 605.5194571000029 | max_lr=0.1;epochs=30;steps_per_epoch=352;pct_start=0.3;div_factor=25.0;final_div_factor=10000.0 |
| cosine | 0.9344 | 0.9332 | 0.9287 | 598.4577466000046 | T_max=30;eta_min=0.0 |
| multistep | 0.9272 | 0.9272 | 0.9195 | 615.1514132000011 | milestones=[15, 22];gamma=0.1 |
| warmup_cosine | 0.9296 | 0.9296 | 0.9194 | 599.723327700005 | warmup_epochs=1;start_factor=0.01;cosine_epochs=29;eta_min=0.0 |
| warmrestart | 0.9194 | 0.8218 | 0.914 | 599.2401953999943 | T_0=7;T_mult=2;eta_min=0.0 |
| step | 0.9202 | 0.9168 | 0.9071 | 621.3528006999986 | step_size=10;gamma=0.1 |
| cyclic | 0.9092 | 0.9008 | 0.8988 | 603.0274438999986 | base_lr=0.01;max_lr=0.1;step_size_up=704;mode=triangular2 |
| constant | 0.8698 | 0.8328 | 0.8622 | 616.0224185999978 | fixed_lr |

## Folder Structure

- one subfolder per scheduler
- each scheduler folder contains `results/`, `plots/`, and local `checkpoints/`
- comparison tables are in `tables/`
- comparison figures are in `plots/comparison/`
