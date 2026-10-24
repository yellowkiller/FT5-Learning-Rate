# FT5 Learning Rate Strategy Study

This repository is the clean experimental workspace for the FT-5 project:

> Optimising Deep Learning Training with Learning Rate Strategies

The project goal is to compare learning-rate strategies under controlled CIFAR-10
training conditions. Keep the dataset, model, optimizer, augmentation, batch
size, weight decay, and training budget fixed whenever possible so that the
learning-rate policy is the main changing variable.

## Latest Week 5 Results

The completed LR-range and targeted pilot campaign is recorded in
`results/week5_diagnostics/20261023_targeted_pilots_v1/`. It contains eight
validation-only pilot runs and six additional 150-epoch runs for the two
strategies whose parameters changed. Original and new parameter versions remain
separate in the comparison tables.

Momentum is fixed at `0.9` and weight decay is fixed at `5e-4` in `config.py`.
They are intentionally not exposed as command-line options for the main
scheduler comparison.

Protocol v2 explicitly disables momentum cycling in CyclicLR and OneCycleLR.
Earlier results for those two methods used scheduler-controlled momentum and
must not be pooled with v2 results. Training seeds vary model initialization,
augmentation and sampling; `--split-seed 1` independently fixes the 45,000/5,000
training/validation partition. Each run saves the split indices, split hash,
source hash and per-epoch momentum alongside existing metrics.

## Validation-Only Pilot And Frozen Configuration

```powershell
python scripts/run_lr_pilot.py --pilot-epochs 3 --range-steps 704
```

This runs a training-only exponential LR range test, then two equal-budget
LR candidates per scheduler. Candidates bracket conservative values below the
observed smoothed-loss minimum on a fixed grid. Every candidate uses the full
training/validation split. Test accuracy is left empty during tuning.
Selection uses final validation accuracy, then lower validation loss and LR.
The three-epoch screen is economical and does not establish optimal long-run
settings. Outputs are timestamped under `results/lr_pilots/`, including plots,
pilot logs, source snapshots, selected settings and `formal_config.json`.

After reviewing a completed pilot, reproduce a main run with its frozen file:

```powershell
python train.py --config results/lr_pilots/<pilot-folder>/formal_config.json --scheduler cosine --seed 1 --run-name formal_v2 --output-dir results/formal_v2/cosine/seed1 --checkpoint-dir checkpoints/formal_v2/cosine/seed1
```

The frozen file sets the 150-epoch budget and selected scheduler LR. Repeat
with seeds 2 and 3 and separate output directories. Explicit CLI arguments can
override defaults and are recorded in metadata. Schedule durations scale with
the chosen epoch budget; momentum and weight decay stay fixed. Run jobs
sequentially on the same GPU for comparable timing. This command is documented
for the next stage; the pilot script does not launch main experiments.

## Completed Formal Comparison and Week 3 Report

The protocol-v2 experiment completed all eight schedulers with seeds 1, 2 and 3
for 150 epochs. The authoritative folder is
`results/formal_multiseed/20261017_092528_epochs150_seeds1-3_fixedsplit_v2/`.
Its `plots/README.md` indexes the per-seed figures; `week3_baseline/README.md`
contains the fixed/StepLR report and checkpoint verification results.
Meeting 02 deliverables and slides are kept outside this Git repository at
`../meeting_submissions/meeting_02_reliable_baseline/`.

Evaluate an existing local checkpoint without training (validation by default):

```powershell
python test.py <checkpoint.pt> --split validation --data-dir ../data --device cuda
```

Generate figures or regenerate the verified baseline report from saved runs:

```powershell
python scripts/plot_multiseed_results.py <experiment-directory>
python scripts/report_week3_baseline.py <experiment-directory>
```

Read [the code audit](docs/code_audit_meeting02.md) for review scope and remaining
limitations. Dataset, checkpoints, temporary logs and abandoned campaigns stay
local; small completed result files and figures are versioned.

Week 4 scheduler implementation and verification are documented in
[docs/week4_scheduler_verification.md](docs/week4_scheduler_verification.md).

## Current Local Assets

- Project plan PDF: `../FT5_Semester_Project_Plan_Wenjun_Huang.pdf`
- Reading papers: `../reading_materials/`
- CIFAR-10 archive and extracted data: `../data/`
- Reference source snapshot: `../external/kuangliu-pytorch-cifar/`

## Reference Source Code

The PDF recommends `kuangliu/pytorch-cifar` as a starting baseline. It is a
small PyTorch CIFAR training repository with common CNN architectures, including
ResNet. In this project it should be treated as a reference implementation, not
as the final experiment framework.

Use it to understand a minimal CIFAR training loop, model definitions, transforms,
checkpointing, and evaluation. Then adapt the useful parts into this repository
with cleaner experiment logging and scheduler control.

## Planned Structure

```text
FT5-Learning-Rate/
  train.py
  test.py
  config.py
  schedulers.py
  models/
    resnet.py
  utils/
    metrics.py
    logger.py
    seed.py
  scripts/
  results/
  checkpoints/
  plots/
  configs/
```

## First Milestone

Establish a reproducible CIFAR-10 + ResNet-18 baseline before comparing
schedulers.

Required first outputs:

- Baseline training log
- Train/validation loss curves
- Train/validation accuracy curves
- Learning-rate curve
- Configuration and seed recorded for every run

`results/results.csv` stores one row per completed run with experiment metadata.
Each run also writes `results/<run_id>_metadata.json` and
`results/<run_id>_epochs.csv`.

## Environment

This workspace uses a dedicated conda environment named `ft5-lr`, created with
Miniconda from `E:\Miniconda3`.

Create the environment from scratch:

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' env create -f environment.yml
```

Run commands inside it:

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n ft5-lr python train.py --epochs 1 --device cuda
```

The current environment has been verified with CUDA-enabled PyTorch.

## Useful Commands

Export preview PNGs from CIFAR-10:

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n ft5-lr python scripts/export_cifar10_preview.py --data-dir ../data --output-dir ../data/cifar10_preview --per-class 10
```

Run a short smoke test:

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n ft5-lr python train.py --epochs 1 --batch-size 256 --scheduler constant --run-name smoke --seed 1 --num-workers 0 --device cuda
```

Plot curves from an epoch log:

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n ft5-lr python scripts/plot_training_curves.py results/smoke_constant_seed1_epochs.csv --output-dir plots
```

## Main Scheduler Candidates

- Constant learning rate
- StepLR or MultiStepLR
- CosineAnnealingLR
- CosineAnnealingWarmRestarts
- CyclicLR
- OneCycleLR
- Optional: warm-up + cosine decay

## Dataset Integrity

The CIFAR-10 Python archive should have MD5:

```text
c58f30108f718f92721af3b95e74349a
```

The local downloaded copy has been checked against this value.

