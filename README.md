# FT5 Learning Rate Strategy Study

This repository is the clean experimental workspace for the FT-5 project:

> Optimising Deep Learning Training with Learning Rate Strategies

The project goal is to compare learning-rate strategies under controlled CIFAR-10
training conditions. Keep the dataset, model, optimizer, augmentation, batch
size, weight decay, and training budget fixed whenever possible so that the
learning-rate policy is the main changing variable.

Momentum is fixed at `0.9` and weight decay is fixed at `5e-4` in `config.py`.
They are intentionally not exposed as command-line options for the main
scheduler comparison.

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

