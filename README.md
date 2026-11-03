# FT5 Learning Rate Strategy Study

Controlled CIFAR-10 / ResNet-18 experiments comparing eight learning-rate policies with SGD.

## Current results

The latest completed comparison has **eight policies × seeds 6/7/8 × 150 epochs**.
All 60 runs in the four recent campaigns have passed a read-only integrity audit:
36 tuning/replication/targeted-validation runs and 24 frozen-configuration confirmation runs.

- [Results index and stage definitions](results/README.md)
- [All 60 runs](results/catalog/all_runs.csv)
- [24 matched-seed confirmation runs](results/catalog/confirmation_per_seed.csv)
- [Eight-policy means, sample SD and counts](results/catalog/confirmation_summary.csv)
- [Audit report](results/catalog/audit.json)
- [Artifact SHA-256 inventory](results/catalog/artifact_manifest.csv)
- [Code cleanup and methodological limitations](docs/code_audit_20261103.md)

These are controlled comparisons of the recorded configurations, **not a completed equal-tuning-budget study**.
Historical search effort differs across policies. Test results have already been inspected;
they must not drive further parameter/model selection. Historical shared-GPU times and the
later serial times do not establish scheduler speedups. Three seeds give descriptive evidence,
not universal rankings. The proposal's fair-budget study and secondary workload remain separate future work.

## Experimental protocol

- Fixed CIFAR-10 split: 45,000 training / 5,000 validation / 10,000 official test; split seed 1.
- CIFAR-adapted ResNet-18, SGD, momentum 0.9, weight decay 0.0005, batch size 128.
- Constant LR, StepLR, MultiStepLR, Cosine, Warm Restarts, CyclicLR, OneCycleLR, Warm-up + Cosine.
- CyclicLR and OneCycleLR use fixed momentum (`cycle_momentum=False`).
- Recent protocol v3 logs every optimizer-update LR and uses validation accuracy,
  lower validation loss, then earlier epoch for checkpoint selection.
- Tuning/replication/targeted runs use `--validation-only`. After configuration freezing,
  each confirmation run evaluates its validation-selected checkpoint once on the test set.
- Stability: validation-accuracy sample SD over epochs 141–150. Convergence: first validation
  crossings at 80%/90% (plus 93%/94% for compatibility), with missing thresholds and attainment counts.
- Protocol versions and historical tuning results are kept separate; do not pool them into confirmation estimates.

## Code layout

| Location | Purpose |
|---|---|
| `train.py`, `test.py` | Training and explicit checkpoint evaluation |
| `config.py`, `schedulers.py`, `models/`, `utils/` | Shared training implementation |
| `scripts/run_focused_lr_study.py` | Original four-policy bounded search/confirmation |
| `scripts/run_lr_followup.py` | Targeted validation plus StepLR/MultiStepLR |
| `scripts/run_fixed_confirmation.py` | Shared fixed Constant / Warm-up + Cosine confirmation |
| `scripts/run_constant_confirmation.py`, `scripts/run_warmup_confirmation.py` | Thin CLI compatibility entry points |
| `scripts/audit_focused_results.py` | Read-only result checks and portable catalog |
| Other `scripts/` | Historical pilots, smoke checks and reports; retained for reproducibility |
| `tests/` | Training, scheduler, checkpoint and queue regression tests |
| `results/` | Versioned metrics, configurations, frozen sources and figures |

Each campaign's `protocol/source_snapshot/` is immutable experimental provenance,
not active duplicate code. Use it and its recorded hashes to reproduce historical runs.
The cleaned active scripts apply to new roots. They do not migrate or rerun completed campaigns.

## Setup and checks

Create the dedicated Conda environment from `environment.yml` or install `requirements.txt`
in a compatible Python/CUDA environment. Per-campaign `protocol/environment.txt` records the
actual dependencies; CPU-only audit of checkpoints still requires compatible PyTorch.
CIFAR-10 data belongs outside this repository, conventionally at `../data`.

```powershell
python -m unittest discover -s tests -v
python scripts/audit_focused_results.py
```

The audit reads existing files and local checkpoints; it performs no training or test-image evaluation.
It writes only `results/catalog/`. It requires all four original local result directories and checkpoints.
A GitHub checkout contains their metrics/configuration/source/figure records, but not checkpoint binaries.

For an intentionally new training run, use `python train.py --help` and a **new output/checkpoint root**.
Existing incomplete directories must be inspected rather than overwritten. Exact resume is not provided:
last checkpoints store recovery state, but there is no validated resume CLI.

## Storage and history

Git versions completed CSV/JSON results, LR logs, curves, frozen source snapshots and their environment records.
Dataset binaries, all `.pt` checkpoints, runtime console logs, process records, lock files and Python caches
stay local. The artifact manifest records the original paths, sizes and hashes, including local-only files.
All original experiment directories are retained in place to preserve frozen metadata and absolute paths;
new catalog tables use repository-relative directories for browsing on another machine.

Older protocol-v2 eight-policy seeds1/2/3 results, short LR pilots, Week5 targeted work and smoke checks
are indexed separately in [results/README.md](results/README.md). V1 momentum-cycling results and abandoned
campaigns must not be pooled with v2/v3. Meeting/proposal/PPT files live outside this repository and are
not part of this cleanup or upload. Publication here does not alter historical experimental status.
