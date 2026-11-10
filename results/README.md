# Results index

Original directories remain in place. Classification is provided by this index and
`catalog/all_runs.csv`, rather than moving records referenced by frozen metadata.

## Primary protocol-v3 study (CIFAR-10)

| Campaign | Category | Runs | Confirmation |
|---|---|---:|---:|
| [Four strategies](focused_lr_study/20261101_four_strategies_v1/) | Cosine, Warm Restarts, CyclicLR, OneCycleLR search, replication and confirmation | 35 | 12 |
| [Conditional follow-up](focused_lr_study/20261101_conditional_followup_v1/) | Three targeted validation checks; StepLR/MultiStepLR search, replication and confirmation | 19 | 6 |
| [Warm-up + Cosine](focused_lr_study/20261103_warmup_cosine_confirmation_v1/) | Historical configuration, no new tuning | 3 | 3 |
| [Constant LR](focused_lr_study/20261103_constant_confirmation_v1/) | Historical fixed-LR baseline, no new tuning | 3 | 3 |

Within each campaign:

- `00_implementation_checks`: expected schedules and implementation checks.
- `00_targeted_validation`: exploratory validation-only evidence gaps, excluded from confirmation.
- `01_tuning`: candidate screening using validation data only.
- `02_replication`: candidate repeats using validation data only.
- `03_frozen_configs`: selected configuration records preceding confirmation.
- `04_confirmation`: seeds6/7/8 under fixed configurations; final test evaluation of validation-selected checkpoints.
- `protocol`: immutable source snapshot, protocol, environment and design records.
- `summary`: campaign-local tables and comparison plots.

## Portable cross-campaign catalog

| File | Contents |
|---|---|
| [all_runs.csv](catalog/all_runs.csv) | 60 rows, classified by campaign, stage, policy and seed |
| [confirmation_per_seed.csv](catalog/confirmation_per_seed.csv) | 24 rows: eight policies, three matched seeds |
| [confirmation_summary.csv](catalog/confirmation_summary.csv) | Flat mean / sample SD / attained count columns |
| [artifact_manifest.csv](catalog/artifact_manifest.csv) | Original file sizes and SHA-256; storage class identifies local-only checkpoints/logs |
| [audit.json](catalog/audit.json) | Validation scope and completion counts |

Convergence uses validation thresholds only. Missing crossings are not zero; means for
crossing metrics describe attaining seeds only, so always read the associated count.
Do not combine tuning, targeted-validation or older-version runs into confirmation means.

## Completed secondary validation (CIFAR-100)

The [secondary index](secondary_validation/README.md) covers eight unchanged configurations,
new matched seeds 16/17/18 and 24 full 150-epoch runs. All selected checkpoints were locked
before any final test evaluation. It is a transfer check with one additional dataset,
not a new hyperparameter search. Its per-seed and aggregate results remain separate from CIFAR-10.

## Historical and exploratory material

| Folder | Status |
|---|---|
| [formal_multiseed/20261017_092528_epochs150_seeds1-3_fixedsplit_v2](formal_multiseed/20261017_092528_epochs150_seeds1-3_fixedsplit_v2/) | Historical protocol-v2 eight-policy comparison, seeds1/2/3 |
| [lr_pilots/20261016_110243_fixedmomentum_split1_seed1](lr_pilots/20261016_110243_fixedmomentum_split1_seed1/) | Short validation-only pilots and historical frozen configuration |
| [week5_diagnostics/20261023_targeted_pilots_v1](week5_diagnostics/20261023_targeted_pilots_v1/) | Targeted exploratory tuning; parameter versions remain separate |
| [smoketests](smoketests/) | Smoke/debug records, not final comparisons |
| `formal_150epoch`, abandoned pilot/preflight directories | Local historical/partial material excluded by existing Git rules; retained locally |

## Interpretation and preservation

The latest tables compare recorded configurations; **historical tuning budgets are unequal**.
They do not establish a proposal-compliant equal-search-budget final study. The test set
has previously been inspected; no further decisions may use test rankings. Shared-GPU
and serial wall times are not a controlled speed benchmark. Three seeds support limited,
descriptive comparisons only. CIFAR-100 transfer is complete; a historical equal-budget
tuning comparison has not been retroactively performed.

Checkpoints and runtime logs remain on the original computer; dataset/checkpoint binaries
are not uploaded to Git. Historical absolute paths and historic "no GitHub upload" notes in
frozen files are retained as records, not rewritten. This upload follows the user's later
explicit instruction. Frozen experimental files are preserved; the local meeting deck is
reviewed separately and the proposal is outside this code/results upload.
