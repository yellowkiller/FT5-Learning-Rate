# Code, experimental records and Meeting 05 audit — 10 November 2026

## Saved-result verification

The completed secondary campaign contains eight frozen configurations, seeds 16/17/18,
24 training runs and 24 locked-model test evaluations. The independent auditor checks
3,600 epoch rows and 1,267,200 optimizer-update records against independent LR formulas,
fixed momentum, source/split/normalization hashes, validation-only checkpoint selection,
checkpoint-lock coverage and train/lock/test chronology. All per-seed, summary and paired
CSV values are recomputed from saved records. No model training or test-image evaluation
is performed by the audit. All checks passed, including 48 final checkpoint hashes on
the original machine; the two pre-resume checkpoint backups remain preserved as well.

Run `python scripts/audit_secondary_results.py` on a source/results checkout.
Add `--verify-local-checkpoints` on the original machine for checkpoint-file hashing.
The auditor writes only the separate `results/secondary_validation/catalog/` directory.

## Maintained code and historical utilities

- Keep the CIFAR-100 runner, optional verified downloader, hourly local watchdog and
  independent saved-result auditor as the maintained entry points.
- Remove the duplicate top-level dated recovery script. Its identical original remains
  in the campaign's `protocol/recovery_20261109_source/` and its regression test reads
  that preserved implementation. The script is specific to OneCycle/seed17/epoch30;
  it is not a general resume interface.
- Archive the hard-coded pause recorder beside that recovery source and remove its
  top-level entry point. It records a particular historical stop and must not be reused
  to stop future campaigns.
- Re-read completed CSV/checkpoint artifacts at each hourly watchdog inspection rather
  than trusting a cache keyed only by completion JSON. Treat intentional pauses as
  paused, without a spurious missing-queue alert. Default interval is now 3,600 seconds.
- Refuse an existing output root before entering the queue's error-writing handler:
  an accidental new-campaign invocation cannot rewrite completed status as `error`.
- Require all 24 unique frozen lock entries and corresponding training records before
  reading the test dataset. Reject optimized Python execution, which disables protocol
  assertions. These changes affect maintained code for future calls; the original
  frozen and recovery source hashes remain unchanged.

Dataset archives, all checkpoint binaries, console logs, process records, queue locks,
watchdog output and the abandoned download attempt stay local. This is storage
classification, not deletion of results. CSVs, fixed protocols, source provenance,
expected LR arrays, figures and recovery records are versioned for reproducibility.
Git attributes preserve the exact secondary-artifact bytes, including JSON/CSV line
endings, so checkout conversion cannot invalidate the recorded SHA256 inventory.

Validation: all 22 repository unit tests passed, including duplicate-lock rejection,
non-mutating existing-root refusal, intentional pauses and changed-checkpoint detection.
The portable audit mode also passed, and a one-shot watchdog check reported the completed
campaign healthy. No experiment was started by these checks.

## Presentation review and corrections

The reviewed local deliverable is `Week_05_Updated_CIFAR100_Results_Checked.pptx`.
It has 36 slides, including the complete 24 individual secondary runs in two appendix
tables. PPT files remain outside the source repository and are not included in this push.

| Slides | Review/correction |
|---|---|
| 3–9, 24–25 | Identify the primary CIFAR-10 / seeds6–8 scope explicitly. |
| 6, 24 | Cyclic through-end 90%: 121.0 ± 29.7 epochs is conditional on **2/3 attaining seeds**, not all three. Seed6 never qualifies; seed7/8 start at 142/100. Tables and notes now disclose this. |
| 10–18 | Secondary numbers agree with raw logs and summary tables. All eight configurations, validation-only selection and the common final-test lock are reported. |
| 13, 19 | Keep five-consecutive-epoch and through-end convergence distinct: Warm first sustains 70% at 36, but Step first maintains 70% until epoch150 from 51. Warm's through-end onset is 117.3 ± 0.6; both through-end statistics attain 3/3. |
| 14–15 | Identify secondary CIFAR-100 / seeds16–18 scope; AUC/stability leaders differ from primary. |
| 16–17 | Paired differences and cross-dataset means checked. Small mixed-sign differences do not establish OneCycle or Warm-up superiority. |
| 18, 26 | Retain serial/resumed-secondary and shared-GPU-primary timing limitations. No scheduler-efficiency claim is supported by those small time differences. |
| 20–22, 28–34 | All eight frozen settings have traceable selection evidence; fixed design choices and unequal historical budgets are disclosed. |
| 23–24 | Sustained crossings/AUC were retrospective primary supplements, but specified before secondary execution. Replace ambiguous “5-run” by “5-epoch.” |
| 35–36 | Add all 24 per-seed best/final validation, selected-model test accuracy and within-run late SD values. |

Primary metrics were checked against `results/catalog/confirmation_per_seed.csv`, raw
epoch logs and the meeting's derived analysis; secondary metrics were independently
recomputed from the completed campaign. The original nine native chart caches and
embedded workbook records are preserved. Layout rendering and package checks are
performed on the corrected deck; they do not replace numerical/source verification
or constitute proof of identical native rendering in every PowerPoint version. The
corrected deck was additionally opened read-only in the installed Microsoft PowerPoint
and all 36 slides exported to PNG; the corrected tables were visually inspected there.

## Interpretation limits

Three seeds and one additional dataset support descriptive evidence about these frozen
configurations. They do not prove global optimality, statistical equivalence or a universal
stability winner. Historical search budgets remain unequal. OneCycle seed17 has segmented
timing; its CPU random-stream regression is not proof of bitwise full-CUDA continuation.
No tuning, rerunning completed experiments or further test evaluation was performed
as part of this cleanup and presentation review.
