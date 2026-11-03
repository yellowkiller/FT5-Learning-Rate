# Fixed Warm-up + Cosine confirmation

Validated runs: 3/3.

Historical LR=.1, warmup7 epochs, start_factor=.01, cosine143 epochs, eta_min=0. No new tuning.
Seeds6/7/8; checkpoint selected only by validation accuracy/loss/earlier epoch; one final test evaluation.
Tables report per-seed metrics, means and sample SD (ddof=1); count identifies unavailable thresholds.
Stability: validation SD over epochs141-150. First80/90/93/94% crossings use validation only; seconds include validation overhead.
Seven-policy tables and paired-vs-Cosine differences are descriptive; three seeds do not establish universal superiority.
Prior test results were inspected; this extension is not a pristine independent holdout. No test-driven changes are allowed.
New runs are serial; historical runs shared the GPU. Training-time differences cannot establish scheduler speedups.
Unequal historical tuning budgets limit fairness claims. ConstantLR is outside this campaign.
