# Warm-up + Cosine: frozen seventh-policy confirmation

User authorization: conduct one additional strategy experiment using the prior design where adequate.

## Choice and scope

Warm-up + Cosine is already in the original eight-policy plan and implemented in the existing frozen training snapshot. It has historical pilot/baseline evidence, but no seeds6/7/8 confirmation alongside the latest six-policy comparison. This campaign fills that gap; it does not introduce a ninth policy or retune existing policies.

Retain the configuration recorded in `results/lr_pilots/20261016_110243_fixedmomentum_split1_seed1/formal_config.json`: peak LR=0.1, linear start factor=0.01, 7 warm-up epochs, followed by 143-epoch cosine decay with eta_min=0. The first training epoch uses LR=0.001; epoch8 reaches 0.1. The mathematical zero is reached after the last scheduler step, not during the final optimizer update. No warm-up-duration or amplitude search is authorized by this campaign design.

## Frozen conditions and evaluation

- Copy, hash-check and run the six-policy campaign's frozen training source; never modify either prior result directory.
- CIFAR-10 / ResNet-18; 45,000 train / 5,000 validation / 10,000 official test, split seed1 and the identical saved split hash.
- Same augmentation, normalization, SGD momentum0.9, weight decay0.0005, batch128, workers0, CPU threads4, CUDA, 150 epochs.
- Register seeds6/7/8 before training; all three use the same fixed configuration. These seeds were not used to tune Warm-up + Cosine. Their other-policy test scores have already been inspected, so this is an extension of a previously inspected benchmark, not a pristine new holdout.
- Select the checkpoint by maximum validation accuracy, then lower validation loss, then earlier epoch. Evaluate that checkpoint once on the test set after training. No test-based changes, additional candidates, early stopping or selective seed reporting.
- Three sequential runs, 450 epochs total. Serial execution avoids competition between this campaign's jobs; historical six-policy times remain shared-GPU descriptive measurements, so unequal timing conditions prohibit a direct speedup claim.

## Checks, reports and boundaries

Before training, verify source hashes, the frozen historical configuration, all six existing seed triplets, and an independent analytic schedule trace. During completion audit, check 150 epoch rows, all52,800 optimizer-update LRs, split/optimizer metadata, validation-selected best and epoch150 last checkpoints, and six plots per run. Never overwrite or automatically restart a partial run.

Report each seed and mean/sample SD of best/final validation accuracy, selected-checkpoint test accuracy, epochs141–150 validation sample SD, and training time. First validation crossings at80%/90% are primary convergence metrics; retain90%/93%/94% for compatibility, with unreached thresholds missing rather than zero. Report epochs and cumulative epoch seconds, which include validation overhead. Recompute these metrics from the old epoch CSVs for a seven-policy paired-seed table without rerunning or changing old experiments. Produce paired differences against ordinary Cosine (same peak LR, no warm-up) for all seeds without selecting a winner from test scores.

The main hypothesis is whether the fixed warm-up improves early convergence or stability at the same training budget. Three seeds and unequal historical tuning effort support descriptive comparisons only. This is not a causal ablation against policies with different amplitudes, nor evidence of global optimality. ConstantLR remains outside this one-strategy request.

Output: `results/focused_lr_study/20261103_warmup_cosine_confirmation_v1`. Save protocol, frozen choice, source hashes/environment, queue/process logs, per-run artifacts and seven-policy summary. No proposal/PPT changes or GitHub upload.
