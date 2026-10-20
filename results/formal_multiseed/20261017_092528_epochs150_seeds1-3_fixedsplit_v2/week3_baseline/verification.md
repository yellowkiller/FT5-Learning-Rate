# Baseline verification

All six runs passed structural, numeric, split, LR-sequence and checkpoint checks.
See [machine-readable checks](tables/verification.csv).

- Each log has epochs 1-150; all numeric values finite, losses nonnegative, accuracies in [0,1].
- Fixed 45,000/5,000 disjoint train/validation split, seed 1; identical across training seeds 1,2,3.
- Logged epoch-start LR matches reconstructed scheduler for every epoch; momentum 0.9, weight decay 5e-4.
- CrossEntropyLoss is averaged by sample count; accuracy uses argmax correct/total counts.
- Evaluation uses model.train(False) and torch.no_grad(); no augmentation or shuffling for validation.
- Best checkpoint is saved on strictly improved validation accuracy (earliest epoch wins ties).
- Loaded all six checkpoints and reevaluated validation: accuracy matches within 1e-9; loss within 1e-5.
- Test metrics were read from existing summaries; no new test-set evaluation or test-driven selection.
- RNG setup covers Python, NumPy, CPU/CUDA PyTorch; cuDNN deterministic=True, benchmark=False.
- Bitwise reproducibility is not guaranteed across hardware/software; deterministic algorithms are not enforced globally.
- PYTHONHASHSEED is set inside Python, so it does not reset the current interpreter's hash seed.
- Checkpoints contain model/config/metadata, not optimizer/scheduler/RNG state; adequate for evaluation, not exact training resumption.
- Logged total duration includes checkpoint saving and final test evaluation; epoch-time sum is provided separately.

Source SHA256: `94450c9eb9a70d3ec5737729c43479c03ae46b34f2f9afeb5f96dd582d1c8efd`

Split SHA256: `eaeb385c407962ceb057cbe0552bff8071a52d9b2c66a0c7432da30205174ad1`

Git commit recorded at training: `55d4d41`.
The training code contained uncommitted modifications; this commit alone is not the exact experiment version.
Use the matching [source snapshot](../../../lr_pilots/20261016_110243_fixedmomentum_split1_seed1/source_snapshot/)
plus the source SHA256. Current hashed training files match the frozen version.
