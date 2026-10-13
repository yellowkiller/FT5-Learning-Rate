# Initial Baseline Report

## Configuration

- Dataset: CIFAR-10
- Split: 45,000 train / 5,000 validation / 10,000 test
- Model: CIFAR-style ResNet-18
- Framework: PyTorch
- Device: CUDA GPU
- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 5e-4
- Batch size: 128
- Epochs: 30
- Scheduler: MultiStepLR
- Initial LR: 0.1
- Seed: 1

## Result

- Best validation accuracy: 0.9272
- Test accuracy: 0.9195
- Test loss: 0.2516
- Training time: 596.36 seconds
- Epoch log: `results/meeting_baseline_multistep_seed1_epochs.csv`
- Curves: `plots/meeting_baseline_multistep_seed1_curves.png`
- Checkpoint: `checkpoints/meeting_baseline_multistep_seed1_best.pt`

## Record Keeping

The training script now records the experimental metadata required by the
project plan:

- experiment ID
- date
- Git commit
- dataset
- architecture
- optimizer
- scheduler
- initial, minimum, and maximum LR
- scheduler parameters
- batch size
- momentum
- weight decay
- epochs
- seed
- best and final validation accuracy
- test accuracy and loss
- training time
- checkpoint path
- metadata JSON path

Per-epoch logs are saved separately with loss, accuracy, LR, and epoch time.

## Observations

Validation accuracy improved sharply after the first MultiStepLR decay from
0.1 to 0.01 at epoch 16. It improved again more gradually after the second decay
to 0.001 at epoch 23. The initial result is strong enough for the next meeting
and confirms that the CIFAR-10 / ResNet-18 / SGD pipeline is functioning.

This is still not the final controlled study. The main experiment should use the
same configuration across schedulers and run at least three random seeds per
scheduler.

