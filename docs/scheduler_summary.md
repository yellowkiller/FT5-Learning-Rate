# Learning-Rate Scheduler Summary

## Fixed LR

The learning rate stays constant throughout training. This is the simplest
control condition and makes it easier to see whether more complex schedules add
value. It may converge slowly or plateau if the LR is too high or too low.

Key values to log: `initial_lr`.

## Step Decay / MultiStepLR

The learning rate is reduced by a fixed multiplier at predefined epochs. This is
a traditional CNN baseline because it gives high LR for early progress and lower
LR for later refinement.

Key values to log: `initial_lr`, milestones, decay factor.

## Cosine Annealing

The learning rate follows a smooth cosine curve from the initial LR toward a low
final LR. It avoids abrupt drops and often gives stable late-stage convergence.

Key values to log: `initial_lr`, `T_max`, minimum LR if used.

## Cosine Annealing With Warm Restarts

The learning rate follows a cosine decay inside each cycle and then restarts to a
higher LR. Restarts can help the optimizer explore new regions and may improve
early or anytime performance.

Key values to log: `initial_lr`, `T_0`, `T_mult`, minimum LR.

## Cyclical LR

The learning rate repeatedly moves between a lower and upper bound. It is based
on the idea that increasing LR periodically can help optimization and reduce the
need for a manually chosen fixed schedule.

Key values to log: `base_lr`, `max_lr`, cycle length, mode.

## OneCycleLR

The learning rate increases to a maximum and then decreases over one main cycle.
It is connected to the super-convergence idea, where large learning rates can
speed training and act as regularization.

Key values to log: `max_lr`, total steps, pct_start, annealing strategy.

## Warm-Up + Cosine Decay

The learning rate starts very small, increases gradually during a warm-up
period, then follows cosine decay. Warm-up reduces early optimization shocks,
while cosine decay gives smooth late-stage convergence.

Key values to log: warm-up length, start factor, maximum LR, cosine length,
minimum LR.

## Controlled Comparison Rule

Only the scheduler should change in the main comparison. Keep dataset split,
architecture, optimizer, momentum, weight decay, batch size, augmentation,
epochs, and evaluation protocol fixed.

In this codebase, momentum is fixed at `0.9` and weight decay is fixed at
`5e-4`. These are controlled baseline constants, not scheduler-specific tuning
knobs.

