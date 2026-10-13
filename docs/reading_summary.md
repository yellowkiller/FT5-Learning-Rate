# One-Page Reading Summary

## Cyclical Learning Rates

Smith proposes cyclical learning rates (CLR): instead of using a fixed learning
rate or only decreasing it over time, the learning rate cycles between a lower
and upper bound. The paper argues that this can reduce manual LR tuning and may
reach good accuracy in fewer iterations. The practical idea most relevant to
this project is the LR range test: increase the LR over a short run to identify
reasonable minimum and maximum LR values before running the full experiment.

Project implication: use CLR as more than just another scheduler. Record the
chosen `base_lr`, `max_lr`, cycle length, and how those values were selected.

## SGDR / Cosine Annealing With Warm Restarts

Loshchilov and Hutter introduce SGDR, which combines cosine annealing with warm
restarts. The learning rate follows a cosine decay within a cycle, then jumps
back up at restart points. The intended benefit is stronger anytime performance:
the model may reach useful validation accuracy earlier, not only at the end of a
fixed training budget.

Project implication: compare ordinary cosine annealing and cosine warm restarts
separately. Track both final accuracy and convergence speed because SGDR's value
may appear before the final epoch.

## OneCycle / Super-Convergence

Smith and Topin describe super-convergence: in some settings, one LR cycle with
a large maximum learning rate can train much faster than a standard schedule.
The key claim is that large learning rates can act as regularization, so other
regularization settings may need careful control.

Project implication: OneCycleLR should be evaluated with an LR range test and
with careful reporting of `max_lr`. Since the project wants controlled
comparison, avoid changing weight decay or augmentation only for OneCycle unless
that is explicitly treated as a separate ablation.

## Large Minibatch Warm-Up

Goyal et al. study large-minibatch ImageNet training and show that warm-up helps
avoid early optimization difficulty when using high learning rates or scaled
batch sizes. The main transferable idea is gradual LR increase at the start of
training before applying the main schedule.

Project implication: warm-up plus cosine decay is a sensible extension after the
baseline schedulers work. Warm-up duration and initial LR should be logged.

## Meeting Takeaway

The project should not ask only which scheduler gets the highest final accuracy.
It should compare learning dynamics: speed to useful accuracy, stability across
seeds, validation/test accuracy, and training cost under the same model,
optimizer, augmentation, and budget.

