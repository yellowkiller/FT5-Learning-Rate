# Conditional LR follow-up

Validated runs: 19 / at most 19.

## Additional validation evidence

Cyclic paired validation seeds4/5/9: {'cyclic_floor0.01': [0.9312, 0.9262, 0.9276], 'cyclic_floor0.01_extension': [0.929, 0.931, 0.9312]}. Report direction and spread; no parent configuration is changed.
OneCycle incumbent comparison, validation seeds4/5: {'onecycle_lr0.1': [0.9548, 0.957], 'onecycle_lr0.03': [0.956, 0.9546]}. No test scores used for this choice.

## New-policy configurations

- step: {'id': 'step_lr0.1', 'scheduler': 'step', 'lr': 0.1, 'options': {}}; uncertain=True
- multistep: {'id': 'multistep_lr0.1', 'scheduler': 'multistep', 'lr': 0.1, 'options': {}}; uncertain=True

## Scope and next steps

Follow-up evidence does not rewrite the parent campaign or its frozen choices.
Uncertain differences remain uncertain; no unbounded search is triggered.
Compare all six policies using paired seeds6/7/8; do not pool tuning results.
The official test set has been inspected previously, so it is not a pristine untouched benchmark.
Wall times from shared GPU jobs are descriptive. Next evaluate controlled serial timing and one secondary workload.