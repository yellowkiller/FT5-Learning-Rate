# Focused four-policy LR study

Isolated subprocesses; up to two concurrent jobs. All runs 150 epochs.
01_tuning: 3 initial candidates per policy, seed4; one conditional single-factor extension except cosine.
02_replication: top two complete configurations per policy, seed5.
03_frozen_configs: all choices fixed before test evaluation.
04_confirmation: seeds6/7/8, selected policies only, test set evaluated once per selected checkpoint.
No pooling across tuning/confirmation or historical versions. No GitHub upload.
Partial logs are never overwritten. status.json and queue logs report failure/completion.
Summary figures remain separated by policy and stage. See protocol/ for exact sources, seeds and settings.
