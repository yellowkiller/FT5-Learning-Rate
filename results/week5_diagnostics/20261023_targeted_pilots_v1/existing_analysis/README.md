# Existing experiment audit

24 runs; 3600 epoch-start LR samples verified against unchanged defaults.

- per_seed_convergence.csv: first 80/90% validation crossing (blank = not reached), time, final gaps and last-10 variability.
- restart_windows.csv: five-epoch windows around restarts at epochs 38 and 112.
- cyclic: triangular2 amplitude halves each cycle; the nonzero LR floor remains.
- warmrestart: epoch 150 falls within a high-LR portion of the last cycle.
- Neither observation alone establishes causality. New paired diagnostics are required.
- Elapsed times are descriptive because historical hardware load was uncontrolled.
