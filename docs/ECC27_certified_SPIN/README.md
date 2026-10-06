# ECC 2027 paper: Certified Robustness for Model-Free PID Tuning by Step-Response Inspection

Authors: P. Otta, D. Pachner, J. Dostál, V. Havlena. Extends the SPIN method (`../SPIN_poster`).

- `spin_ecc27.tex`: paper source (ieeeconf, 6 pages). `spin_ecc27_preview.pdf`: preview build
  (article-class shim of ieeeconf; final layout needs the real `ieeeconf.cls`).
- `fig_stair.pdf`, `fig_record.pdf`, `fig_certified.pdf`: figures 1-3.
- `scripts/`: every number and figure; see `scripts/README.md` for the mapping and run commands.
  Simulation results (`*.jsonl`) are included, so figures and tables rebuild without rerunning.

Main results: (1) on lag-cancelled loops a turn count grows by exactly half a turn per setpoint
crossing, no crossing iff k <= 1/e (M_s <= 1.39); (2) for any stable loop the setpoint-step error
is the step response of S, which gives M_s, the loop L = 1/S - 1 and an exact prediction of M_s
for any other tuning. Algorithm 2 uses this as an acceptance test and a band-shaping rule inside SPIN.
