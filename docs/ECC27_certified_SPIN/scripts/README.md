# Certified Robustness for Model-Free PID Tuning by Step-Response Inspection

Code and results for the ECC 2027 paper by P. Otta, D. Pachner, J. Dostál and V. Havlena.
Every number and figure in the paper is reproduced by the scripts below.
Needs Python 3 with numpy, scipy and matplotlib. Run everything from this folder.

## Result 1: lag-cancelled loops (Section 2)

| paper item | script |
|---|---|
| Prop. 1, Prop. 2, Table I | `python3 reproduce.py` |
| Prop. 2(a) no crossing iff k ≤ 1/e (M_s ≤ 1.39), error of Prop. 2(c) | `lagcancel.py` (exact record of de/dt = -k e(t-1)) |
| Fig. 1 (staircase, record, portrait) | `python3 fig_stair.py` |
| Per-axis counts of SPIN's own limits | `lagcancel.py` + `compare.counts_old` |
| 630-loop grid, slow-integral loop | `python3 reproduce.py --grid`, `fig_record.py` |

## Result 2: the certificate and certified SPIN (Sections 3–4)

| paper item | script / data |
|---|---|
| Lemma 1 exactness, sampling convergence | `python3 conv.py` |
| Two-degree-of-freedom form (Corollary 1) | `python3 twodof.py` |
| Noise paragraph | `python3 noise.py` |
| Fig. 2 (122 readings) | `python3 fig_record.py` (cache: `record_points.json`) |
| Original SPIN as published (gain box), cautious start | `SPIN_BOX=1 RULES_ONLY=original OUT_PREFIX=box python3 designed.py 0 2` and `... 1 2` → `box_*.jsonl` |
| Same, bold start | add `START_SCALE=2`, `OUT_PREFIX=box2` → `box2_*.jsonl` |
| Certified SPIN (published SPIN + acceptance test + band shaping, Algorithm 2), both starts | `SPIN_BOX=1 RULES_ONLY=original+shape-hi OUT_PREFIX=shapehi python3 designed.py 0 1` (and `START_SCALE=2 OUT_PREFIX=shapehi2`) → `shapehi*_0.jsonl` |
| Other rules compared (elasticity band choice `original+test`, `shape`, uniform back-off `unif*`, `msbands*`) | same runner, `RULES_ONLY=<rule>`; tables by `python3 analyze_shape2.py`, shaping statistics by `analyze_shape.py` |
| Runs without the gain box, ablation | `RULES_ONLY=original,original+shape-hi OUT_PREFIX=nbshape python3 designed.py 0 2` / `1 2` → `nbshape_*.jsonl` (older rules: `designed_*.jsonl`, `ablation_*.jsonl`) |
| Owners of the peaks (Table II, Fig. 3a) | `OWN_IN=box python3 owner_designed.py` → `owner_box.json` |
| Load-step IAE (Table II, "The price") | `python3 analyze_shape2.py` (uses `iae_fine.load_iae`) |
| Fig. 3 | `python3 fig_certified.py` |
| Plant that does not converge without the box | `diag16.py` |

The SPIN runs take about 20–40 minutes per rule on two cores; their results are included,
so the figures and tables can be rebuilt without rerunning them.

## Notes

- `designed.py` simulates the controller finely (PID: T_f/20, PI: 1/1000 of the plant time scale),
  so that the running loop matches its continuous design.
- SPIN's gain box [0.001, 10] × start is off by default in `designed.py`; `SPIN_BOX=1` turns it on.
- `spin_run.py`, `spin_cert_run.py`, `compare.py`, `battery.py`, `spin_cert.py`, `ex2.py`,
  `exactlaw.py` are helper modules.

- Proposition 3 (band shaping): `designed.py` function `shape_move` implements the gradient, the exact prediction and Algorithm 2.
