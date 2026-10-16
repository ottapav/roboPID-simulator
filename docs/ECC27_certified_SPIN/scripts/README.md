# Scripts for "Certified Robustness from One Setpoint Step: A Safe Acceptance Test for Model-Free PID Tuning"

Needs Python 3 with numpy, scipy (>= 1.8, for `scipy.signal.czt`) and matplotlib. Run everything from
this folder. All counts are SPIN's own (Definition 1 of SPIN: each axis divided by its peak; `compare.counts_old`).
Every claim of the paper is a lemma, proposition or theorem; the scripts below
*confirm* them in simulation (Section IV) and draw the figures. Simulation results are included, so
figures and tables rebuild without rerunning SPIN.

## Core

| file | content |
|---|---|
| `cert.py` | Theorem 1: certified lower/upper bound on M_s from one finite record (tail term, Lipschitz grid term, Abel high-frequency term); the exact digital loop `L_d`, `C_d`, `P_d` of the simulator; exact `Ms_d` |
| `cert_ahead.py` | Theorem 3: certify a proposed controller before applying it, from the record of the running one (stability condition (4) and bound (5) on a grid with Lipschitz margins, Abel bound above the grid) |
| `spin_cert_run.py` | `sim`: the digital loop (PID with forward-Euler integral and filtered derivative, lags discretized exactly, delay rounded to samples) |
| `designed.py` | SPIN runner. Rule `original` = SPIN as published; rule `cert-shape` = certified SPIN, Algorithm 1 with lines 7a-7c (acceptance iff the certified upper bound <= M*, band move predicted by Prop. 2, at most `NMAX` in a row, else uniform back-off). rule `cert-ahead` = safe SPIN (Corollary 2: every move certified by Theorem 3 before it is applied; else the next band move or a uniform back-off; else hold). `SPIN_BOX=1` turns on SPIN's gain box, `START_SCALE=2` the bold start, `CERT_LIMITS` the count limits of `cert-shape`/`cert-ahead` (default 0.5,0.75,1.0), `FIXED_TS=1` keeps the sampling period of the start tuning. Every run logs the exact M_s of the tuning applied in each iteration (`traj`) |

## Paper item -> script

All simulation results of the paper come from one experiment: published, certified and safe SPIN on the
19 loops, gain box, fixed sampling period (`FIXED_TS=1`), cautious and bold start.

| paper item | command / data |
|---|---|
| The experiment (Table 1, Sec. 5, Fig. 3) | `bash run_safe.sh` -> `safe1_*.jsonl` (cautious), `safe2_*.jsonl` (bold); `python3 analyze_safe.py` -> `summary_safe.json` (Table 1 incl. totals, applied tunings above M*, IAE) |
| Fig. 3 | `python3 fig_certified.py` (panel (a) from `safe1_*.jsonl`; panel (b) from `traj_P3PID.json`, `python3 traj_P3PID.py`) |
| Sec. 5 "The certificate": Lemma 1 and Theorem 1 on all 276 accepted tunings (true tail, identity, bracketing) | `RUNS=safe1,safe2 OUT=thm1_safe.json python3 confirm_thm1.py` |
| Theorem 3 on 456 random moves from the accepted tunings of safe SPIN | `python3 confirm_thm3.py` -> `thm3_points.json` |
| Limit values under the certificate (every limit 1/2) | `bash run_limits_fixed.sh` -> `limf1_*.jsonl`, `limf2_*.jsonl`; `python3 analyze_limits_fixed.py` -> `limits_fixed.json` |
| Sec. 4.2: SPIN's counts on the lag-cancelled loop (steps at 1.534, 1.842, 2.210) and the 3800 count decisions of certified SPIN | `FIXED_TS=1 python3 confirm_limits.py` -> `limits.json` (run by `run_limits_fixed.sh`) |
| Lemma 2 and Prop. 1 against a direct search | `python3 check_result1.py` |
| Fig. 1, Fig. 2 | `python3 fig_record.py`, `python3 fig_stair.py` |

## Earlier material

Runs with a sampling period that changes with the gains (earlier drafts): `box_*`, `box2_*`, `cbox*_0`,
`designed_*`, `cnob_*` (published/certified SPIN with and without the gain box), `lim101*`, `lim050*`
(`run_limits.sh`, `analyze_limits.py`, `limits_runs.json`), `analyze_cert.py` / `summary_cert.json`,
`confirm_thm2.py` / `thm2_points.json`, `thm1_points.json`, `owner_box.json`.
`shape*_0.jsonl`, `shapehi*_0.jsonl`, `boxtest_0.jsonl`, `box2test_0.jsonl`, `analyze_shape*.py`,
`iae_*.json`, `record_points.json`, `confirm_counts_delta.py` / `counts_delta.json` (counts at small delta), `bounds_vs_length.py` / `bounds_vs_length.json` (bounds versus record length), `conv.py`, `twodof.py`, `noise.py` belong to earlier drafts
(acceptance by the read value instead of the certified bound, continuous-design M_s). They are kept
for reference and are not used by the current paper.
