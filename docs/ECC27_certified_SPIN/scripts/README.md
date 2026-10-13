# Scripts for "Certified Robustness for Model-Free PID Tuning by Step-Response Inspection"

Needs Python 3 with numpy, scipy (>= 1.8, for `scipy.signal.czt`) and matplotlib. Run everything from
this folder. All counts are SPIN's own (Definition 1 of SPIN: each axis divided by its peak; `compare.counts_old`).
Every claim of the paper is a lemma, proposition or theorem; the scripts below
*confirm* them in simulation (Section IV) and draw the figures. Simulation results are included, so
figures and tables rebuild without rerunning SPIN.

## Core

| file | content |
|---|---|
| `cert.py` | Theorem 1: certified lower/upper bound on M_s from one finite record (tail term, Lipschitz grid term, Abel high-frequency term); the exact digital loop `L_d`, `C_d`, `P_d` of the simulator; exact `Ms_d` |
| `spin_cert_run.py` | `sim`: the digital loop (PID with forward-Euler integral and filtered derivative, lags discretized exactly, delay rounded to samples) |
| `designed.py` | SPIN runner. Rule `original` = SPIN as published; rule `cert-shape` = certified SPIN, Algorithm 1 with lines 7a-7c (acceptance iff the certified upper bound <= M*, band move predicted by Prop. 2, at most `NMAX` in a row, else uniform back-off). `SPIN_BOX=1` turns on SPIN's gain box, `START_SCALE=2` the bold start, `CERT_LIMITS` the count limits of `cert-shape` (default 0.5,0.75,1.0) |

## Paper item -> script

| paper item | command / data |
|---|---|
| Lemma 2 (half turns) and Prop. 1 (eq. (5) against a direct search) on the lag-cancelled loop (Sec. IV, "The counts") | `python3 check_result1.py` |
| What the published limits allow (Sec. III-B: SPIN's per-axis counts at delta = 0.02 on the lag-cancelled loop, steps at 1.534, 1.842, 2.210; the 3800 count decisions of the runs) | `python3 confirm_limits.py` -> `limits.json` |
| Limit values under the certificate (Sec. IV): certified SPIN with limits (0.5, 1.0, 1.0) and (0.5, 0.5, 0.5), both starts | `bash run_limits.sh` (`designed.py` with `CERT_LIMITS`) -> `lim101*_0.jsonl`, `lim050*_0.jsonl` (`run_rest.py` re-ran the bold-start PI loops on a second core: `lim050b_rest.jsonl`, identical results); then `python3 analyze_limits.py` -> `limits_runs.json` |
| Fig. 2 | `python3 fig_stair.py` |
| Published SPIN with gain box, cautious / bold start | `box_*.jsonl`, `box2_*.jsonl` (`SPIN_BOX=1 RULES_ONLY=original OUT_PREFIX=box python3 designed.py 0 2`; add `START_SCALE=2`, `OUT_PREFIX=box2`) |
| Certified SPIN (lines 7a-7c) with gain box, both starts | `cbox_0.jsonl`, `cbox2_0.jsonl` (`SPIN_BOX=1 RULES_ONLY=cert-shape OUT_PREFIX=cbox python3 designed.py 0 1`; add `START_SCALE=2`, `OUT_PREFIX=cbox2`) |
| Without the gain box | published: `designed_*.jsonl` (rule `original`); certified: `cnob_0.jsonl`, `cnob_1.jsonl` (`RULES_ONLY=cert-shape OUT_PREFIX=cnob python3 designed.py 0 1`; `run_rest.py` ran the remaining loops on a second core) |
| Lemma 1 and Theorem 1 on all 146 accepted tunings (true tail, identity, bracketing) | `python3 confirm_thm1.py` -> `thm1_points.json` |
| Theorem 2(ii) along the uniform back-off; Prop. 2 | `python3 confirm_thm2.py` -> `thm2_points.json` |
| Table I, numbers of Section IV, load-step IAE | `python3 analyze_cert.py` -> `summary_cert.json` |
| Fig. 1 | `python3 bounds_vs_length.py` (panel b data), then `python3 fig_record.py` |
| Fig. 3 | `python3 fig_certified.py` |
| Owner of the peak | `owner_box.json` (`OWN_IN=box python3 owner_designed.py`) |

## Earlier material

`shape*_0.jsonl`, `shapehi*_0.jsonl`, `boxtest_0.jsonl`, `box2test_0.jsonl`, `analyze_shape*.py`,
`iae_*.json`, `record_points.json`, `confirm_counts_delta.py` / `counts_delta.json` (counts at small delta), `conv.py`, `twodof.py`, `noise.py` belong to earlier drafts
(acceptance by the read value instead of the certified bound, continuous-design M_s). They are kept
for reference and are not used by the current paper.
