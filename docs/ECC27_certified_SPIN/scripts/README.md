# Scripts for "Certified Robustness for Model-Free PID Tuning by Step-Response Inspection"

Needs Python 3 with numpy, scipy (>= 1.8, for `scipy.signal.czt`) and matplotlib. Run everything from
this folder. Every claim of the paper is a lemma, proposition or theorem; the scripts below
*confirm* them in simulation (Section V) and draw the figures. Simulation results are included, so
figures and tables rebuild without rerunning SPIN.

## Core

| file | content |
|---|---|
| `cert.py` | Theorem 1: certified lower/upper bound on M_s from one finite record (tail term, Lipschitz grid term, Abel high-frequency term); the exact digital loop `L_d`, `C_d`, `P_d` of the simulator; exact `Ms_d` |
| `spin_cert_run.py` | `sim`: the digital loop (PID with forward-Euler integral and filtered derivative, lags discretized exactly, delay rounded to samples) |
| `designed.py` | SPIN runner. Rule `original` = SPIN as published; rule `cert-shape` = Algorithm 2 (acceptance iff the certified upper bound <= M*, band move predicted by Prop. 3(d), at most `NMAX` in a row, else uniform back-off). `SPIN_BOX=1` turns on SPIN's gain box, `START_SCALE=2` the bold start |

## Paper item -> script

| paper item | command / data |
|---|---|
| Prop. 1, Prop. 2 (c),(d), Lemma 1, Cor. 1 on the lag-cancelled loop (Sec. V, "Result 1") | `python3 check_result1.py` |
| Fig. 1 | `python3 fig_stair.py` |
| Published SPIN with gain box, cautious / bold start | `box_*.jsonl`, `box2_*.jsonl` (`SPIN_BOX=1 RULES_ONLY=original OUT_PREFIX=box python3 designed.py 0 2`; add `START_SCALE=2`, `OUT_PREFIX=box2`) |
| Certified SPIN (Algorithm 2) with gain box, both starts | `cbox_0.jsonl`, `cbox2_0.jsonl` (`SPIN_BOX=1 RULES_ONLY=cert-shape OUT_PREFIX=cbox python3 designed.py 0 1`; add `START_SCALE=2`, `OUT_PREFIX=cbox2`) |
| Without the gain box | published: `designed_*.jsonl` (rule `original`); certified: `cnob_0.jsonl`, `cnob_1.jsonl` (`RULES_ONLY=cert-shape OUT_PREFIX=cnob python3 designed.py 0 1`; `run_rest.py` ran the remaining loops on a second core) |
| Lemma 2 and Theorem 1 on all 146 accepted tunings (true tail, identity, bracketing) | `python3 confirm_thm1.py` -> `thm1_points.json` |
| Theorem 2(ii) along the uniform back-off; Prop. 3(a),(d) | `python3 confirm_thm2.py` -> `thm2_points.json` |
| Table II, numbers of Section V, load-step IAE | `python3 analyze_cert.py` -> `summary_cert.json` |
| Fig. 2 | `python3 fig_record.py` |
| Fig. 3 | `python3 fig_certified.py` |
| Owner of the peak (Table II) | `owner_box.json` (`OWN_IN=box python3 owner_designed.py`) |

## Earlier material

`shape*_0.jsonl`, `shapehi*_0.jsonl`, `boxtest_0.jsonl`, `box2test_0.jsonl`, `analyze_shape*.py`,
`iae_*.json`, `record_points.json`, `conv.py`, `twodof.py`, `noise.py` belong to earlier drafts
(acceptance by the read value instead of the certified bound, continuous-design M_s). They are kept
for reference and are not used by the current paper.
