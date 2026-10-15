# ECC 2027 paper: Certified Robustness from One Setpoint Step: A Safe Acceptance Test for Model-Free PID Tuning

Authors: P. Otta, D. Pachner, J. Dostál, V. Havlena. Extends the SPIN method (`../SPIN_poster`).

- `spin_ecc27.tex`: paper source (ieeeconf, 6 pages). `spin_ecc27_preview.pdf`: preview build
  (article-class shim of ieeeconf; the final layout needs the real `ieeeconf.cls`).
- `fig_record.pdf`, `fig_stair.pdf`, `fig_certified.pdf`: figures 1-3.
- `scripts/`: confirmation of every statement and all figures; see `scripts/README.md`.

Results.

1. **The certificate** (Section II). Lemma 1: in a stable one-degree-of-freedom digital loop, the
   error increments after a unit setpoint step are exactly the impulse response of S. Theorem 1:
   one finite record gives certified lower and upper bounds on M_s, without a model, under one
   assumption on the unrecorded tail; the bounds converge to M_s.
2. **Why turn counts cannot certify** (Section III-B). Lemma 2: each setpoint crossing adds exactly
   half a turn to a SPIN portrait, in any linear coordinates. Proposition 1: on lag-cancelled loops
   k e^{-sL}/(sL), M_s increases with k and the error never crosses iff M_s <= 1.394. At delta = 0.02
   the published limits (0.5, 0.75, 1.0) allow 0, 1, 2 crossings, i.e. M_s <= 1.534, 1.842, 2.210 on
   these loops; on general loops crossings do not bound M_s.
3. **Certified SPIN** (Section III-C). Theorem 2: accepting a tuning only if its certified upper bound
   is at most M* guarantees M_s <= M*, whatever the limits, and never traps SPIN. Proposition 2: the
   record predicts the sensitivity after any gain change.
4. **Simulations** (Section IV). Published SPIN exceeded M* = 1.7 on up to 9 of 19 loops, certified
   SPIN on none (load-step IAE +7.8%). Under the certificate the limit values set performance only.
