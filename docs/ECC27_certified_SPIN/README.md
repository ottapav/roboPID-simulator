# ECC 2027 paper: Certified Robustness for Model-Free PID Tuning by Step-Response Inspection

Authors: P. Otta, D. Pachner, J. Dostál, V. Havlena. Extends the SPIN method (`../SPIN_poster`).

- `spin_ecc27.tex`: paper source (ieeeconf, 6 pages). `spin_ecc27_preview.pdf`: preview build
  (article-class shim of ieeeconf; the final layout needs the real `ieeeconf.cls`).
- `fig_stair.pdf`, `fig_record.pdf`, `fig_certified.pdf`: figures 1-3.
- `scripts/`: confirmation of every statement and all figures; see `scripts/README.md`.

Results. The paper separates two sources of robustness.

1. **Robustness from the limits** (Section II). Lemma 1: between two zeros of its first coordinate,
   a SPIN portrait turns by exactly half a turn, in any linear coordinates. Lag-cancelled loops
   k e^{-sL}/(sL): M_s strictly increasing in k, no crossing iff M_s <= 1.394 (Theorem 1);
   at small delta every count grows like ln(1/delta) * omega/(2 pi |sigma|) (Proposition 2). At
   delta = 0.02 the counts form a staircase; the published limits (0.5, 0.75, 1.0) allow 0, 1, 2
   crossings above delta, i.e. M_s <= 1.534, 1.842, 2.210 there. On general loops no limit bounds M_s.
2. **Robustness from the certificate** (Section III). Lemma 2: the error increments after a unit
   setpoint step are exactly the impulse response of S_d. Theorem 2: certified bounds on M_s from
   one finite record. Theorem 3: certified SPIN (acceptance iff the upper bound <= M*) accepts only
   M_s <= M*, whatever the limits, and is never trapped. Proposition 3: the same record predicts the
   sensitivity after any gain change.
3. **Simulations** (Section IV). Published SPIN exceeded M* = 1.7 on up to 9 of 19 loops, certified
   SPIN on none. Under the certificate the limit values set performance only: all limits 0.5 keep
   M_s <= M* but raise the load-step IAE (median +11/+16%); 0.75 and 1.0 for Gamma_1 act alike.
