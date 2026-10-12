# ECC 2027 paper: Certified Robustness for Model-Free PID Tuning by Step-Response Inspection

Authors: P. Otta, D. Pachner, J. Dostál, V. Havlena. Extends the SPIN method (`../SPIN_poster`).

- `spin_ecc27.tex`: paper source (ieeeconf, 6 pages). `spin_ecc27_preview.pdf`: preview build
  (article-class shim of ieeeconf; the final layout needs the real `ieeeconf.cls`).
- `fig_stair.pdf`, `fig_record.pdf`, `fig_certified.pdf`: figures 1-3.
- `scripts/`: confirmation of every statement and all figures; see `scripts/README.md`.

Results (all proved; simulation only confirms):

1. **What a count means.** Lemma 1: between two zeros of its first coordinate, a SPIN portrait
   turns by exactly half a turn, in any linear coordinates. Corollary 1: the count of Gamma_0 is
   within half a turn of half the number of setpoint crossings when the error lobes shrink.
   Lag-cancelled loops k e^{-sL}/(sL): M_s strictly increasing in k (parametric closed form),
   no crossing iff k <= 1/e (M_s <= 1.394), extrema decay by r(k) < 1, lobes shrink for k < 3/2.
2. **The step certifies M_s.** Lemma 2: for a digital loop the error increments after a unit
   setpoint step are exactly the impulse response of S_d. Theorem 1: certified lower and upper
   bounds on M_s from one finite record, under one assumption (tail variation <= tau).
   Theorem 2: as SPIN's acceptance test, every accepted tuning has M_s <= M*; the uniform
   back-off is stable with M_s <= 1/(1 - lambda|m|). Proposition 3: the same record predicts
   exactly the sensitivity after any gain change, which selects the band to lower.
