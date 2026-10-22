# ECC 2027 paper: Certified Robustness from One Setpoint Step: A Safe Acceptance Test for Model-Free PID Tuning

Authors: P. Otta, D. Pachner, J. Dostál, V. Havlena. Extends the SPIN method (`../SPIN_poster`).

- `spin_ecc27.tex`: paper source (ieeeconf, 6 pages). `ieeeconf.cls`: the
  PaperPlaza class from the ECC author kit. `spin_ecc27.pdf`: build with it
  (`pdflatex` twice; 6 pages).
- `fig_record.pdf`, `fig_stair.pdf`, `fig_history.pdf`, `fig_response.pdf`: figures 1-4.
- `scripts/`: confirmation of every statement and all figures; see `scripts/README.md`.

Results.

1. **The certificate** (Section 2). Lemma 1: in a stable one-degree-of-freedom digital loop, the
   error increments after a unit setpoint step are exactly the impulse response of S. Theorem 1:
   one finite record gives certified lower and upper bounds on M_s, without a model, under one
   assumption on the unrecorded tail; the bounds converge to M_s.
2. **Certified acceptance and certify-before-applying** (Section 3). Theorem 2: any iterative tuner
   that accepts a tuning only if its certified upper bound is at most M* ends with M_s <= M*, and a
   uniform back-off always restores acceptance. Theorem 3: the same record certifies stability and
   bounds M_s of any proposed controller (same T_s, integral action) *before* it is applied, via
   S' = S / (1 + T Delta), Delta = C'/C - 1.
3. **Why turn counts cannot certify** (Section 4.2). Lemma 2: each setpoint crossing adds exactly
   half a turn to a SPIN portrait, in any linear coordinates. Proposition 1: on lag-cancelled loops
   k e^{-sL}/(sL), M_s increases with k and the error never crosses iff M_s <= 1.394. At delta = 0.02
   the published limits (0.5, 0.75, 1.0) allow 0, 1, 2 crossings, i.e. M_s <= 1.534, 1.842, 2.210 on
   these loops; on general loops crossings do not bound M_s.
4. **From certified to safe SPIN** (Section 4.3). Corollary 1: certified SPIN (checks each tuning after
   applying it) ends with M_s <= M*, whatever the limits, and is never trapped. Corollary 2: safe SPIN
   certifies every move before applying it, so from the first accepted tuning on, every applied tuning
   is stable with M_s <= M*.
5. **Simulations** (Section 5; one experiment: 19 loops, gain box, fixed T_s, two starts). Published SPIN
   ended above M* = 1.7 on 9 and 7 loops and applied a tuning above M* in 4371 of 7600 iterations (up
   to 2.38); certified SPIN ended below M* everywhere but applied such tunings in 3398 (up to 1.98);
   safe SPIN in none (at most 1.696). Mean load-step IAE vs published: certified +9%/+8%, safe
   +15%/+22% (medians: 0% and +4%/+2%). Under the certificate the limit values set performance only.
- `proposal.md`: follow-up proposal, certified robustness from noisy steps (numbers from `scripts/noise_exp.py`).
