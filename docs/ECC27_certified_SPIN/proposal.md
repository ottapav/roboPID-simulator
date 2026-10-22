# Proposal: certified robustness from noisy setpoint steps

Follow-up to the ECC 2027 paper (`spin_ecc27.tex`). Status: idea and first numbers,
22 October 2026. The numbers below come from `scripts/noise_exp.py`.

## Question

The paper reads $M_s$ from one noise-free setpoint step. Can the same estimate,

$$\hat S(\theta)=\sum_{n=0}^{N}a_n\,\mathrm e^{-jn\theta},\qquad a_n=e_n-e_{n-1},$$

be extended to records with measurement noise? Short answer: yes, but the certificate
becomes probabilistic, and the parts of the current bound that use absolute variation
(the tail $\tau$, the slope constant $J$, the high-frequency constant $B$) must be
replaced, because noise inflates them linearly with the record length.

## Where the noise enters

With output measurement noise $v$ (or any output-equivalent disturbance), the recorded
error of a one-degree-of-freedom loop is

$$e = S\,(r - v).$$

The loop is linear, so the record splits into the step response and the loop's own
noise response $e^w=-S\,v$:

$$\hat S(\theta)=S_N(\theta)+D_N(\theta),\qquad
D_N(\theta)=\sum_{n=0}^{N}\big(e^w_n-e^w_{n-1}\big)\,\mathrm e^{-jn\theta}.$$

Two observations make this tractable:

1. **The error term is observable without a step.** $D_N$ is the Fourier sum of the
   differenced error of the same loop at rest. A quiet window before the step, with the
   same controller, has the same statistics, so the size of $D_N(\theta)$ can be
   estimated from data, without a model.
2. **Its size has a simple form.** For stationary noise with two-sided spectrum
   $\Phi_v(\omega)$, a step of height $A$ and a record of length $T_\mathrm{rec}$,
   the relative error near frequency $\omega$ is about

   $$\frac{|D_N|}{|S|}\approx\frac{\omega\sqrt{T_\mathrm{rec}\,\Phi_v(\omega)}}{A}.$$

   The sampling period drops out. The factor $\omega$ comes from the differencing and
   suppresses slow noise, which helps: $M_s$ sits at mid frequencies.

## First numbers (running example: $P_3$ with PID, basic tuning, $M_s=1.79$ at $T_s=0.02$ s)

Noise: first-order filtered, standard deviation 1 % of the step, filter time constant 0.5 s.
Forty noise realizations per row.

| record | rel. error at the peak, rms | predicted | $\max\lvert\hat S\rvert$, mean ± sd | worst of 40 |
|---|---|---|---|---|
| 225 s | 0.023 | 0.024 | 1.81 ± 0.022 | 1.86 |
| 100 s | 0.015 | 0.016 | 1.80 ± 0.019 | 1.84 |
| 225 s, average of 4 steps | | | sd 0.011 | |

- The formula predicts the error at the peak within a few per cent.
- At 1 % noise, $M_s$ is read to about ±0.02 (1 sd). That is of the same order as the
  margins in the paper's results (final $M_s$ of 1.66–1.69 against $M^*=1.7$), so the
  noise bound must be part of the test, not ignored.
- Noise biases $\max|\hat S|$ slightly upward (+0.018 at 225 s): the conservative
  direction for an upper bound, the wrong one for a lower bound.

**Why the app rejects noisy records today.** The tail heuristic `tau_for` (twice the
variation over the last 10 % of the record) rises from 0.001 to 5.1 with 1 % noise, and
the certified upper bound from 1.80 to 7.06. The variation $\sum|a_n|$ of the record
grows from 2.1 to 26.5: absolute variation measures the noise, not the loop.

## Proposed extension

1. **Separate the deterministic and the stochastic parts.** Keep Assumption 1 for the
   noise-free response (a tail $\tau$ of the step response itself, a design assumption
   as in the paper), and bound the noise term statistically instead of through
   $\sum|a_n|$.

2. **Probabilistic certificate.** Estimate $s(\theta)^2=\mathbb E|D_N(\theta)|^2$ from a
   quiet window (or from the settled end of previous records). With probability at
   least $1-\delta$,

   $$M_s\le\max_\theta\Big(|\hat S(\theta)|+\tau+c_\delta\,s(\theta)\Big),$$

   with $c_\delta$ from a Gaussian union bound over the roughly $N$ independent
   frequencies, $c_\delta\approx\sqrt{2\ln(2N/\delta)}$, or a tighter chaining bound.
   The lower bound follows in the same way.

3. **Grid without $J$ and $B$.** $\hat S$ is a trigonometric polynomial of degree $N$.
   Bernstein's inequality, $\max|\hat S'|\le N\max|\hat S|$, gives
   $\max_\theta|\hat S|\le\max_{\text{grid}}|\hat S|/(1-Nh/2)$ for grid spacing $h$.
   This holds for any record, noisy or not, and needs no total-variation constant;
   a chirp $z$-transform with 20–50 points per degree is cheap. For clean, fast-decaying
   records the current $J$-based margin is tighter, so both can be offered.

4. **Make the noise term small.**
   - Larger step height $A$: the error scales as $1/A$.
   - Averaging $K$ records: error $/\sqrt K$, confirmed above. One iteration could apply an
     up step and a down step, which also cancels slow drift.
   - Shorter records: the error grows as $\sqrt{T_\mathrm{rec}}$, the tail $\tau$ falls
     with $T_\mathrm{rec}$; choose the length that balances the two.
   - A tapered window $g_n=\rho^n$ evaluates $S$ on a circle of radius $1/\rho>1$, where
     noise is damped; its bias can be bounded in the same way as the tail.

5. **Certify before applying (Theorem 3) carries over.** Its proof only needs
   $|S-\hat S|$ bounded at each frequency. Replacing $\tau$ by the frequency-dependent
   $\tau+c_\delta s(\theta)$ in conditions (5)–(6) keeps the result, now with
   probability at least $1-\delta$.

6. **Safe SPIN over many iterations.** With failure probability $\delta$ per certificate,
   the union bound gives at most $n_\mathrm{iter}\delta$ for a whole run. Choosing
   $\delta=10^{-3}/n_\mathrm{iter}$ costs little, since $c_\delta$ grows only like
   $\sqrt{\ln(1/\delta)}$.

7. **What a noisy test should do when it cannot decide.** Today, safe and certified SPIN
   back off until the gain box. A noise-aware tuner should hold and average another step
   when $\max|\hat S|$ is within $c_\delta s$ of $M^*$, and back off only when the record
   clearly exceeds $M^*$.

## Experiments

1. Coverage: over many noise realizations and the 19 loops of the paper, check that the
   bound in item 2 holds with the claimed probability, for several noise levels and
   spectra.
2. Tuning: certified and safe SPIN with noise in the app's noise model ($\sigma$, filter
   $\tau$). Measure applied tunings above $M^*$, iterations needed, and the cost in
   load-step IAE against the noise-free results of Table III.
3. Up/down step pairs and averaging: how many extra steps per iteration buy how much margin.
4. Real loops (for example, UCEEB building HVAC), where noise, quantization and slow drift
   come together.

## Open questions

- **Non-stationary disturbances.** A load change during the step is not noise; it biases
  $\hat S$. Comparing the quiet windows before and after the step could detect it.
- **Estimating $s(\theta)$ online.** How long a quiet window is needed, and how to share
  it across iterations when the controller changes (the noise response $e^w=-Sv$ depends
  on the controller through $S$).
- **Quantization.** It is bounded and deterministic, but treating it as noise is probably
  the practical route.
- **The turn counts under noise.** SPIN's counts are themselves noise-sensitive (the
  $\delta$ guard of the counts). The certificate and the counts may need the same
  averaged record.
