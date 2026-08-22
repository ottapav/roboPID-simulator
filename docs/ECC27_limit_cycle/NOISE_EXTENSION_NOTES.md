# SPIN limit cycles under output noise — research notes

Findings from the July 2026 discussion, recorded for the journal follow-up.
**Not in the ECC paper**, which is deliberately deterministic (see the
assumption clause now in its Sec. 2). Nothing below has been verified by
simulation; the two combinatorial results are exact, everything else is
derivation or scaling and needs a numerical pass.

Status legend: **[exact]** proved by direct computation · **[derived]**
analytic but unchecked · **[scaling]** order-of-magnitude · **[open]** needs
work.

---

## 0. Noise model

Use the first-order shaping filter

$$ n(s) \;=\; \frac{\sigma\sqrt{2\tau}}{\tau s + 1}\, w(s), \qquad w = \text{unit white noise} $$

i.e. an OU process with $R(t) = \sigma^2 e^{-|t|/\tau}$. **[exact]** The
$\sqrt{2\tau}$ normalization gives $\mathrm{Var}[n] = \sigma^2$ independent of
$\tau$, so amplitude and correlation time are orthogonal knobs. Numerically
confirmed (std, autocorrelation at lag $\tau$, and the averaging law below).

Why this model rather than plain white noise:

1. **Orthogonal sweeps.** $\sigma$ and $\tau$ can be varied independently in
   both the analysis and the battery.
2. **Rational spectrum.** Closed loop + shaping filter is one finite-dimensional
   LTI system driven by white noise, so portrait mean and covariance follow
   from Lyapunov recursions — bias and variance of $N_\varepsilon$ become
   semi-analytic instead of Monte Carlo. Probably the strongest argument.
3. **Two physical limits** (see §0a) that bracket the behaviour and give
   sanity checks.

**[exact]** Averaging law: the mean of $n$ over a window $W$ has variance
$\sigma^2 \cdot 2\tau / W$. Hence
$$ \sigma_N \;\sim\; \sigma\sqrt{2\tau/T}, \qquad
   \sigma_\theta \;=\; \frac{\sigma}{\lVert J\rVert}\sqrt{\frac{2\tau}{T}} $$
which supplies the parameter missing from §3a below.

### 0a. $\tau/T_s$ is the group that decides unbiasedness

**[derived] — this supersedes the framing of §1a.** With $T_s$ the settling
time of the step response:

| regime | behaviour | consequence |
|---|---|---|
| $\tau \ll T_s$ | noise averages out within one experiment; band comparisons effectively independent | fair-coin argument of §1 holds, $\theta^*$ **unbiased** |
| $\tau \gg T_s$ | noise acts as a random DC offset for the whole experiment | displaces the origin of *every* portrait coherently; winding about a displaced origin differs systematically, so all bands shift together — independence fails and a genuine **bias** appears |

So the band-correlation caveat of §1a and the slow-noise limit are the *same
phenomenon*, with $\tau/T_s$ as its single knob. This is far more tractable
than computing correlations from the band filters, and it makes the §1
unbiasedness claim precise: it is a statement about the fast-noise regime.

**[derived] Design condition.** In the slow-noise limit the origin offset is
$O(\sigma)$, while the regularized index weights radius $\rho$ by
$\rho^2/(\rho^2+\varepsilon^2)$. An offset well below $\varepsilon$ is
suppressed; one well above lets the settled tail contribute spurious winding.
Hence

$$ \boxed{\;\sigma \;\lesssim\; \varepsilon\;} $$

The $\varepsilon$ introduced for well-posedness doubles as the noise
tolerance — a satisfying closure of the regularization theme.

**[open] Caveat the model does not fix.** An OU process is not
differentiable, so if a portrait axis is a derivative its noise variance is
unbounded. The band filters must close that channel, which means the
follow-up has to specify them explicitly; the noise model alone does not make
the derivative channel well posed. (See §5.)

---

## 1. The central result: noise does not move the target

**[exact]** Under heavy noise at $\theta^*$ all three comparisons
$N_k \gtrless \bar N_k$ degrade to fair coins. Feeding fair coins through the
priority logic ("cut the lowest failing band, else expand") gives

| branch | probability | move |
|---|---|---|
| band 0 fails | 1/2 | $d_0$ |
| band 0 ok, band 1 fails | 1/4 | $d_1$ |
| bands 0,1 ok, band 2 fails | 1/8 | $d_2$ |
| all pass | 1/8 | $e$ |

i.e. $(w_e, w_{d_0}, w_{d_1}, w_{d_2}) = (\tfrac18, \tfrac12, \tfrac14,
\tfrac18)$ — **exactly the deterministic balance weights** of Prop. 1.
Verified for $n = 1, 2, 3$ (script: the `move_from_signs` enumeration used in
the discussion; trivial to re-derive).

Not a coincidence: the priority logic is a binary decision tree and the
deterministic orbit is a ripple-carry odometer. Same combinatorial object,
two readings.

**Consequence.** Because the fair-coin frequencies *are* the balance weights,
and the balance weights give zero net displacement (Prop. 1), the mean drift
at $\theta^*$ under fully noise-dominated switching is **exactly zero**. Away
from $\theta^*$ the signs regain reliability and restore. So:

> the deterministic cube inflates into a noise ball, but stays centred on the
> same plant-determined $\theta^*$.

This is the headline for the follow-up: the design rule's *centre* is
noise-invariant; only its radius changes. Much stronger claim than
"noise degrades the method."

### 1a. Scope of the claim

The fair-coin argument assumes the three comparisons carry *independent*
noise. Under the model of §0 this holds in the fast-noise regime
$\tau \ll T_s$ and fails for $\tau \gg T_s$, where a coherent origin offset
biases all bands together — see §0a, which replaces the vaguer
"compute the band correlation" item recorded earlier. State §1 as a
fast-noise result and quantify the crossover numerically (experiment 2).

---

## 2. Correction to an earlier claim — the duty cycle is NOT a noise detector

Earlier in the discussion I suggested that duty-cycle drift away from the
ruler weights toward uniform $(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$ would
flag the noise floor. **This is wrong** — §1 shows the noise-dominated duty
cycle *equals* the ruler weights, so the frequency statistic is blind to the
distinction. Do not build a detector on it.

**[derived]** The right statistic is the **ordering**, not the frequencies.
Deterministic cycle: rigid periodic sequence
$d_0\, d_1\, d_0\, d_2\, d_0\, d_1\, d_0\, e$. Noise-driven process:
identical marginals, i.i.d. ordering. So test periodicity —
autocorrelation of the move sequence at lag 8, or simply whether $d_0$
occupies every odd position. Still needs no new instrumentation.

Three regimes for the unified accelerate/decelerate logic, revised:

| observation | regime | action |
|---|---|---|
| few reversals, long same-move runs | travelling | double $\beta$ |
| reversals + **periodic** ordering | on the cycle | halve $\beta$ |
| reversals + **i.i.d.** ordering | noise floor | stop |

---

## 3. Scaling of the inflated cycle

**[scaling]** Two length scales in gain space:

- $h = \log(1+\beta)$ — deterministic cube side
- $\sigma_\theta = \sigma_N / \lVert J \rVert$ — width of the noise-blurred
  switching layer ($J$ = local Jacobian $\partial N_j/\partial\theta_i$,
  nonsingular but non-diagonal per the ECC paper's Sec. 7)

Balancing per-step diffusion $\sim h^2$ against restoring rate
$\sim h/\sigma_\theta$ gives stationary spread $\sim \sqrt{h\,\sigma_\theta}$.
Terminal error therefore

$$ \text{err} \;\approx\; \max\!\left(\tfrac{\sqrt3}{2} h,\; C\sqrt{h\,\sigma_\theta}\right) $$

so **Corollary 3 (Design Rule) acquires a noise term**. Note both terms
increase with $h$, so there is no accuracy floor in $h$ — smaller steps are
still better; what degrades is time, not achievable accuracy.

### 3a. Plant-time budget — the practically interesting form

**[derived]** Since $\sigma_N \sim \sigma\sqrt{2\tau/T}$ (§0) for experiment length $T$, reaching gain accuracy $\delta$ needs
$T \sim (\sigma_v / \lVert J \rVert \delta)^2$, while the iteration count
stays $\log(D/\delta)$ from the accelerated schedule. Total plant time:

$$ T \;\sim\; \frac{2\tau\,\sigma^2}{(\lVert J\rVert \delta)^2},
\qquad
T_{\text{total}} \;\sim\; \frac{2\tau\,\sigma^2}{(\lVert J\rVert\delta)^2}\,
\log(D/\delta) $$

Correlation time enters **linearly**: slow noise is proportionally more
expensive than fast noise of the same variance.

The log law survives in the iteration count; noise enters as a
multiplicative $1/\delta^{-2}$ on experiment length. Gives a real design
trade-off — **more iterations vs. longer experiments per iteration** — with a
clean answer. Plant time is the actual currency on a plant floor, so this is
likely the most quotable practical result of the follow-up.

---

## 4. Why noise may make the proof *easier*

**[derived, promising]** Noise smooths the switching law. The discontinuous
Filippov field is replaced by a genuinely smooth averaged field

$$ F(s) \;=\; h \sum_v P(v \mid s)\, v, \qquad s = \theta - \theta^* $$

with $P(v\mid s)$ computed from the noise distribution through the priority
tree. Standard smooth-ODE stability arguments then apply, and the discrete
iteration follows by averaging/perturbation.

So the §7 open problem of the ECC paper may be **more tractable in the noisy
case than the deterministic one**. Appealing paper structure: the
regularization theme runs twice — geometric ($\varepsilon$ on the index) and
stochastic ($\sigma$ on the switching) — and both make the analysis well
posed.

---

## 5. Practical issues specific to filtered white noise

**[open]** *Derivative amplification.* If a portrait axis is a derivative,
OU noise hits it hard (the derivative of an OU process is not classically
defined; for band-limited noise the variance is large and grows with
sampling rate). The band filtering in the SPIN construction is already the
main defence — worth stating explicitly, since it means the method is
better placed here than a raw $(e, \dot e)$ portrait would be.

**[open]** *The $\varepsilon$–$\sigma$ trade-off.* $\varepsilon$ acquires a
second role under noise: it damps exactly the near-origin region where
noise-induced spurious winding concentrates. Larger $\varepsilon$ suppresses
noise bias but blunts discrimination. There should be an optimum
$\varepsilon(\sigma)$ — tractable analytically for filtered white noise, and
a good figure (bias and variance vs. $\varepsilon$, crossing).

**[derived]** *Why SPIN is structurally well placed.* IFT and extremum
seeking need gradient *magnitudes*; the triangular rule needs only three
*signs*, and a sign survives noise whenever $|N_k - \bar N_k|$ exceeds it.
Worth a sentence in the follow-up's introduction contrast.

---

## 6. Suggested experiments (none run yet)

1. Lyapunov computation of portrait mean/covariance under the §0 model —
   analytic, do first; it feeds every other item.
2. Noisy battery: P1/P3/P4 sweeping $\sigma$ and $\tau$ **independently** at
   constant $\beta$; check (a) the centre stays at $\theta^*$ for
   $\tau \ll T_s$ and drifts for $\tau \gg T_s$ (the §0a crossover),
   (b) spread scales as $\sqrt{h\sigma_\theta}$.
3. Periodicity detector of §2: measure lag-8 autocorrelation of the move
   sequence across the noise levels; confirm it separates the regimes where
   the duty cycle does not.
4. $\varepsilon$ sweep at fixed $\sigma$, in the slow-noise regime: locate the
   bias/variance optimum and test the $\sigma \lesssim \varepsilon$ condition of §0a.
5. Plant-time budget of §3a: verify the $\tau\sigma^2 \delta^{-2}\log(D/\delta)$
   scaling by sweeping the iterations-vs-$T$ allocation at fixed total time.

---

## 7. Relation to the ECC paper

The ECC submission stays deterministic. Its Sec. 2 now carries an explicit
noise-free assumption naming the follow-up, so nothing here needs to be
retrofitted and no claim in the ECC paper is contradicted by §1 above (the
zero-noise limit is consistent).

Follow-up paper spine, tentatively: Filippov proof of the deterministic
cycle (the ECC open problem) **+** the noise extension above, with the two
regularizations as the unifying theme.
