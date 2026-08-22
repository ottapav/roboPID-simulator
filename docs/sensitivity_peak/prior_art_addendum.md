# Addendum to the Prior-Art Assessment

*Written after reading the full LaTeX source of the parent paper,
Pachner, Otta, Dostál & Havlena, "Model-free PID tuning by step-response
inspection". This resolves the caveat in the original assessment that the
paper under review could not be accessed.*

---

## 1. Verdict: the novelty claim strengthens

The parent paper **never mentions the sensitivity function, Ms, robustness, or
stability margins anywhere in its text.** A search of the full source for
"sensitivity", "Ms", "robustness" and "margin" returns only:

- one passing phrase about Ziegler–Nichols having been "revisited with modern
  robustness tools" (introduction, referring to Åström–Hägglund 2004);
- the stability-screen margin constant `c = 2` (Definition 3, Table 3);
- "margin" in the sense of gain headroom inside the tuning rule's proofs
  ("the margin to the first touch of −1", Proposition 6).

The frequency-domain reading of the turn index is therefore **not anticipated
by the authors' own prior work.** The strongest possible form of
self-pre-emption is ruled out.

The parent's conclusion lists its intended follow-up work explicitly: the
asymptotics of the fixed-step iteration, a diminishing step size, smoothing of
the index for that analysis, and measurement noise. **The sensitivity-peak
reading is not among them.** The new paper is a genuinely separate contribution
rather than a promised sequel — the better position for a second submission,
since a reviewer cannot argue it was carved out of work already announced.

## 2. Definitions cross-checked — all consistent

Proposition 3 of the parent reads exactly as the new paper uses it:

    N = [ln(1/ε)/2π] · √(1−ζ²)/ζ + η,    |η| < 1

Since ω/α = √(1−ζ²)/ζ, the new paper's `N = c·(ω/α) + η` with
`c = ln(1/ε)/2π` is a faithful restatement, not a reinterpretation. The
defaults ε = 0.1, N̄ = (0.5, 0.75, 1.0), δ = 0.02 all match.

**One detail worth exploiting.** The parent already observes that "the
truncation index makes η, and hence N_k, piecewise constant with unit jumps."
This is precisely the sawtooth behaviour the new paper measured in η — it
oscillates with ζ rather than varying smoothly. Remark 1 of the new paper
should cite that sentence directly. It converts an empirical observation into a
mechanism the parent already identified, and it costs nothing.

## 3. A notation collision to fix before submission

**The symbol `c` means different things in the two papers.**

| | parent paper | new paper |
|---|---|---|
| `c` | stability screen margin, default **2** | counting constant, **0.3665** |
| where | Definition 3, Table 3 | the central law `Ms·|ν| = N/c` |

Because the new paper cites the parent throughout, readers will hold both at
once, and the clash lands in exactly the equation that matters most.

**Recommendation:** rename the counting constant. `κ` is unused in both papers
and reads cleanly: `Ms·|ν| = N/κ`.

## 4. Citation gaps confirmed against the parent's bibliography

The parent's 20 references cover the tuning literature thoroughly — Ziegler–
Nichols, Cohen–Coon, Rivera (IMC), Skogestad (SIMC), Åström–Hägglund (1984,
2001, 2004, 2006), Schlegel value sets, Ho and Brabec–Schlegel on H∞ parametric
regions, Hjalmarsson (IFT), Campi (VRFT), Killingsworth–Krstić, Kraus–Myron
(EXACT), Graham–McRuer, Nyquist, Basseville–Nikiforov, Desborough–Miller.

It contains **none** of the control-performance-monitoring literature the
novelty argument must address. Add to the new paper:

| reference | why it must be cited |
|---|---|
| Hägglund (1995), CLPM | nearest model-free routine-data competitor |
| Hägglund (1999), Idle Index | nearest model-free "phase relationship" index |
| Harris (1989) | origin of the variance-benchmarking branch |
| Huang & Shah (1999) | standard monograph for that branch |
| Bauer et al. (2016) | evidence that robustness monitoring lags variance |
| Isoshima, Tanemura & Chida (2023) | closest model-free-margins-from-data result |

Desborough–Miller and the Åström–Hägglund Ms framework are already in the
parent's list, so the new paper inherits them naturally.

## 5. One overlap to handle visibly

The parent's Proposition 2 already establishes that stability is lost through a
single pole pair at the phase-crossover frequency ω₁₈₀, and Proposition 6
already reasons about "the margin to the first touch of −1" and about crossover
frequencies not moving under uniform gain scaling. The new derivation builds
directly on all of this.

This is legitimate reuse of a cited result, but it should be attributed **at the
point of use**, because a reviewer comparing the two papers will notice that
ω₁₈₀, the single critical pair, and the touch of −1 all originate in the parent.

The new contribution is: the complex slope ν defined at that frequency, the
substitution w = μ + νu reducing the loop to 1 + L = 1 − e^(−w), and the
cancellation of μ between the pole and the peak. Not the crossover framing
itself. Saying so explicitly protects the claim rather than weakening it.

## 6. Revised phrasing of the novelty claim

With the parent read, the defensible one-sentence claim is:

> The turn index of a closed-loop step record, introduced previously as a
> damping diagnostic, is here shown to measure the sensitivity peak:
> Ms·|ν| = N/κ, where ν is the complex logarithmic slope of the loop at its
> −180° crossing. The distance to instability cancels from the relation, so the
> peak follows from one ordinary setpoint step with no model, no identification,
> and no dedicated experiment.

This is narrower than "first model-free robustness from routine data" — which
Hägglund (1995, 1999) and Isoshima et al. (2023) could contest — and wider than
a mere corollary of the parent, which the parent's own silence on Ms disproves.

## 7. What did not change

The eight-part prior-art survey stands as written. Reading the parent affects
only the self-pre-emption question and the citation/notation housekeeping above.
The recommendation for a final targeted sweep of Journal of Process Control,
Control Engineering Practice, IEEE TCST and the IFAC PID symposia (2020–2026)
before submission remains, as does the caveat about the Isoshima et al. article
number (111008 vs 111160) needing verification against the publisher record.
