# Is the terminal orbit unique? — findings on the current algorithm, 1 Nov 2026

> **Status.** This is the investigation log. The draft
> (`conference_draft_ecc.tex`) now uses a different setting — derivative on
> the error, linear loop, starts around the AMIGO gains — and its numbers come
> from `scripts/lc_results.py` / `lc_results.json`. Two findings below were
> refined there: the event the orbits sit on is the **overshoot** reaching the
> δ band (not the undershoot), and `encirc` had a second error (crossings by a
> segment with one end at x ≥ 0), now fixed together with the last-segment one.
> The complex-zero guard of §5 was not adopted: with the derivative on the
> error the orbits it addressed do not occur.

The measurements in the ECC draft predate small changes to the algorithm and
are not used here; everything below is from the current code. Script:
`docs/scripts/orbit_uniqueness.py`, results: `orbit_uniqueness.json`.

**Configuration** — the app's defaults (`core/params.py`, `robopid.config`):
500-sample auto grid, actuator saturation ±10 (simtype 1), derivative on the
measurement (Basic SPIN), box [0.001, 10], β = 0.1, Nbar = (0.5, 0.75, 1.0).
Per plant, 27 starts on a 3×3×3 grid of factor 8 around the orbit reached from
unit gains; every run until the first exact state revisit (the iteration is
deterministic, so this is the closed terminal orbit).

Three indices are compared:

- **hard** — the published count over the δ-guarded window [k1, k_δ]: what the
  algorithm runs;
- **reg-guard** — the regularized winding integral of the ECC draft over the
  same guarded window: what `regindex.patch()` gives, since it swaps only the
  metric;
- **reg-full** — the regularized integral over [k1, k2]: what Sec. 3 of the
  draft defines.

## 1. The algorithm as it ran (hard count, rule before the fix of §5)

M_s at the orbit centroid. These numbers are from the run before the fix;
`orbit_uniqueness.json` now holds the run with the fixed rule.

| plant | Ki | Kp | Kd | M_s | orbits with a gain on the box |
|---|---|---|---|---|---|
| P1 | 0.113–0.130 | 1.23–1.64 | 0.001–1.34 | 1.50–1.57 | 9/27, all Kd at floor |
| P2 | 0.023–0.025 | 0.31–0.39 | 0.001 | 1.64–1.76 | 27/27, all Kd at floor |
| P3 | 0.013–0.058 | 0.07–1.00 | 0.001–4.2 | **1.72–9.54** | 7/27, all Kd at floor |
| P4 | 0.016–0.017 | 0.40–0.51 | 0.001 | 1.64–1.76 | 27/27, all Kd at floor |

P1, P2, P4: the destination is effectively unique in what matters (Ki within
15 %, M_s within 0.12). On P2 and P4 the rule removes derivative action
entirely, from every start. On P1, Kd is left anywhere in [0.001, 1.34], with
no effect on Ki or M_s.

**P3: 3 of 27 starts end in orbits with M_s 8–11.** Two distinct mechanisms,
both deterministic:

1. *A blocked cut still pays out.* Start (Ki/8, 8 Kp, Kd/8): the orbit
   `1 2 0 0` (period 4) at (Ki, Kp, Kd) ≈ (0.013, 1.00, 0.001), M_s 8.0–10.0.
   Band 1 rings (N1 = 4.5) and leaks into Γ2 (N2 = 4.3). When the rule fires
   d2, Kd is already at the floor, so the cut is clipped away, but the raises
   of Ki and Kp that come with it are still applied and undo the d1
   correction. The orbit closes only because of the clipping: (+1,−1,0) +
   (+1,+1,0) + 2·(−1,0,0) = 0. This is a sharper form of Limitation 6 of the
   companion paper: a band at its floor that keeps ringing does not just block
   the cascade, it pumps the bands below it.
2. *Ringing produced by the derivative, attributed to Kp.* Start (Ki, Kp/8,
   8 Kd): orbit at (0.023, 0.074, 4.2), M_s 9.5, with N = (0.12, 1.09, 0.52).
   The under-damped mode is created by the large Kd, Γ1 sees it, the rule cuts
   Kp, and Kd is never touched because N2 stays below its limit. This is the
   case the companion paper's Remark (Scope) excludes from its argument
   ("strong derivative action ... critical pair captured by the controller
   zeros"); here it occurs from an 8× mistuned start.

## 2. The analysis index (regularized) does not describe that destination

| plant | index | Ki | Kd | M_s |
|---|---|---|---|---|
| P1 | hard | 0.113–0.130 | 0.001–1.34 | 1.50–1.57 |
| P1 | reg-guard | 0.17–0.51 (4 clusters) | 4.8–6.6 | 2.14–2.92 |
| P1 | reg-full | 0.37–0.49 (24/27 one cluster) | 4.4–6.3 | 2.49–2.94 |
| P2 | hard | 0.023–0.025 | 0.001 | 1.64–1.76 |
| P2 | reg-full | 0.027–0.050 | 0.001–5.7 | 1.90–2.32 |
| P4 | hard | 0.016–0.017 | 0.001 | 1.64–1.76 |
| P4 | reg-full | 0.019–0.024 | 0.001–10 | 1.86–2.01 |

The regularized index tunes more aggressively than the count it is meant to
stand in for (P1: Ki ×3–4, M_s ≈ 2.7 against 1.5), and keeps Kd where the hard
count removes it. The premise of the draft's Sec. 3, that the regularized index
changes no sign comparison of the rule and is "used only for the analysis",
does not hold on the current code: the analysed orbit is not the one the
algorithm reaches.

Two further properties of the regularized index:

- **The guarded window makes it discontinuous.** On P1 a gain change of 0.1h
  moves k_δ (one more lobe crossing the δ band) and all three indices jump
  together, e.g. from (0.18, 0.40, 0.74) to (0.63, 1.37, 1.78). Orbits close on
  such cliffs without a root nearby (reg-guard: 4 clusters on P1). Over the full
  window the index is continuous and 24/27 starts reach one orbit.
- **Kd is not identifiable on P2 and P4.** At the centroids of the 17 orbits
  found (reg-full), the smallest singular value of dN/d(log K) (central
  differences, step 0.02) is 0.000–0.18, and in 16 of them its singular vector
  is almost pure log Kd (Kd component ≥ 0.99); N = Nbar then has a curve of
  solutions along Kd, and the
  orbit closes wherever the start brings it onto the curve (Kd from the floor
  to the ceiling). Ki and M_s barely change along it.

## 3. What survives

The combinatorial results (Prop. 1 balance, Cor. 2 counts n_e(1, 4, 2, 1),
Cor. 3 excursion bound) concern orbits in the interior of the box and hold for
every interior orbit found. Orbits on the box can be shorter (period 4 on P3),
because clipping removes a component of a move; the draft's "Boundary Case"
paragraph should say so.

## 4. Consequences

1. **For the ECC draft:** Corollary 1 (one triple point, independent of the
   start) needs a transversality hypothesis, and it is about the regularized
   index, whose destination differs from the algorithm's. Either the analysis
   moves to the hard count (piecewise constant, so a different argument), or
   the draft must state that it analyses a different rule. All numbers need
   regenerating from the current code in any case.
2. **For the algorithm:** the blocked-cut pump (§1.1) is fixed in
   `core.tuning.triangular_rule` (§5). Mechanism §1.2 has a candidate fix, a
   complex-zero guard, measured but not applied (§5).
3. **For acceleration:** the target set is a point only on some plants; on the
   others the quality to compare is Ki and M_s, not the distance to a
   reference orbit. And any acceleration must not make the §1 failure modes
   more likely: larger steps reach more of the gain space, including the
   P3 basins with M_s ≈ 10.

## 5. Fix applied: a blocked cut passes down

`core.tuning.triangular_rule` now treats a band as *cuttable* only if it is
active (nonzero base gain) and its multiplier is above its floor (relative
tolerance 1e-9, so a floor reached through a log/exp round trip counts). When
the lowest violated band k_min is not cuttable, the cut passes to the nearest
lower band that is, with the usual raises below that band (rows `N2/N1`,
`N2/N0`, `N1/N0`); if none is, the gains are held (row `blocked`). Rule of the
companion paper otherwise unchanged: the pass-down never fires while k_min can
move. The same defect hit PI (Kd = 0) and I (Kp = Kd = 0) controllers on every
Γ2 (resp. Γ1) violation, since there the cut of the inactive band did nothing
and only the raises took effect.

Measured with the same 27 starts per plant (worst M_s over the terminal orbit):

| plant | before: starts with M_s > 3 | after: starts with M_s > 3 | after: worst / median M_s |
|---|---|---|---|
| P1 | 0 | 0 | 1.62 / 1.60 |
| P2 | 0 | 0 | 1.88 / 1.81 |
| P3 | 3 | 1 | 11.39 / 2.02 |
| P4 | 0 | 0 | 1.89 / 1.80 |

From the AMIGO start without noise the fixed rule returns exactly the gains of
the old one on P1–P4, and the golden tuning trajectories are unchanged. Under
the app's default output noise (σ = 1 %) both rules collapse Kp to the floor on
all four plants (Γ2 sees the noise as buzzing); the fix does not change that
picture, it only changes which gain absorbs the noise-driven cut.

**Not applied — complex-zero guard for mechanism §1.2.** The remaining P3 orbit
has Kp = 0.07, Kd = 4.2, i.e. Td = Kd/Kp ≈ 57 s against Ti = Kp/Ki ≈ 3 s: the
controller zeros are complex (Kp² < 4 Ki Kd), which is the "critical pair
captured by the controller zeros" case the companion paper excludes. Tested in
the harness only: when a band is violated and Kp² < 4 Ki Kd, cut Kd alone.

| plant | fixed rule: worst M_s | + zero guard: worst M_s | Ki range with guard |
|---|---|---|---|
| P1 | 1.62 | 1.62 | unchanged |
| P2 | 1.88 | 1.88 | unchanged |
| P3 | 11.39 | **2.10** | 0.050–0.057 (was 0.023–0.058) |
| P4 | 1.89 | 1.89 | unchanged |

It is plant-free and changes nothing where the zeros are real. It adds a row to
the rule, so it is left for a decision.

## 6. The regularized index: what it should regularize

Scripts: `docs/scripts/regindex_sharpness.py`, `docs/scripts/regindex_envelope.py`.

**Observation about the published count.** `find_disc_entrypoint` returns the
last segment that *touches* the ε-disc. A portrait that ends inside the disc
touches it to the very end of the window, so the published count truncates
almost nothing: on the P1 orbit it stops at sample 59 of 61. In practice the
count is the full winding over the guarded window [k1, k_δ], and it is k_δ, not
the disc, that ends the count — and that makes it jump.

**The draft's regularization measures something else.** Weighting by the
radius, ρ^p/(ρ^p + ε^p), discards the small-radius winding the hard count
includes. Share of states (visited by the hard-count runs) where it makes the
same move as the hard count:

| plant | ρ-weight p=2, guarded | p=2, full | p=16, guarded | p=16, full |
|---|---|---|---|---|
| P1 | 30 % | 36 % | 36 % | 37 % |
| P2 | 32 % | 38 % | 39 % | 39 % |
| P3 | 45 % | 50 % | 53 % | 53 % |
| P4 | 38 % | 48 % | 49 % | 44 % |

Sharpening the weight does not help, and neither does an exact segment angle
(per-sample rotation is only 2–12°): the difference is in what is weighted.

**Regularize k_δ instead.** Weight each segment by the future envelope of the
error, E_k = max_{m≥k} |e_m| / max|e|, with w_k = E_k^p / (E_k^p + δ^p): a smooth
version of "k < k_δ", continuous in the gains, tending to the hard count as p
grows. Over [k1, k2], p = 32:

| plant | same move | median max\|N − N_hard\| | Ki (hard → envelope) | worst M_s (hard → envelope) |
|---|---|---|---|---|
| P1 | 88 % | 0.18 | 0.110–0.130 → 0.113–0.431 | 1.55–1.62 → 1.55–2.94 |
| P2 | 92 % | 0.11 | 0.023–0.025 → 0.023–0.026 | 1.75–1.88 → 1.73–1.90 |
| P3 | 93 % | 0.12 | 0.023–0.058 → 0.052–0.070 | 1.85–11.39 → 1.87–2.28 |
| P4 | 97 % | 0.13 | 0.015–0.017 → 0.016–0.017 | 1.73–1.89 → 1.76–1.87 |

On P2 and P4 the destinations coincide. On P3 the envelope index avoids the
M_s ≈ 11 orbit. On P1 some starts reach a second, more aggressive orbit
(Ki ≈ 0.43, M_s ≈ 2.9) that the hard count does not; the remaining differences
come from the normalization (full window instead of [k1, k_δ]) and from Γ0's
integral axis being taken relative to k2 instead of k_δ.

The envelope index was a first step; §7 completes it.

## 7. A smooth count that explains the rule

Scripts: `docs/scripts/regindex_soft.py` (definition),
`regindex_soft_check.py` (hard limit), `regindex_soft_eval.py` (finite
sharpness; results in `regindex_soft_eval.json`).

**Construction.** The published count is the composition of discrete steps:
window start k1, window end k_δ, per-axis peak normalization over the window,
Γ0's reference value at k_δ, the cut at the last segment touching the ε-disc,
and the winding sum itself, which jumps when a segment passes through the
origin. `soft_counts` replaces each by a smooth version with one sharpness p:

| published step | smooth version |
|---|---|
| start k1 = first(e ≥ −0.9) + pad | B_k = logistic(p_s · (max_{m≤k−pad} e_m + 0.9)) |
| end k_δ | P_k = 1/(1 + (δ/E_{k−1})^p), E_k = max_{m≥k}\|e_m\| / peak\|e\| |
| peak over [k1, k_δ] | max_k B_k P_k \|x_k\| |
| Γ0 reference y(k_δ) | Σ π_k y_k, π_k = P_k − P_{k+1} |
| cut at last disc touch | V_k = smooth "a later segment touches the disc" |
| segment through the origin | angle weighted by d_k²/(d_k² + r0²), r0 = ε/100 |

Every weight is a continuous function of the record, so the index is
continuous in the gains for every finite p, and it is the published count as
p → ∞.

**The hard limit is exact.** On 7459 states visited by hard-count runs (P1–P4,
27 starts × 120 iterations), the p → ∞ limit equals the published count to
rounding on 7447. The other 12 are all explained, none is open:

- 6: `encirc` misses a crossing of the negative x-axis on the last segment
  (its crossing loop runs over segments 0..j−2 while the angle is read at
  point j) — a one-turn error of the published implementation;
- 3: a segment through the origin, where the winding number is ambiguous;
- 3: the corner case j ≤ 1, where `encirc` returns 0 (difference 0.011).

**Finite sharpness.**

| plant | p | same move | median \|ΔN\| | Ki of terminal orbits (hard → smooth) | worst M_s (hard → smooth) |
|---|---|---|---|---|---|
| P1 | 8 | 96.7 % | 0.026 | 0.110–0.130 → 0.113–0.130 | 1.55–1.62 → 1.59–1.62 |
| P1 | 32 | 98.5 % | 0.008 | → 0.110–0.130 | → 1.55–1.62 |
| P2 | 32 | 98.7 % | 0.006 | 0.023–0.025 → 0.023–0.026 | 1.75–1.88 → 1.75–1.92 |
| P2 | 128 | 99.8 % | 0.005 | → 0.023–0.026 | → 1.75–1.88 |
| P3 | 8 | 96.9 % | 0.023 | 0.023–0.058 → 0.050–0.059 | 1.85–11.39 → 1.80–2.13 |
| P3 | 32 | 99.3 % | 0.007 | → 0.023–0.057 | → 1.85–11.39 |
| P4 | 32 | 99.6 % | 0.014 | 0.015–0.017 → 0.015–0.017 | 1.73–1.89 → 1.73–1.89 |

From p = 32 the smooth count reaches the hard count's destinations on all four
plants, including the M_s ≈ 11 orbit on P3, i.e. it reproduces the rule's
failure as well as its successes. (At p = 8 that orbit disappears: a softer
count is a different, and here better, rule — but then it no longer explains
the published one.) The second, aggressive P1 orbit of the envelope index (§6)
is gone; it came from the missing start, normalization and reference pieces.

**Continuity.** On the P1 line where the δ-guarded index jumped, at 1/400 of a
rule step, the hard count changes by up to 1.01 between neighbours, the smooth
count (p = 32) by up to 0.22. Refining that interval 10×, 100×, 1000× shrinks
the step to 0.025, 0.0025, 0.0002: continuous, and steep there because a Γ0
segment passes 0.0009 from the origin, inside r0. With r0 = ε/10 the same
interval is flat to 0.0002 at 100×: r0 trades smoothness near the origin
against agreement with the rule.

**Recommendation for the ECC draft.** Sec. 3 should define the analysed index
as this smooth count (p = 32, r0 = ε/100), replacing the radius-weighted N_ε:
the switching surfaces of the rule are then level sets of a continuous function
that makes the rule's own decision on ≈ 99 % of the states the rule visits and
leads to the rule's own destinations. The radius-weighted N_ε explained 30–50 %.

**For the algorithm.** The `encirc` last-segment error is an off-by-one in the
published code (6 of 7459 states, one turn each); fixing it changes the rule
only there. Not applied.
