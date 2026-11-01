# Is the terminal orbit unique? — findings on the current algorithm, 1 Nov 2026

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

## 1. The algorithm as it runs (hard count)

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
2. **For the algorithm:** the blocked-cut pump (§1.1) has a simple
   deterministic fix — when the cut of band k_min is clipped by the floor, do
   not raise the bands below it (hold, or treat band k_min as satisfied and
   pass to the next violated band). Mechanism §1.2 needs an attribution check
   for Kd and is harder.
3. **For acceleration:** the target set is a point only on some plants; on the
   others the quality to compare is Ki and M_s, not the distance to a
   reference orbit. And any acceleration must not make the §1 failure modes
   more likely: larger steps reach more of the gain space, including the
   P3 basins with M_s ≈ 10.
