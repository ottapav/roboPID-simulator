# Is the terminal orbit unique? — findings, 1 Nov 2026

Prompted by `docs/scripts/accel_schedule.py`: fixed-step runs from different
starts closed different terminal orbits, which contradicts Corollary 1 of the
ECC draft ("the triple point depends only on the plant ... not on the starting
gains"). Script: `docs/scripts/orbit_uniqueness.py`, results:
`orbit_uniqueness.json`. Setup as in the draft (regularized index, dtype='y',
box [0.01, 10], beta = 0.1); 27 starts per plant on a 3x3x3 grid of factor 8
around the unit-gain orbit; every run until the first exact state revisit.

Four explanations were tested: H1 several roots, H2 an ill-conditioned root,
H3 the grid Ts = 1, H4 the analysis window. **Two independent causes were
found: H4 on P1, and a degenerate (non-transversal) root on P2 and P4.** The
grid (H3) is not the cause: everything below holds on Ts = 1 and on the app's
own grid Ts = span/49.9.

## 1. P1: the delta guard makes the "continuous" index discontinuous

`regindex.patch()` swaps only the metric. `loop_response_features` still cuts
the window at the settling index k_delta, so the index the draft *runs* is the
regularized integrand over the guarded window [k1, k_delta], not the
full-window index of Sec. 3. A small gain change can add one oscillation lobe
above the delta band and move k_delta, which moves the index by a jump.

Measured on P1 (Ts = 1), along the uniform direction through the low-Ki orbit:

| offset | k_delta | N, window [k1, k_delta] | N, window [k1, k2] |
|---|---|---|---|
| −0.2h | 16 | (0.18, 0.40, 0.74) | (0.45, 1.28, 1.93) |
| −0.1h | 18 | (0.40, 0.73, 1.04) | (0.45, 1.35, 1.96) |
|  0.0h | 28 | (0.63, 1.37, 1.78) | (0.45, 1.40, 1.99) |

All three guarded indices jump together within 0.1h, the full-window ones move
smoothly. On such a cliff all four regions of the rule are adjacent, so an
orbit closes there although no triple point exists: least squares from that
orbit finds no root (residual 0.32). Note also that at Ts = 1 the guarded
window of P1 is only 8–20 samples long.

| index | P1 orbits from 27 starts |
|---|---|
| guarded window (as run in the draft) | 4 clusters, Ki from 0.19 to 1.03 (×5.3) |
| full window (as defined in Sec. 3) | **1 cluster**, all 27 starts, exact root; Ki ≈ 0.97 at Ts = 1 (residual 1e-12), Ki ≈ 0.58 on the app grid (residual 5e-10) |

The two indices also lead to different tunings: from unit gains the guarded
index closes at (Ki, Kp, Kd) = (0.19, 3.21, 8.36), M_s ≈ 2.6, the full-window
index at (0.96, 3.66, 6.59), M_s ≈ 3.0. So on P1 the premise of Sec. 3, that
the regularized index changes no sign comparison of the rule, does not hold.

## 2. P2, P4: Kd is not determined, the "triple point" is a curve

With either window and either grid, the three indices hardly depend on Kd over
a range of two to three decades. At every orbit found, the smallest singular
value of the Jacobian dN/d(log K) is 0.00–0.06, and its singular vector is
almost pure log Kd (e.g. P4: (0.00, 0.01, 1.00) at Kd = 0.13). The equations
N(theta) = Nbar then have a curve of solutions along Kd, not an isolated root,
and the cycle closes wherever the start brings the iterate onto that curve.

| plant (full window) | Ki | Kp | Kd | M_s |
|---|---|---|---|---|
| P2, app grid | 0.029–0.035 (×1.19) | 0.48–0.70 | **0.01–3.3** | 1.84–2.00 |
| P4, app grid | 0.020–0.026 (×1.31) | 0.62–0.99 | **0.01–10** | 1.82–2.01 |

Orbits that touch the box (app grid: 5/27 on P2, 9/27 on P4; Ts = 1: 0/27 and 14/27) are the same curve leaving
the box: in every case it is Kd at the floor 0.01 or the ceiling 10, never Ki
or Kp. On the app grid the curve is benign for the loop: Ki, which sets the
integrated error for a load step, varies by at most 31 %, and M_s by 0.2. On
the coarse grid Ts = 1 it is less so: P2 spreads to Ki ×1.85 and M_s 1.88–2.38. It is the direction
the loop cares least about, and the rule has no evidence to pick a point on it.

P3 is practically unique (Ki within ×1.17–1.40, 2–3 adjacent clusters).

## 3. What survives

The balance results do not depend on any of this. From unit gains at Ts = 1,
with either window, every plant closes an orbit with counts n_e·(1, 4, 2, 1)
and half-spans inside the excursion bound (Prop. 1, Cor. 2, Cor. 3). With the
full window all four orbits have n_e = 1 and period 8. With the guarded window
P3 closes with n_e = 3, period 24, in this exact-revisit run, where Table
`tab:duty` reports period 16 from a 2000-iteration run.

## 4. Consequences for the ECC draft

1. **Sec. 3 / regindex.py.** Compute N_eps over [k1, k2], as Sec. 3 defines it.
   This is a one-line change (pass k2 instead of k_delta), but it moves the P1
   destination, so every number produced through `regindex.patch()` must be
   regenerated: Table `tab:duty`, the theta* errors of Sec. 3, the Jacobian
   determinants of Sec. 6, Fig. `fig_meter` and the schedule results of Sec. 7.
2. **Corollary 1 needs a transversality hypothesis**: the switching surfaces
   meet transversally, i.e. dN/dtheta is nonsingular at theta*. Without it,
   the balance argument only places the orbit on the set where the surfaces
   come within O(h) of each other, and that set can be a curve.
3. **State the degenerate case on P2 and P4 rather than hide it.** On these
   plants Kd is not identifiable from the three indices, so the destination in
   Kd depends on the start (or on the box), while Ki, Kp and M_s do not, to
   within the numbers above. This is also why Sec. 6 should not report one
   nonsingular Jacobian for P4: at the orbits found here the smallest singular
   value is 0.00–0.06.
4. **Acceleration experiments** (`accel_schedule.py`) must be judged by Ki and
   M_s on P2/P4, not by the distance to a reference orbit, and rerun with the
   full-window index. On P1 and P3 the target is then unique and the
   comparison is meaningful.

Open: whether the rule should get a tie-break along the unidentified direction
(the companion paper's principle "rate is the cheapest gain to give up" would
pick the smallest Kd on the curve), or whether the curve should be reported as
the honest answer.
