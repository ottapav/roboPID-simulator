"""
A continuous turn index that is a regularization of the published count.

The published count (core.features.compute_features + encirc) is built from
discrete steps, each of which makes it jump with the gains:

  0. the window starts at k1 = (first k with e_k >= -0.9) + K1_PADDING;
  1. the window ends at the settling index k_delta (Definition 4): the first
     sample after which |e| stays inside delta * peak|e|;
  2. each axis is normalized by its peak over the window [k1, k_delta];
  3. Gamma0's integral axis is taken relative to its value at k_delta;
  4. the winding is cut at the last segment that touches the eps-disc
     (find_disc_entrypoint), and is the unwrapped angle up to there.

and, as for any winding number, a segment passing through the origin turns its
angle from +pi to -pi.

soft_counts() replaces each step by a smooth version with a sharpness p, and is
the published count again as p -> infinity:

  0. point k is past the start with weight B_k = s(max_{m <= k-pad} e_m + 0.9),
     s a logistic of sharpness p_start (in units of e);
  1. point k is in the window with weight P_k = s(E_{k-1}), where
     E_k = max_{m >= k} |e_m| / peak|e| is the future envelope of the error and
     s(E) = 1 / (1 + (delta/E)^p). P_k -> 1 for k <= k_delta, 0 after;
  2. peaks become max_k B_k P_k |x_k|;
  3. the reference value is sum_k pi_k y_k with pi_k = P_k - P_{k+1}, the
     smooth distribution of the last point of the window;
  4. segment k counts with S_k * V_k: S_k = B_k P_{k+1} (both ends in the
     window), V_k the smooth "some later segment still touches the disc", or 1
     if none ever does (the published count then keeps the whole window);
  5. each segment angle is weighted by d_k^2 / (d_k^2 + r0^2), d_k the
     segment's distance from the origin, r0 = eps/100 by default: zero for a
     segment through the origin, where the winding number itself is ambiguous,
     and 1 to within 1e-4 for every segment the eps-disc test treats as
     outside it (d > eps).

Each weight is a continuous function of the record, and the record of the
gains, so the index is continuous in the gains for every finite p. The angle
of a segment is its exact angle atan2(cross, dot).

sharp=None evaluates the p = infinity limit with hard indicators (and r0 = 0),
which is used to check the construction against the published count. Checked
on 7459 states visited by hard-count runs on P1-P4 (regindex_soft_check.py):
identical to the published count up to rounding except (a) 6 states where
encirc misses the last segment's crossing of the negative x-axis (a one-turn
error of the published implementation), (b) 3 states with a segment through
the origin (ambiguous winding, resolved differently), (c) 3 states in the
corner case j <= 1, where encirc returns 0 (difference 0.011).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from core.features import DELTA, EPSILON
from core.params import K1_PADDING

# (x column, y column, y relative to the window end) of Gamma0, Gamma1, Gamma2
# in the add_derivatives matrix: 0 = e, 1 = diff, 2 = 2nd diff, 3 = cumsum.
AXES = ((0, 3, True), (1, 0, False), (2, 1, False))


def _step(z, sharp):
    """Smooth indicator of z > 0 for z in log units; hard if sharp is None."""
    if sharp is None:
        return (z > 0).astype(float)
    return expit(sharp * z)


def _seg_dist(x, y):
    """Distance from the origin to each segment (x_k, y_k) -> (x_{k+1}, y_{k+1})."""
    ax, ay, bx, by = x[:-1], y[:-1], x[1:], y[1:]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, np.clip(-(ax * dx + ay * dy) / np.where(L2 > 0, L2, 1), 0, 1), 1.0)
    return np.hypot(ax + t * dx, ay + t * dy)


def soft_counts(ext_e, e, k1, k2, sharp=32.0, sharp_disc=None, sharp_start=200.0,
                r0=None, delta=DELTA, eps=EPSILON, return_parts=False):
    """
    Smooth counts (N0, N1, N2) from the derivative matrix ext_e of the error
    (core.signals.add_derivatives), the error e (app convention y - r), and the
    published window indices k1, k2 (k1 is used only in the hard limit).
    sharp: sharpness of the window end (log units); sharp_disc: of the disc
    test, default sharp; sharp_start: of the window start (units of e);
    r0: origin regularization radius, default eps/100 (0 in the hard limit).
    """
    hard = sharp is None
    if sharp_disc is None:
        sharp_disc = sharp
    if r0 is None:
        r0 = 0.0 if hard else eps / 100
    e = np.asarray(e, float)
    peak = np.max(np.abs(e[:k2 + 1]))
    if peak < 1e-12 or k2 < 2:
        return np.zeros(3)
    n = k2 + 1
    # 0. start
    if hard:
        B = (np.arange(n) >= k1).astype(float)
    else:
        lagged = np.concatenate([np.full(K1_PADDING, -np.inf), np.maximum.accumulate(e[:n - K1_PADDING])])
        B = expit(sharp_start * (np.where(np.isfinite(lagged), lagged, -1e9) + 0.9))
    # 1. end: future envelope of |e|
    a = np.abs(e[:n]) / peak
    E = np.maximum.accumulate(a[::-1])[::-1]
    with np.errstate(divide='ignore'):
        W = _step(np.log(np.maximum(E, 1e-300)) - np.log(delta), sharp)
    P = np.concatenate([[1.0], W[:-1]])                         # point k <= k_delta
    if hard:                                                    # k_delta is clipped to [k1, k2]
        P[:k1 + 1] = 1.0
    Q = B * P                                                   # point in the window
    pi = P - np.concatenate([P[1:], [0.0]])                     # last point of the window
    S = B[:-1] * P[1:]                                          # segment in the window
    N, parts = [], []
    for ix, iy, rel in AXES:
        x = ext_e[:n, ix].astype(float)
        y = ext_e[:n, iy].astype(float)
        if rel:
            y = y - np.sum(pi * y) / np.sum(pi)
        mx, my = np.max(Q * np.abs(x)), np.max(Q * np.abs(y))
        if mx < 1e-12 or my < 1e-12:
            N.append(0.0); parts.append(None); continue
        x, y = x / mx, y / my
        dphi = np.arctan2(x[:-1] * y[1:] - y[:-1] * x[1:], x[:-1] * x[1:] + y[:-1] * y[1:])
        d = _seg_dist(x, y)
        org = d * d / (d * d + r0 * r0) if r0 > 0 else np.ones_like(d)
        with np.errstate(divide='ignore'):
            hit = _step(np.log(eps) - np.log(np.maximum(d, 1e-300)), sharp_disc) * S
        later = 1 - np.concatenate([np.cumprod((1 - hit)[::-1])[::-1][1:], [1.0]])
        none = np.prod(1 - hit)
        V = np.minimum(later + none, 1.0)
        N.append(float(np.sum(S * V * org * dphi) / (2 * np.pi)))
        parts.append(dict(S=S, V=V, dphi=dphi, d=d))
    return (np.array(N), parts) if return_parts else np.array(N)
