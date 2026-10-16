"""
Certify before applying (Theorem 3): from the record of the running controller C, bound the
sensitivity peak of a proposed controller C' on the same plant and sampling period, before C' is
applied.

With Delta = C'/C - 1 on the unit circle, L' = L (1 + Delta) and, exactly,
    S' = S / (1 + T Delta),   T = 1 - S.
The record gives S^ with |S - S^| <= tau (Lemma 1, Assumption 1), hence |T - T^| <= tau. Then
  (i)  C' stabilizes the plant if, for all theta, dist(0, [1, 1 + T^ Delta]) > tau |Delta|
       (the Nyquist curve of L(1 + lambda Delta) never reaches -1 for lambda in [0, 1]);
  (ii) M_s(C') <= sup_theta (|S^| + tau) / (|1 + T^ Delta| - tau |Delta|).
On [0, th_a] both are evaluated on a grid with Lipschitz margins; on [th_a, pi] the Abel bound
|S - 1| <= r = B / sin(theta/2) of Theorem 1 gives |T| <= r, |S| <= 1 + r, so (i) holds if
r |Delta| < 1 and |S'| <= (1 + r) / (1 - r |Delta|).
"""
import math
import numpy as np
from cert import increments, S_grid, C_d


def ahead(e, dt, g_old, g_new, tau, Mstar, eta=2e-3, Mmax=4_000_000, nD=20001, Mhf=None):
    a = increments(e); N = len(a) - 1
    B = float(np.sum(abs(np.diff(a[1:]))) + abs(a[-1]) + 2 * tau)
    w = abs(a); cw = np.cumsum(w); nc = int(np.searchsorted(cw, cw[-1] / 2))
    J = float(np.sum(abs(np.arange(N + 1) - nc) * w))           # Lipschitz constant of |S^|
    J0 = float(np.sum(np.arange(N + 1) * w))                    # Lipschitz constant of S^ itself
    # Delta on a dense uniform grid of [0, pi]: its size and slope
    thD = np.linspace(1e-6, math.pi, nD); zD = np.exp(1j * thD)
    D = C_d(*g_new, dt, zD) / C_d(*g_old, dt, zD) - 1
    Dabs = abs(D); hD = thD[1] - thD[0]
    Dp = 1.2 * float(np.max(abs(np.diff(D)) / hD))               # slope of Delta (numerical, +20%)
    Dm = float(Dabs.max())
    # high frequencies: th_a such that (1 + r)/(1 - r Dm) <= Mhf, r = B / sin(th_a / 2); the default
    # Mhf = 1 + (M* - 1)/2 keeps this part well below M*, so the peak is resolved on the grid
    Mhf = 1 + (Mstar - 1) / 2 if Mhf is None else Mhf
    r_need = (Mhf - 1) / (1 + Mhf * Dm)
    th_a = math.pi if B >= r_need else 2 * math.asin(B / r_need)
    hf_ok, U_hf = True, 0.0
    if th_a < math.pi:
        m = thD >= th_a
        r = B / np.sin(thD[m] / 2); rD = r * Dabs[m]
        hf_ok = bool(np.all(rD < 1)); U_hf = float(np.max((1 + r) / (1 - rD))) if hf_ok else math.inf
    # grid part on [0, th_a]
    Sm0 = 1 + tau
    M = int(min(Mmax, max(2001, math.ceil(th_a * max(J, J0 * Dm + 2 * Dp) / (2 * eta)) + 1)))
    S, h = S_grid(a, th_a, M)
    th = np.arange(M) * h; th[0] = 1e-6
    z = np.exp(1j * th); Dg = C_d(*g_new, dt, z) / C_d(*g_old, dt, z) - 1
    T = 1 - S; q = T * Dg
    Sm = float(abs(S).max()) + h * J / 2
    Lq = J0 * Dm + (1 + Sm) * Dp                                 # Lipschitz constant of q = T^ Delta
    mq, mD = h * Lq / 2, h * Dp / 2                              # margins between grid points
    lam = np.clip(-(q.real) / np.maximum(abs(q) ** 2, 1e-300), 0, 1)
    dist = abs(1 + lam * q)                                      # distance from 0 to [1, 1 + q]
    slack = dist - mq - tau * (abs(Dg) + mD)
    stable = bool(np.all(slack > 0)) and hf_ok
    den = abs(1 + q) - mq - tau * (abs(Dg) + mD)
    U_grid = float(np.max((abs(S) + h * J / 2 + tau) / den)) if np.all(den > 0) else math.inf
    U = max(U_grid, U_hf) if stable else math.inf
    return dict(stable=stable, U=U, U_grid=U_grid, U_hf=U_hf, th_a=th_a, M=M, h=h,
                margins=(h * J / 2, mq, mD), Dm=Dm, Dp=Dp, J=J, J0=J0, B=B, read=float(abs(S).max()))
