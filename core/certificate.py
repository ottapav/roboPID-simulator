"""
Certified robustness from one setpoint step (docs/ECC27_certified_SPIN).

The tuner already applies a unit setpoint step in every iteration. In a stable,
one-degree-of-freedom digital loop the increments of the error after that step,

    a_n = e_n - e_{n-1},   e = r - y,   e_{-1} = 0,  e_0 = 1,

are the impulse response of the sensitivity function (Lemma 1):

    S(e^{j theta}) = sum_n a_n e^{-j n theta}.

From the finite record this module computes

  * certificate()   -- certified lower/upper bounds on M_s = max |S| (Theorem 1),
                       under the assumption that the unrecorded tail
                       sum_{n>N} |a_n| is at most tau;
  * certify_ahead() -- for a proposed controller C' with the same sampling
                       period, a certificate that C' stabilizes the plant and a
                       bound on its M_s, *before* C' is applied (Theorem 3),
                       from S' = S / (1 + T*rho), rho = C'/C - 1, T = 1 - S;
  * predict_Ms()    -- the same prediction with tau = 0 (no certificate), used
                       to choose which band to lower after a rejection;
  * true_Ms()       -- the exact M_s of the simulated digital loop, from the
                       model. Only a simulator knows it; it is shown next to the
                       certified bound so the two can be compared.

Everything except true_Ms() uses the record alone, no model. The certificate
needs a linear loop: no actuator saturation and no measurement noise in the
record (check_record() says whether a record qualifies).
"""

from __future__ import annotations

import math
from core.params import DERIV_FILTER_N

import numpy as np
from scipy.signal import czt

from .pid import _controller_tf
from .plant import plant_tf

TAU_CERT = 1e-3     # assumed bound on the unrecorded tail variation (Assumption 1)
ETA_CERT = 2e-3     # grid tolerance: h*J/2 <= eta
MSTAR = 1.7         # default target sensitivity peak


def increments(e_r: np.ndarray) -> np.ndarray:
    """a_n = e_n - e_{n-1} with e_{-1} = 0 (e_r is the error r - y)."""
    return np.diff(np.concatenate([[0.0], np.asarray(e_r, dtype=float)]))


def tail_estimate(e_r, frac: float = 0.1, margin: float = 2.0) -> float:
    """
    A data-based guess of the unrecorded tail sum_{n>N} |a_n|: margin times the
    variation over the last frac of the record. It is a heuristic, not a proof
    -- no finite record can show that its tail is small (Assumption 1) -- but
    for a response that still decays geometrically it errs on the safe side,
    and it keeps a record cut short by the simulation horizon from being
    certified with too small a tau.
    """
    a = increments(e_r)
    n0 = int((1 - frac) * (len(a) - 1))
    return float(margin * np.sum(np.abs(a[n0:])))


def tau_for(e_r, tau_min: float = TAU_CERT) -> float:
    """The tail bound used for a record: at least tau_min, at least tail_estimate."""
    return max(tau_min, tail_estimate(e_r))


def _S_grid(a: np.ndarray, th1: float, M: int) -> tuple[np.ndarray, float]:
    """S_N on the uniform grid theta_k = k*th1/(M-1), k = 0..M-1, by one chirp z-transform."""
    h = th1 / (M - 1)
    return czt(a, m=M, w=np.exp(-1j * h), a=1.0), h


def _J(a: np.ndarray) -> float:
    """Slope bound of |S_N|: J = sum |n - n_c| |a_n| with n_c the weighted median."""
    w = np.abs(a)
    cw = np.cumsum(w)
    nc = int(np.searchsorted(cw, cw[-1] / 2))
    return float(np.sum(np.abs(np.arange(len(a)) - nc) * w))


def _B(a: np.ndarray, tau: float) -> float:
    """High-frequency (Abel) constant: |S - 1| <= B / sin(theta/2)."""
    return float(np.sum(np.abs(np.diff(a[1:]))) + abs(a[-1]) + 2 * tau)


def certificate(e_r, tau: float = TAU_CERT, eta: float = ETA_CERT,
                Mhf: float | None = None, Mmax: int = 2_000_000) -> dict:
    """
    Theorem 1. Returns dict(lower, upper, read, ...): lower <= M_s <= upper
    whenever the loop is stable and sum_{n>N} |a_n| <= tau.

    The high-frequency cut theta_1 is chosen so that 1 + B/sin(theta_1/2) = Mhf
    (theta_1 = pi if 1 + B > Mhf). Any choice gives a valid bound; the default,
    the peak |S_N| read on a coarse grid, makes the bound tight (upper is then
    within about tau + eta of M_s).
    """
    a = increments(e_r)
    J, B = _J(a), _B(a, tau)
    if Mhf is None:
        Sc, _ = _S_grid(a, math.pi, 4001)          # coarse: only picks theta_1; any theta_1 is valid
        Mhf = max(float(np.max(np.abs(Sc))), 1.0 + 1e-6)
    th1 = math.pi if B >= Mhf - 1 else 2 * math.asin(B / (Mhf - 1))
    M = int(min(Mmax, max(2001, math.ceil(th1 * J / (2 * eta)) + 1)))
    S, h = _S_grid(a, th1, M)
    read = float(np.max(np.abs(S)))
    up_grid = read + h * J / 2 + tau
    up_hf = 1 + B / math.sin(th1 / 2) if th1 < math.pi else 0.0
    return dict(lower=read - tau, upper=max(up_grid, up_hf), read=read,
                th1=th1, grid_term=h * J / 2, up_hf=up_hf, B=B, J=J, M=M)


def controller_freq(Kp: float, Ki: float, Kd: float, Ts: float, z: np.ndarray) -> np.ndarray:
    """C(z) of the app's controller (derivative on the error), evaluated at z.

    The sum of the PI part (Ki*Ts + Kp - Kp/z) / (1 - 1/z) and the filtered derivative
    Kd (z - 1) / ((Ts + Tf) z - Tf), Tf = Kd/N, evaluated directly: the same function as
    the polynomial form of core.pid._controller_tf, without building polynomials."""
    z = np.asarray(z)
    pi = (Ki * Ts + Kp - Kp / z) / (1.0 - 1.0 / z)
    if Kd == 0.0:
        return pi
    Tf = Kd / DERIV_FILTER_N
    return pi + Kd * (z - 1.0) / ((Ts + Tf) * z - Tf)


def certify_ahead(e_r, Ts: float, gains_old, gains_new, tau: float = TAU_CERT,
                  Mstar: float = MSTAR, eta: float = ETA_CERT, nD: int = 20001,
                  Mmax: int = 2_000_000) -> dict:
    """
    Theorem 3: from the record of the running controller C (gains_old =
    (Kp, Ki, Kd)), certify the proposed controller C' (gains_new) before it is
    applied. Returns dict(stable, U): if stable, C' stabilizes the plant and
    M_s(C') <= U.

    Conditions, for all theta in (0, pi]:
      (4)  min_{0<=lam<=1} |1 + lam*T^*rho| - tau*|rho| > 0     -> C' stabilizes
      (5)  M_s(C') <= sup (|S^| + tau) / (|1 + T^*rho| - tau*|rho|)
    checked on a grid with Lipschitz margins; above theta_a the bound
    |S - 1| <= r = B/sin(theta/2) replaces the record.
    """
    a = increments(e_r)
    N = len(a) - 1
    B = _B(a, tau)
    J = _J(a)
    J0 = float(np.sum(np.arange(N + 1) * np.abs(a)))       # slope bound of S^ itself

    # rho on a dense grid: its size and (numerical, +20 %) slope
    thD = np.linspace(1e-6, math.pi, nD)
    zD = np.exp(1j * thD)
    rho = controller_freq(*gains_new, Ts, zD) / controller_freq(*gains_old, Ts, zD) - 1
    rabs = np.abs(rho)
    Dp = 1.2 * float(np.max(np.abs(np.diff(rho)) / (thD[1] - thD[0])))
    Dm = float(rabs.max())

    # high frequencies: theta_a with (1 + r)/(1 - r*Dm) <= Mhf, Mhf halfway to M*
    Mhf = 1 + (Mstar - 1) / 2
    r_need = (Mhf - 1) / (1 + Mhf * Dm)
    th_a = math.pi if B >= r_need else 2 * math.asin(B / r_need)
    hf_ok, U_hf = True, 0.0
    if th_a < math.pi:
        m = thD >= th_a
        r = B / np.sin(thD[m] / 2)
        rD = r * rabs[m]
        hf_ok = bool(np.all(rD < 1))
        U_hf = float(np.max((1 + r) / (1 - rD))) if hf_ok else math.inf

    # grid part on [0, th_a]
    M = int(min(Mmax, max(2001, math.ceil(th_a * max(J, J0 * Dm + 2 * Dp) / (2 * eta)) + 1)))
    S, h = _S_grid(a, th_a, M)
    th = np.arange(M) * h
    th[0] = 1e-6
    z = np.exp(1j * th)
    rg = controller_freq(*gains_new, Ts, z) / controller_freq(*gains_old, Ts, z) - 1
    q = (1 - S) * rg
    Sm = float(np.max(np.abs(S))) + h * J / 2
    Lq = J0 * Dm + (1 + Sm) * Dp                             # slope bound of q = T^*rho
    mq, mD = h * Lq / 2, h * Dp / 2
    lam = np.clip(-q.real / np.maximum(np.abs(q) ** 2, 1e-300), 0, 1)
    dist = np.abs(1 + lam * q)                               # distance from 0 to [1, 1+q]
    slack = dist - mq - tau * (np.abs(rg) + mD)
    stable = bool(np.all(slack > 0)) and hf_ok
    den = np.abs(1 + q) - mq - tau * (np.abs(rg) + mD)
    U_grid = (float(np.max((np.abs(S) + h * J / 2 + tau) / den))
              if np.all(den > 0) else math.inf)
    U = max(U_grid, U_hf) if stable else math.inf
    return dict(stable=stable, U=U, U_grid=U_grid, U_hf=U_hf, th_a=th_a, M=M)


def predict_Ms(e_r, Ts: float, gains_old, gains_new, M: int = 8001) -> float:
    """Prediction of M_s(C') from the record of C (Theorem 3 with tau = 0)."""
    a = increments(e_r)
    S, h = _S_grid(a, math.pi, M)
    th = np.arange(M) * h
    th[0] = 1e-6
    z = np.exp(1j * th)
    rho = controller_freq(*gains_new, Ts, z) / controller_freq(*gains_old, Ts, z) - 1
    return float(np.max(np.abs(S / (1 + (1 - S) * rho))))


def plant_freq(tau, K: float, L: float, Ts: float, z: np.ndarray, simtype: int = 0) -> np.ndarray:
    """
    P(z) of the plant as the app simulates it. simtype 0 (linear TF path) uses
    plant_tf, a product of separately discretized lags; simtype 1 (anti-windup
    loop) discretizes the whole lag cascade at once (zero-order hold), so its
    P(z) = K z^-nd c^T (zI - A)^-1 B differs slightly for more than one lag.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if simtype == 0:
        num, den = plant_tf(tau, K, L, Ts)
        return np.polyval(num, z) / np.polyval(den, z)
    import scipy.linalg
    ns = len(tau)
    Mc = np.zeros((1 + ns, 1 + ns))
    for i in range(1, 1 + ns):
        Mc[i, i] = -1.0 / tau[i - 1]
        Mc[i, i - 1] = 1.0 / tau[i - 1]
    Md = scipy.linalg.expm(Mc * Ts)
    A, Bv = Md[1:, 1:], Md[1:, 0]
    nd = int(round(L / Ts))
    zz = np.atleast_1d(z)
    X = np.linalg.solve(zz[:, None, None] * np.eye(ns)[None] - A[None], np.broadcast_to(Bv, (len(zz), ns))[..., None])
    return K * zz ** (-nd) * X[:, -1, 0]


def _plant_poly(tau, K: float, L: float, Ts: float, simtype: int):
    """(num, den) of P(z) in descending powers of z, as the app simulates it."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if simtype == 0:
        return plant_tf(tau, K, L, Ts)
    import scipy.linalg
    from scipy.signal import ss2tf
    ns = len(tau)
    Mc = np.zeros((1 + ns, 1 + ns))
    for i in range(1, 1 + ns):
        Mc[i, i] = -1.0 / tau[i - 1]
        Mc[i, i - 1] = 1.0 / tau[i - 1]
    Md = scipy.linalg.expm(Mc * Ts)
    A, Bv = Md[1:, 1:], Md[1:, :1]
    C = np.zeros((1, ns)); C[0, -1] = K
    num, den = ss2tf(A, Bv, C, np.zeros((1, 1)))
    nd = int(round(L / Ts))
    return np.atleast_1d(num[0]), np.concatenate([den, np.zeros(nd)])


def loop_stable(tau, K: float, L: float, Ts: float, Kp: float, Ki: float, Kd: float,
                simtype: int = 0) -> bool:
    """Closed-loop stability of the simulated digital loop, from its poles (model-based)."""
    num_p, den_p = _plant_poly(tau, K, L, Ts, simtype)
    num_c, den_c = _controller_tf(Kp, Ki, Kd, 'e', Ts)
    num_l, den_l = np.polymul(num_c, num_p), np.polymul(den_c, den_p)
    n = max(len(num_l), len(den_l))
    char = np.concatenate([np.zeros(n - len(den_l)), den_l]) + \
        np.concatenate([np.zeros(n - len(num_l)), num_l])
    char = np.trim_zeros(char, 'f')
    return bool(np.all(np.abs(np.roots(char)) < 1 - 1e-9))


def true_Ms(tau, K: float, L: float, Ts: float, Kp: float, Ki: float, Kd: float,
            simtype: int = 0, n: int = 4000) -> float:
    """
    Exact M_s of the simulated digital loop, L = C*P (model-based; for display),
    or inf if that loop is unstable. The loop gain is the same whether the
    derivative acts on the error or on the output, so this is the robustness of
    either realization.
    """
    if not loop_stable(tau, K, L, Ts, Kp, Ki, Kd, simtype):
        return math.inf
    def S_at(th):
        z = np.exp(1j * th)
        Lz = plant_freq(tau, K, L, Ts, z, simtype) * controller_freq(Kp, Ki, Kd, Ts, z)
        return np.abs(1 / (1 + Lz))

    th = np.logspace(-6, math.log10(math.pi), n)
    S = S_at(th)
    for _ in range(3):
        i = int(np.argmax(S))
        lo, hi = th[max(i - 1, 0)], th[min(i + 1, len(th) - 1)]
        th = np.linspace(lo, hi, 401)
        S = S_at(th)
    return float(np.max(S))


def check_record(e_r, u=None, minu: float | None = None, maxu: float | None = None,
                 noisy: bool = False) -> str | None:
    """
    Why a record cannot be certified, or None if it can. The certificate needs
    a linear, noise-free record of a unit step from rest: e_0 = 1, no actuator
    saturation, no measurement noise, and a finite, settled response.
    """
    e_r = np.asarray(e_r, dtype=float)
    if not np.all(np.isfinite(e_r)):
        return 'record not finite'
    if noisy:
        return 'measurement noise'
    if abs(e_r[0] - 1.0) > 1e-9:
        return 'record does not start at e_0 = 1'
    if u is not None and minu is not None and maxu is not None:
        u = np.asarray(u, dtype=float)
        span = maxu - minu
        if np.any(u >= maxu - 1e-9 * span) or np.any(u <= minu + 1e-9 * span):
            return 'actuator saturated'
    return None
