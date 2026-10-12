"""
Theorem 1 (certified bound on M_s from one finite setpoint-step record) and the
exact discrete loop of the simulator, used to confirm it.

Record: e_0..e_N after a unit setpoint step at n = 0 (e_{-1} = 0), a_n = e_n - e_{n-1}.
Exact identity (Lemma 3):  S_d(e^{j th}) = sum_{n>=0} a_n e^{-j th n}.
Assumption 1: tail variation  sum_{n>N} |a_n| <= tau.
Theorem 1:
  lower  = max_k |S_N(th_k)| - tau                                 <= M_s
  upper  = max( max_k |S_N(th_k)| + h J / 2 + tau ,  1 + B / sin(th1/2) )  >= M_s
  with grid th_k = k h on [0, th1], J = sum |n - n_c| |a_n| (Lipschitz constant of |S_N|),
  B = sum_{n=1}^{N-1} |a_{n+1} - a_n| + |a_N| + 2 tau (Abel bound for th >= th1).
"""
import math
import numpy as np
from scipy.signal import czt


def increments(e):
    return np.diff(np.concatenate([[0.0], np.asarray(e, float)]))


def S_grid(a, th1, M):
    """S_N on th_k = k*th1/(M-1), k = 0..M-1, by the chirp z-transform (exact sums)."""
    h = th1 / (M - 1)
    return czt(a, m=M, w=np.exp(-1j * h), a=1.0), h


def certificate(e, tau, eta=2e-3, Mhf=None, Mmax=4_000_000):
    """Return dict(lower, upper, read, th_p, th1, parts) of Theorem 1.
    eta: target size of the grid term h*J/2.  Mhf: level the high-frequency bound must reach
    (th1 is the smallest frequency where 1 + B/sin(th/2) <= Mhf); default: the read value."""
    a = increments(e); N = len(a) - 1
    B = np.sum(abs(np.diff(a[1:]))) + abs(a[-1]) + 2 * tau
    w = abs(a); cw = np.cumsum(w); nc = int(np.searchsorted(cw, cw[-1] / 2))
    J = float(np.sum(abs(np.arange(N + 1) - nc) * w))
    # first pass: coarse read to set the high-frequency level
    if Mhf is None:
        Sc, hc = S_grid(a, math.pi, 20000); Mhf = float(abs(Sc).max())
    th1 = math.pi if B >= Mhf - 1 else 2 * math.asin(B / (Mhf - 1))
    M = int(min(Mmax, max(2001, math.ceil(th1 * J / (2 * eta)) + 1)))
    S, h = S_grid(a, th1, M)
    k = int(np.argmax(abs(S))); read = float(abs(S[k]))
    grid_term = h * J / 2
    up_grid = read + grid_term + tau
    up_hf = 1 + B / math.sin(th1 / 2) if th1 < math.pi else 0.0
    return dict(lower=read - tau, upper=max(up_grid, up_hf), read=read, th_p=k * h, th1=th1,
                grid_term=grid_term, up_hf=up_hf, B=float(B), J=J, M=M, N=N)


# --- the simulator's loop, exactly (spin_cert_run.sim) --------------------------------------
def C_d(Ki, Kp, Kd, dt, z):
    """discrete PID of sim(): forward-Euler integral, filtered derivative with Tf = Kd/(10 Kp)"""
    zi = 1 / z
    C = Kp + Ki * dt * zi / (1 - zi)
    if Kd > 0:
        Tf = Kd / Kp / 10; af = math.exp(-dt / Tf)
        C = C + (Kd / Tf) * (1 - zi) / (1 - af * zi)
    return C


def P_d(K, L, lags, dt, z):
    nd = int(round(L / dt)); zi = 1 / z; P = K * zi ** nd
    for t in lags:
        aj = math.exp(-dt / t); P = P * (1 - aj) / (1 - aj * zi)
    return P


def L_d(K, L, lags, Ki, Kp, Kd, dt, th):
    z = np.exp(1j * th)
    return (1 / z) * P_d(K, L, lags, dt, z) * C_d(Ki, Kp, Kd, dt, z)


def Ms_d(K, L, lags, Ki, Kp, Kd, dt, n=200000):
    """sup over [0, pi] of |S_d|: dense log grid, then refinement around the maximum"""
    th = np.concatenate([[1e-9], np.logspace(-9, math.log10(math.pi), n)])
    S = abs(1 / (1 + L_d(K, L, lags, Ki, Kp, Kd, dt, th)))
    for _ in range(3):
        i = int(np.argmax(S)); lo, hi = th[max(i - 1, 0)], th[min(i + 1, len(th) - 1)]
        th = np.linspace(lo, hi, 2001); S = abs(1 / (1 + L_d(K, L, lags, Ki, Kp, Kd, dt, th)))
    i = int(np.argmax(S))
    return float(S[i]), float(th[i])


def minReL(K, L, lags, Ki, Kp, Kd, dt, n=400000):
    th = np.logspace(-8, math.log10(math.pi), n)
    return float(np.min(L_d(K, L, lags, Ki, Kp, Kd, dt, th).real))
