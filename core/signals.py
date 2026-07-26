"""
Signal utilities: assembling loop signals dict, scaling, and derivative augmentation.

`settling_index` implements the settling-anchored window guard of
RoboPID_JPC_paper/main.tex (Definition 4, the delta guard restoring window
independence of the encirclement counts). `find_index` implements the
maximum-likelihood stability screen from the same paper.
"""

from __future__ import annotations
import numpy as np

from .pid import pid_response_linear, pid_response_awup, action_components

K1_PADDING = 5
K2_PADDING = 5


def min_sim_time(tau, L: float) -> float:
    """
    Minimum simulation time (in the same units as tau/L) guaranteeing enough
    samples for the K1_PADDING/K2_PADDING analysis window, regardless of how
    small tau/L are.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    return max(10.0 * (float(np.sum(tau)) + L), 50.0)


def loop_signals(tau, K, L, Ts, Kp, Ki, Kd, dtype: str = 'y',
                 T_sim: float | None = None,
                 simtype: int = 0,
                 minu: float = -1.0, maxu: float = 1.0,
                 dist_a: float = 0.0, dist_b: float = 0.0,
                 delta: float = 0.02) -> dict:
    """
    Compute and return a dict of all loop signals.

    Keys: 'y', 'u', 'e', 'v', 'uP', 'uI', 'uD', 'vP', 'vI', 'vD', 't'
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if T_sim is None:
        T_sim = min_sim_time(tau, L)

    if simtype == 0:
        y, u, t, _, _, _ = pid_response_linear(
            tau, K, L, Kp, Ki, Kd, T_sim, Ts, corr_type=False, dtype=dtype)
    else:
        y, u, t, _, _, _ = pid_response_awup(
            tau, K, L, Kp, Ki, Kd, T_sim, Ts, corr_type=False, dtype=dtype,
            minu=minu, maxu=maxu, dist_a=dist_a, dist_b=dist_b)

    r = np.ones_like(y)
    e, v, k1, k2 = scaled_variables(y, u, r)
    k_delta = settling_index(e, k1, k2, delta)

    uP, uI, uD = action_components(y, Kp, Ki, Kd, Ts, T_sim)
    N = len(t)
    uP, uI, uD = uP[:N], uI[:N], uD[:N]

    vP = uP
    vI = uI - uI[k2]
    vD = uD

    return {
        'y': y, 'u': u, 'e': e, 'v': v,
        'uP': uP, 'uI': uI, 'uD': uD,
        'vP': vP, 'vI': vI, 'vD': vD,
        't': t,
        'k1': k1, 'k2': k2, 'k_delta': k_delta,
    }


def scaled_variables(y: np.ndarray, u: np.ndarray, r: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute normalized error e and control v, plus window indices k1, k2.

    k2: last index minus padding (endpoint of analysis window).
    k1: first index where e >= -0.9 (plant has risen to ~10% of setpoint) + padding.
    """
    e = y - r
    k2 = len(y) - 1 - K2_PADDING
    v = (u - u[k2]) / (abs(u[k2]) + 0.01)

    idxs = np.where(e >= -0.9)[0]
    if len(idxs) > 0:
        k1 = int(idxs[0]) + K1_PADDING
    else:
        k1 = k2
    k1 = min(k1, k2)
    return e, v, k1, k2


def settling_index(e: np.ndarray, k1: int, k2: int, delta: float = 0.02) -> int:
    """
    Settling-anchored window guard (paper Definition 4).

    k_delta = 1 + last index where |e_k| exceeds delta * peak|e|, clipped to
    [k1, k2]. Truncating the encirclement windows here keeps the winding
    counts independent of how long the simulation happens to run, since a
    quiet micro-oscillation past this point would otherwise dominate the
    per-axis-normalized differenced coordinates and wind indefinitely.
    """
    e = np.asarray(e, dtype=float).ravel()
    peak = np.max(np.abs(e[:k2 + 1])) if k2 + 1 <= len(e) else np.max(np.abs(e))
    if peak < 1e-12:
        return k1

    band = delta * peak
    idxs = np.nonzero(np.abs(e[:k2 + 1]) > band)[0]
    k_delta = (int(idxs.max()) + 1) if idxs.size else 0
    return int(np.clip(k_delta, k1, k2))


def add_derivatives(signals: dict, nd: int = 2) -> dict:
    """
    Augment each signal with nd forward differences and nd cumulative sums.

    Extended matrix columns for signal s:
      col 0: s
      col 1..nd: successive np.diff (Δs, Δ²s, ...)
      col nd+1..2*nd: successive np.cumsum (∫s, ∫∫s, ...)

    Returns a new dict with the same keys; values are 2D arrays (N × (2*nd+1)).
    """
    result = {}
    for name, s in signals.items():
        if name in ('k1', 'k2') or not isinstance(s, np.ndarray):
            result[name] = s
            continue
        s = np.asarray(s, dtype=float).ravel()
        N = len(s)
        mat = np.zeros((N, 2 * nd + 1))
        mat[:, 0] = s

        for i in range(1, nd + 1):
            d = np.diff(mat[:, i - 1])
            mat[:N - 1, i] = d   # last sample stays 0

        j = 0
        for i in range(nd + 1, 2 * nd + 1):
            mat[:, i] = np.cumsum(mat[:, j])
            j = i - nd

        result[name] = mat
    return result


def find_index(m: int, n: int, M: np.ndarray) -> tuple[int, bool]:
    """
    Maximum-likelihood change-point search over M[m:n+1] (Definition 3).

    Splits the window at each candidate index k into an early segment M[m:k]
    and a late segment M[k+1:n], each modeled as zero-mean Gaussian, and picks
    the k maximizing their combined log-likelihood, restricted to splits that
    leave at least 10% of the record on each side. Returns (ind, unstable):
    unstable is True when, at the best split, the late segment's variance
    exceeds the early segment's — the signal is getting noisier/more active
    over time rather than settling.

    Implements the stability screen of RoboPID_JPC_paper/main.tex (Section
    "A stability screen"): a settling response front-loads its energy, a
    diverging one back-loads it, and the verdict is scale- and time-invariant
    since both mean squares scale identically.
    """
    M = np.asarray(M, dtype=float)
    minvar = -float(np.finfo(np.float32).max)
    Llog = -float(np.finfo(np.float32).max)
    ind = 2
    unstable = False

    for k in range(m + 1, n - 1):
        if M[k] > minvar:
            minvar = M[k]
    minvar = minvar / 1e3

    prefix = np.concatenate(([0.0], np.cumsum(M ** 2)))  # prefix[i] = sum(M[:i] ** 2)

    # Definition 3: n ranges over splits leaving >=10% of the record on
    # each side.
    L = n - m + 1
    margin = max(1, round(0.1 * L))
    k_lo = m - 1 + margin
    k_hi = n - margin

    for k in range(k_lo, k_hi + 1):
        sum1M2 = float(prefix[k + 1] - prefix[m])
        var1 = max(minvar, sum1M2 / (k - m + 1))
        P1log = -(k - m + 1) * np.log(2 * np.pi * var1) - sum1M2 / var1

        sum2M2 = float(prefix[n + 1] - prefix[k + 1])
        var2 = max(minvar, sum2M2 / (n - k))
        P2log = -(n - k) * np.log(2 * np.pi * var2) - sum2M2 / var2

        Plog = P1log + P2log

        if Llog < Plog:
            Llog = Plog
            ind = k
            unstable = var2 > var1

    return ind, unstable
