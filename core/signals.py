"""
Signal utilities: assembling loop signals dict, scaling, and derivative augmentation.

Mirrors MATLAB pidtool static methods:
  loop_signals_struct, scaled_variables, add_derivatives, pathratio, getsignal
"""

from __future__ import annotations
import numpy as np

from .plant import plant_step_response
from .pid import pid_response_linear, pid_response_awup, action_components

K1_PADDING = 5
K2_PADDING = 5


def loop_signals(tau, K, Td, Ts, Kp, Ki, Kd, dtype: str = 'y',
                 T: float | None = None,
                 simtype: int = 0,
                 minu: float = -1.0, maxu: float = 1.0,
                 dist_a: float = 0.0, dist_b: float = 0.0) -> dict:
    """
    Compute and return a dict of all loop signals.

    Keys: 'y', 'u', 'e', 'v', 'uP', 'uI', 'uD', 'vP', 'vI', 'vD', 't', 'y_plant'
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if T is None:
        T = float(np.ceil(5.0 * (np.sum(tau) + Td)))

    y_plant, _ = plant_step_response(tau, K, Td, T, Ts)

    if simtype == 0:
        y, u, t, _, _, _ = pid_response_linear(
            tau, K, Td, Kp, Ki, Kd, T, Ts, corr_type=False, dtype=dtype)
    else:
        y, u, t, _, _, _ = pid_response_awup(
            tau, K, Td, Kp, Ki, Kd, T, Ts, corr_type=False, dtype=dtype,
            minu=minu, maxu=maxu, dist_a=dist_a, dist_b=dist_b)

    r = np.ones_like(y)
    e, v, k1, k2 = scaled_variables(y, u, r)

    uP, uI, uD = action_components(y, Kp, Ki, Kd, Ts, T)
    N = len(t)
    uP, uI, uD = uP[:N], uI[:N], uD[:N]
    y_plant = y_plant[:N]

    vP = uP
    vI = uI - uI[k2]
    vD = uD

    return {
        'y': y, 'u': u, 'e': e, 'v': v,
        'uP': uP, 'uI': uI, 'uD': uD,
        'vP': vP, 'vI': vI, 'vD': vD,
        't': t, 'y_plant': y_plant,
        'k1': k1, 'k2': k2,
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


def pathratio(names: list[str], signals: dict, k1: int, k2: int) -> dict:
    """
    Compute path ratio for each named signal: activity in second half vs first half.

    ratio = sum|Δ signal[mid:k2]| / sum|Δ signal[k1:mid]|

    A ratio > 1 means the signal is still active late — indicates sluggish tuning.
    """
    mid = (k1 + k2) // 2
    result = {}
    for name in names:
        s = signals.get(name)
        if s is None:
            result[name] = 0.0
            continue
        if isinstance(s, np.ndarray) and s.ndim == 2:
            s = s[:, 0]
        s = np.asarray(s, dtype=float).ravel()
        second_half = np.sum(np.abs(np.diff(s[mid:k2 + 1])))
        first_half = np.sum(np.abs(np.diff(s[k1:mid + 1])))
        result[name] = float(second_half / first_half) if first_half > 1e-12 else 0.0
    return result
