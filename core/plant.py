"""
Plant model for generalized FOPTD process:

    P(s) = K * exp(-Td*s) / prod(s*tau_i + 1)

Discretized exactly as MATLAB pidtool.plantmodel:
  - First pole:  K*(1-p1) / (z^nd * (z - p1))
  - Each extra:  (1-pi)*z / (z - pi)
where p_i = exp(-Ts/tau_i), nd = round(Td/Ts).
"""

from __future__ import annotations
import numpy as np
from scipy.signal import dlsim, dlti


def plant_tf(tau: np.ndarray, K: float, Td: float, Ts: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (num, den) arrays of the discrete plant TF (coefficient polynomials
    in descending powers of z).

    Full plant: num(z) / den(z) where den already encodes the delay.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    nd = int(round(Td / Ts))

    p0 = float(np.exp(-Ts / tau[0]))
    num = np.array([K * (1.0 - p0)])   # scalar numerator
    den = np.array([1.0, -p0])          # (z - p0)

    for i in range(1, len(tau)):
        pi = float(np.exp(-Ts / tau[i]))
        # MATLAB: multiply by tf([1-p, 0], [1, -p]) → adds zero at z=0
        num = np.polymul(num, [1.0 - pi, 0.0])
        den = np.polymul(den, [1.0, -pi])

    # Dead-time delay: append nd zeros to denominator (multiply by z^nd)
    if nd > 0:
        den = np.concatenate([den, np.zeros(nd)])

    return num, den


def plant_step_response(tau: np.ndarray, K: float, Td: float,
                        T: float, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Open-loop unit-step response of the plant.

    Returns (y, t) where t = [0, Ts, 2*Ts, ..., T].
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    t = np.arange(0.0, T + Ts * 0.5, Ts)
    N = len(t)

    num, den = plant_tf(tau, K, Td, Ts)
    sys = dlti(num, den, dt=Ts)
    u = np.ones(N)
    _, y = dlsim(sys, u, t=t)
    return y.ravel(), t
