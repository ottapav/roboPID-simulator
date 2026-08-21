"""
Plant model for generalized FOPTD process:

    P(s) = K * exp(-L*s) / prod(s*tau_i + 1)

Discretized exactly as MATLAB pidtool.plantmodel:
  - First pole:  K*(1-p1) / (z^nd * (z - p1))
  - Each extra:  (1-pi)*z / (z - pi)
where p_i = exp(-Ts/tau_i), nd = round(L/Ts).
"""

from __future__ import annotations
import numpy as np

from .params import time_grid


def plant_tf(tau: np.ndarray, K: float, L: float, Ts: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (num, den) arrays of the discrete plant TF (coefficient polynomials
    in descending powers of z).

    Full plant: num(z) / den(z) where den already encodes the delay.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    nd = int(round(L / Ts))

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


def plant_step_response(tau: np.ndarray, K: float, L: float,
                        Tsim: float, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Open-loop unit-step response of the plant.

    Returns (y, t) where t = [0, Ts, 2*Ts, ..., Tsim].
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    t = time_grid(Tsim, Ts)
    N = len(t)

    from .pid import _tf_step   # local import: pid imports plant_tf from here

    num, den = plant_tf(tau, K, L, Ts)
    return _tf_step(num, den, np.ones(N)), t
