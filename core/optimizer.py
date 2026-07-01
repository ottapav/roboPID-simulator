"""
PID controller optimizer: finds optimal (Kp, Ki, Kd) gains by minimizing a
weighted cost function over the closed-loop step response.

Mirrors MATLAB pidtool.pid_design / optimize_pid / optimize_pi / optimize_i.

Cost function (6-component weighted sum):
    J = w1*||e||² + w2*|e_final|² + w3*||max(e-1,0)||²
      + w4*||max(u-1,0)||² + w5*||Δe||² + w6*||Δu||²
where e = y-1 (error), u is the control action.

The optimizer uses scipy SLSQP with iteratively expanded bounds (mirrors
MATLAB fmincon loop).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, Bounds

from .pid import pid_response_linear


# Default simulation horizon multiplier
T_MULTIPLIER = 5.0

WEIGHT_FIELDS = ('e2', 'eTerm2', 'ePlus2', 'vPlus2', 'de2', 'du2')


def weights_to_array(weights) -> np.ndarray:
    """Convert dict or array weights to a length-6 numpy array."""
    if isinstance(weights, dict):
        w = np.zeros(6)
        for i, key in enumerate(WEIGHT_FIELDS):
            w[i] = float(weights.get(key, 0.0))
        return w
    return np.asarray(weights, dtype=float).ravel()[:6]


def signal_cost(y: np.ndarray, u: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the 6-component weighted cost for a simulated response.

    y: closed-loop output (step response, target = 1)
    u: control action (scaled by K so target ~ 1 at steady state)
    """
    N = len(y)
    J = np.zeros(6)
    J[0] = np.linalg.norm(y - 1.0) / N
    J[1] = abs(y[-1] - 1.0)
    J[2] = np.linalg.norm(np.maximum(y - 1.0, 0.0)) / N
    J[3] = np.linalg.norm(np.maximum(u - 1.0, 0.0)) / N
    J[4] = np.linalg.norm(np.diff(y)) / N
    J[5] = np.linalg.norm(np.diff(u)) / N
    return float(J @ weights)


def pid_cost(tau, K, Td, Cp, Ci, Cd, T, Ts, weights, dtype='y') -> float:
    """Simulate and evaluate cost for given (Cp, Ci, Cd) normalized gains."""
    y, u, _, _, _, _ = pid_response_linear(
        tau, K, Td, Cp, Ci, Cd, T, Ts, corr_type=True, dtype=dtype)
    return signal_cost(y, u * K, weights)


def _run_bounded_optimize(cost_fn, C_init: np.ndarray) -> np.ndarray:
    """
    Iteratively tighten bounds around the optimum (mirrors MATLAB fmincon loop).

    Starts from C_init, expands bounds 0.2x–5x around current estimate,
    stops when the optimum is well inside the bounds or after 10 iterations.
    """
    C = C_init.copy()
    for _ in range(10):
        lb = 0.2 * C
        ub = 5.0 * C
        bounds = Bounds(lb, ub)
        result = minimize(cost_fn, C, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-9})
        C = np.clip(result.x, lb, ub)
        if np.all(C * 1.05 < ub) and np.all(C / 1.05 > lb):
            break
    return C


def optimize_pid(tau, K, Td, T, Ts, weights, dtype='y') -> tuple[float, float, float]:
    """Optimize PID (Kp, Ki, Kd) normalized gains."""
    w = weights_to_array(weights)

    def cost(C):
        return pid_cost(tau, K, Td, C[0], C[1], C[2], T, Ts, w, dtype)

    C = _run_bounded_optimize(cost, np.ones(3))
    return float(C[0]), float(C[1]), float(C[2])


def optimize_pi(tau, K, Td, T, Ts, weights) -> tuple[float, float, float]:
    """Optimize PI (Kp, Ki) normalized gains."""
    w = weights_to_array(weights)

    def cost(C):
        return pid_cost(tau, K, Td, C[0], C[1], 0.0, T, Ts, w, 'y')

    C = _run_bounded_optimize(cost, np.ones(2))
    return float(C[0]), float(C[1]), 0.0


def optimize_i(tau, K, Td, T, Ts, weights) -> tuple[float, float, float]:
    """Optimize I-only gain."""
    w = weights_to_array(weights)

    def cost(C):
        return pid_cost(tau, K, Td, 0.0, C[0], 0.0, T, Ts, w, 'y')

    C = _run_bounded_optimize(cost, np.ones(1))
    return 0.0, float(C[0]), 0.0


def pid_design(tau, K, Td, Ts, ctype: str = 'PID',
               weights=None, dtype: str = 'y',
               T: float | None = None) -> tuple[float, float, float]:
    """
    Main entry point: design a PID/PI/I controller for the given plant.

    Returns (Kp, Ki, Kd) physical gains (not normalized).

    The optimizer works in normalized gain space (Cp, Ci, Cd) and converts
    back to physical (Kp, Ki, Kd) via the scaling:
        Kp = (1/K) * Cp
        Ki = (1/K) / (sum(tau)+Td) * Ci
        Kd = (1/K) * (sum(tau)+Td) / 8 * Cd
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if weights is None:
        weights = {'e2': 1.0, 'du2': 5.0}
    if T is None:
        T = T_MULTIPLIER * (float(np.sum(tau)) + Td)

    ctype = ctype.upper()
    if ctype == 'PID':
        Cp, Ci, Cd = optimize_pid(tau, K, Td, T, Ts, weights, dtype)
    elif ctype == 'PI':
        Cp, Ci, Cd = optimize_pi(tau, K, Td, T, Ts, weights)
    else:  # 'I'
        Cp, Ci, Cd = optimize_i(tau, K, Td, T, Ts, weights)

    # Convert from normalized to physical gains (corr_type scaling)
    Kr = 1.0 / K
    tau_i = float(np.sum(tau)) + Td
    Kp = Kr * Cp
    Ki = Kr / tau_i * Ci
    Kd = Kr * tau_i / 8.0 * Cd
    return Kp, Ki, Kd
