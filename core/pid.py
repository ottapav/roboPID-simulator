"""
PID simulation functions.

Two modes mirror MATLAB pidtool:
  - Linear (simtype=0): closed-loop TF via polynomial arithmetic + dlsim
  - Anti-windup discrete (simtype=1): sample-by-sample loop with saturation

Gain scaling (corr_type=True) mirrors MATLAB's normalized parameterization:
    Ki = (1/K) / (sum(tau)+L) * gi
    Kp = (1/K) * gp
    Kd = (1/K) * (sum(tau)+L) / 8 * gd
"""

from __future__ import annotations
import numpy as np
from scipy.signal import dlsim, dlti, lfilter
import scipy.linalg

from .plant import plant_tf


K1_PADDING = 5
K2_PADDING = 5


def _scale_gains(tau, K, L, gp, gi, gd, corr_type: bool):
    """Apply normalized → physical gain conversion."""
    if corr_type:
        Kr = 1.0 / K
        tau_i = float(np.sum(tau)) + L
        Kp = Kr * gp
        Ki = Kr / tau_i * gi
        Kd = Kr * tau_i / 8.0 * gd
    else:
        Kp, Ki, Kd = gp, gi, gd
    return float(Kp), float(Ki), float(Kd)


def _controller_tf(Kp: float, Ki: float, Kd: float, dtype: str
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete controller TF coefficients.

    dtype='y': derivative on output  C(z) = (Ki+Kp) - Kp*z^-1  over (1 - z^-1)
               → tf([Ki+Kp, -Kp], [1, -1])
    dtype='e': derivative on error   C(z) full PID
               → tf([Ki+Kp+Kd, -(Kp+2Kd), Kd], [1, -1, 0])
    """
    if dtype == 'y':
        num_c = np.array([Ki + Kp, -Kp])
        den_c = np.array([1.0, -1.0])
    else:
        num_c = np.array([Ki + Kp + Kd, -(Kp + 2 * Kd), Kd])
        den_c = np.array([1.0, -1.0, 0.0])
    return num_c, den_c


def _strip_leading_zeros(num: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Remove leading coefficients that are essentially zero."""
    arr = np.asarray(num, dtype=float)
    first = np.argmax(np.abs(arr) > tol)
    return arr[first:] if len(arr) > 0 else arr


def _negative_feedback_cl(num_c, den_c, num_p, den_p):
    """
    Closed-loop under negative unit feedback: L/(1+L) where L = C*P.

    Returns num_cl, den_cl of the closed-loop TF.
    """
    num_l = np.polymul(num_c, num_p)
    den_l = np.polymul(den_c, den_p)
    n = max(len(num_l), len(den_l))
    num_l = np.concatenate([np.zeros(n - len(num_l)), num_l])
    den_l = np.concatenate([np.zeros(n - len(den_l)), den_l])
    num_cl = _strip_leading_zeros(num_l)
    den_cl = _strip_leading_zeros(den_l + num_l)
    return num_cl, den_cl


def _sensitivity_tf(num_c, den_c, num_p, den_p):
    """Control action TF: u = C/(1+CP)*r."""
    num_l = np.polymul(num_c, num_p)
    den_l = np.polymul(den_c, den_p)
    n = max(len(num_l), len(den_l))
    num_l = np.concatenate([np.zeros(n - len(num_l)), num_l])
    den_l = np.concatenate([np.zeros(n - len(den_l)), den_l])
    num_u = _strip_leading_zeros(np.polymul(num_c, den_p))
    den_u = _strip_leading_zeros(den_l + num_l)
    return num_u, den_u


def pid_response_linear(tau, K, L, gp, gi, gd, T_sim, Ts,
                        corr_type: bool = False, dtype: str = 'y',
                        dist_a: float = 0.0, dist_b: float = 0.0):
    """
    Simulate closed-loop step response using linear TF arithmetic.

    Returns (y, u, t, Kp, Ki, Kd).
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    Kp, Ki, Kd = _scale_gains(tau, K, L, gp, gi, gd, corr_type)

    T_sim = max(T_sim, 10 + K1_PADDING + K2_PADDING)
    t = np.arange(0.0, T_sim + Ts * 0.5, Ts)
    N = len(t)
    u_in = np.ones(N)

    num_p, den_p = plant_tf(tau, K, L, Ts)

    if dtype == 'y':
        # Derivative on output: D(z) = Kd*(1 - z^-1) = Kd*[1,-1]/[1,0]
        # Inner-loop plant: P_mod = P / (1 + D*P)
        # = (den_D * num_P) / (den_D * den_P + num_D * num_P)
        num_d = np.array([Kd, -Kd])
        den_d = np.array([1.0, 0.0])
        num_p_mod = np.polymul(den_d, num_p)
        den_p_mod_a = np.polymul(den_d, den_p)
        den_p_mod_b = np.polymul(num_d, num_p)
        n2 = max(len(den_p_mod_a), len(den_p_mod_b))
        den_p_mod_a = np.concatenate([np.zeros(n2 - len(den_p_mod_a)), den_p_mod_a])
        den_p_mod_b = np.concatenate([np.zeros(n2 - len(den_p_mod_b)), den_p_mod_b])
        den_p_mod = _strip_leading_zeros(den_p_mod_a + den_p_mod_b)
        num_p_mod = _strip_leading_zeros(num_p_mod)
    else:
        num_p_mod, den_p_mod = num_p, den_p

    # PI part of controller (for dtype='y' Kd is already in inner loop)
    if dtype == 'y':
        num_c = np.array([Ki + Kp, -Kp])
        den_c = np.array([1.0, -1.0])
    else:
        num_c = np.array([Ki + Kp + Kd, -(Kp + 2 * Kd), Kd])
        den_c = np.array([1.0, -1.0, 0.0])

    num_cl, den_cl = _negative_feedback_cl(num_c, den_c, num_p_mod, den_p_mod)
    num_u, den_u = _sensitivity_tf(num_c, den_c, num_p_mod, den_p_mod)

    sys_y = dlti(num_cl, den_cl, dt=Ts)
    sys_u = dlti(num_u, den_u, dt=Ts)

    _, y = dlsim(sys_y, u_in, t=t)
    _, u = dlsim(sys_u, u_in, t=t)

    if dist_b != 0.0:
        rng = np.random.default_rng()
        w = rng.standard_normal(N)
        y = y.ravel() + lfilter([dist_b], [1.0, -dist_a], w)
    else:
        y = y.ravel()

    return y, u.ravel(), t, Kp, Ki, Kd


def pid_response_awup(tau, K, L, gp, gi, gd, T_sim, Ts,
                      corr_type: bool = False, dtype: str = 'y',
                      minu: float = -1.0, maxu: float = 1.0,
                      dist_a: float = 0.0, dist_b: float = 0.0):
    """
    Anti-windup (saturation clamping) discrete-time PID simulation.

    State-space plant discretized via matrix exponential.
    Returns (y, u, t, Kp, Ki, Kd).
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    Kp, Ki, Kd = _scale_gains(tau, K, L, gp, gi, gd, corr_type)

    nd = int(round(L / Ts))
    ns = len(tau)

    # Build continuous-time state matrix for cascade of first-order systems
    # States: [u_input, x1, x2, ..., xns]
    # x_dot_i = -x_i/tau_i + x_{i-1}/tau_i
    M_cont = np.zeros((1 + ns, 1 + ns))
    for i in range(1, 1 + ns):
        M_cont[i, i] = -1.0 / tau[i - 1]
        M_cont[i, i - 1] = 1.0 / tau[i - 1]

    M = scipy.linalg.expm(M_cont * Ts)
    A = M[1:, 1:]       # ns×ns plant state matrix
    B = M[1:, 0]        # ns×1 input vector

    N = 1 + int(np.ceil(T_sim / Ts))
    y = np.zeros(N)
    u_out = np.zeros(N)
    t = np.arange(N) * Ts

    s = np.zeros(ns)          # plant state
    Y = np.zeros(1 + nd)      # output delay buffer (Y[nd] is current output)
    ys = np.zeros(3)          # last 3 output samples
    rs = np.array([0.0, 0.0, 1.0])  # last 3 reference samples (step at n=0)

    U = 0.0
    dist = 0.0
    rng = np.random.default_rng()

    for n in range(N):
        y[n] = Y[nd] + dist
        dist = dist_a * dist + dist_b * rng.standard_normal()

        es = ys - rs
        e = es[2]
        de = es[2] - es[1]
        if dtype == 'y':
            dde = ys[2] - 2.0 * ys[1] + ys[0]
        else:
            dde = es[2] - 2.0 * es[1] + es[0]

        U = U - (de * Kp + e * Ki + dde * Kd)
        U = float(np.clip(U, minu, maxu))

        s = A @ s + B * U
        # Shift output delay buffer: newest plant output goes to position 0
        Y = np.concatenate([[K * s[ns - 1]], Y[:nd]])

        ys = np.array([ys[1], ys[2], Y[nd]])
        rs = np.array([rs[1], rs[2], 1.0])

        u_out[n] = U

    return y, u_out, t, Kp, Ki, Kd


def action_components(y: np.ndarray, Kp: float, Ki: float, Kd: float,
                      Ts: float, T_sim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose total control action into P, I, D contributions.

    Mirrors MATLAB action_components using lfilter (dlsim equivalent).
    """
    t = np.arange(0.0, T_sim + Ts * 0.5, Ts)
    N = len(t)
    y_trim = y[:N]
    e = y_trim - 1.0

    uP = -Kp * e
    uI = lfilter([Ki, 0.0], [1.0, -1.0], -e)
    uD = lfilter([Kd, -Kd], [1.0, 0.0], -e)

    return uP, uI, uD
