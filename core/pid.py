"""
PID simulation functions.

Two modes mirror MATLAB pidtool:
  - Linear (simtype=0): closed-loop TF via polynomial arithmetic, stepped with
    lfilter (see _tf_step)
  - Anti-windup discrete (simtype=1): sample-by-sample loop with saturation

Gains are physical, i.e. continuous-time: Kp is dimensionless, Ki is in 1/s and
Kd in s, so the Ts factors of RoboPID_JPC_paper/main.tex's difference equation

    du_k = Kp*(e_k - e_{k-1}) + Ki*Ts*e_k + (Kd/Ts)*(e_k - 2e_{k-1} + e_{k-2})

appear explicitly wherever the controller is realized (_controller_tf,
pid_response_awup, action_components). This is what makes a gain mean the same
loop on any grid: the simulator now chooses Ts per plant and lets the user
override it, and per-sample gains would have quietly retuned the controller
every time either moved.

Gain scaling (corr_type=True) mirrors MATLAB's normalized parameterization:
    Ki = (1/K) / (sum(tau)+L) * gi
    Kp = (1/K) * gp
    Kd = (1/K) * (sum(tau)+L) / 8 * gd
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter
import scipy.linalg

from .params import DERIV_FILTER_N, time_grid
from .plant import plant_tf


def _deriv_tf(Kd: float, Ts: float, N: float = DERIV_FILTER_N
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete filtered derivative D(z), in descending powers of z.

        D(s) = Kd*s / (1 + Tf*s),   Tf = Kd / N

    discretized by backward Euler (s -> (1 - z^-1)/Ts, the same substitution
    the P and I channels already use), which gives

        D(z) = Kd*(z - 1) / ((Ts + Tf)*z - Tf)

    and reduces to the unfiltered (Kd/Ts)*(1 - z^-1) exactly when Tf = 0.

    The filter is what keeps the derivative implementable at small Ts: without
    it |D| at the Nyquist frequency is 2*Kd/Ts and grows without bound as the
    grid is refined, which for derivative-on-output puts an inner-loop pole at
    -Kd*K/tau. See DERIV_FILTER_N in core.params for the full argument.

    Backward Euler rather than Tustin because it maps the whole left half plane
    inside the unit circle: the filter pole Tf/(Ts+Tf) is in [0, 1) for every
    Kd >= 0 and Ts > 0, so the roll-off can never itself ring.
    """
    Tf = Kd / N if N > 0 else 0.0
    return np.array([Kd, -Kd]), np.array([Ts + Tf, -Tf])


def _tf_step(num, den, u: np.ndarray) -> np.ndarray:
    """
    Zero-state response of num(z)/den(z) to the input u.

    This is what dlsim(dlti(num, den, dt=Ts), u, t) computes, about 300x
    faster and, measured against an exact rational recursion, marginally more
    accurate (3.6e-9 vs 6.3e-9 max error on the closed-loop TF of the P2
    plant, whose response peaks around 1e4).

    The one subtlety is the coefficient convention: dlti reads num/den as
    descending powers of z, while lfilter reads them as powers of z^-1. For a
    strictly proper TF the numerator must therefore be left-padded with zeros
    to len(den) — without the pad the two disagree by orders of magnitude.
    """
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    if num.size < den.size:
        num = np.concatenate([np.zeros(den.size - num.size), num])
    return lfilter(num, den, u)


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


def _controller_tf(Kp: float, Ki: float, Kd: float, dtype: str, Ts: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete controller TF coefficients, with Ki and Kd in physical units:
    integration accumulates Ki*Ts per sample, differentiation goes through the
    filtered _deriv_tf.

    dtype='y': derivative on output — Kd is handled by the caller's inner loop,
               so only PI is left here.
               C(z) = (Ki*Ts+Kp) - Kp*z^-1  over (1 - z^-1)
               → tf([Ki*Ts+Kp, -Kp], [1, -1])
    dtype='e': derivative on error — full PID, PI + D over a common denominator.
               C(z) = [(Ki*Ts+Kp) - Kp*z^-1]/(1 - z^-1) + D(z)
    """
    Ki_d = Ki * Ts
    if dtype == 'y':
        return np.array([Ki_d + Kp, -Kp]), np.array([1.0, -1.0])

    # PI over (z - 1), plus the filtered D over its own pole: sum them onto the
    # common denominator rather than restating the unfiltered three-tap form,
    # so 'e' and 'y' cannot drift apart in how the derivative is realized.
    num_pi, den_pi = np.array([Ki_d + Kp, -Kp]), np.array([1.0, -1.0])
    num_d, den_d = _deriv_tf(Kd, Ts)
    num_c = np.polyadd(np.polymul(num_pi, den_d), np.polymul(num_d, den_pi))
    return num_c, np.polymul(den_pi, den_d)


def _strip_leading_zeros(num: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Remove leading coefficients that are essentially zero."""
    arr = np.asarray(num, dtype=float)
    first = np.argmax(np.abs(arr) > tol)
    return arr[first:] if len(arr) > 0 else arr


def _loop_tf(num_c, den_c, num_p, den_p):
    """
    Open-loop L = C*P, zero-padded to a common length, plus the closed-loop
    denominator 1 + L that both closed-loop transfer functions share.

    Returns (num_l, den_l, den_cl).
    """
    num_l = np.polymul(num_c, num_p)
    den_l = np.polymul(den_c, den_p)
    n = max(len(num_l), len(den_l))
    num_l = np.concatenate([np.zeros(n - len(num_l)), num_l])
    den_l = np.concatenate([np.zeros(n - len(den_l)), den_l])
    return num_l, den_l, _strip_leading_zeros(den_l + num_l)


def _closed_loop_tfs(num_c, den_c, num_p, den_p):
    """
    The two closed-loop transfer functions driven by the reference step:

        y = L/(1+L) * r        (output)
        u = C/(1+L) * r        (control action)

    Computed together because they share 1 + L; splitting them meant building
    the same open-loop product twice per simulation.
    """
    num_l, _, den_cl = _loop_tf(num_c, den_c, num_p, den_p)
    num_y = _strip_leading_zeros(num_l)
    num_u = _strip_leading_zeros(np.polymul(num_c, den_p))
    return (num_y, den_cl), (num_u, den_cl)


def pid_response_linear(tau, K, L, gp, gi, gd, Tsim, Ts,
                        corr_type: bool = False, dtype: str = 'y',
                        dist_a: float = 0.0, dist_b: float = 0.0,
                        rng: np.random.Generator | None = None):
    """
    Simulate closed-loop step response using linear TF arithmetic.

    rng seeds the output-noise realization; pass one to make a noisy run
    reproducible (default: a fresh unseeded Generator, as before).

    Returns (y, u, t, Kp, Ki, Kd).
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    Kp, Ki, Kd = _scale_gains(tau, K, L, gp, gi, gd, corr_type)

    t = time_grid(Tsim, Ts)
    N = len(t)
    u_in = np.ones(N)

    num_p, den_p = plant_tf(tau, K, L, Ts)

    if dtype == 'y':
        # Derivative on output, filtered (see _deriv_tf).
        # Inner-loop plant: P_mod = P / (1 + D*P)
        # = (den_D * num_P) / (den_D * den_P + num_D * num_P)
        num_d, den_d = _deriv_tf(Kd, Ts)
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

    # For dtype='y' Kd is already folded into the inner loop above, so what is
    # left in the forward path is the PI part — which is exactly what
    # _controller_tf('y') returns.
    num_c, den_c = _controller_tf(Kp, Ki, Kd, dtype, Ts)

    (num_cl, den_cl), (num_u, den_u) = _closed_loop_tfs(
        num_c, den_c, num_p_mod, den_p_mod)

    y = _tf_step(num_cl, den_cl, u_in)
    u = _tf_step(num_u, den_u, u_in)

    if dist_b != 0.0:
        rng = np.random.default_rng() if rng is None else rng
        w = rng.standard_normal(N)
        y = y + lfilter([dist_b], [1.0, -dist_a], w)

    return y, u, t, Kp, Ki, Kd


def pid_response_awup(tau, K, L, gp, gi, gd, Tsim, Ts,
                      corr_type: bool = False, dtype: str = 'y',
                      minu: float = -1.0, maxu: float = 1.0,
                      dist_a: float = 0.0, dist_b: float = 0.0,
                      rng: np.random.Generator | None = None):
    """
    Anti-windup (saturation clamping) discrete-time PID simulation.

    State-space plant discretized via matrix exponential. rng seeds the
    output-noise realization; pass one to make a noisy run reproducible
    (default: a fresh unseeded Generator, as before).

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

    t = time_grid(Tsim, Ts)
    N = len(t)
    y = np.zeros(N)
    u_out = np.zeros(N)

    s = np.zeros(ns)          # plant state

    # Output delay line as a ring buffer: buf[p] always holds the sample
    # delayed by nd, and the slot just consumed is where the newest sample is
    # written. Equivalent to rebuilding the array with concatenate every step,
    # without the O(N*nd) copying. The last three output/reference samples are
    # plain scalars for the same reason.
    buf = np.zeros(nd + 1)
    p = 0
    ys0 = ys1 = ys2 = 0.0
    rs0, rs1, rs2 = 0.0, 0.0, 1.0   # step at n=0

    # Physical -> per-sample once, outside the loop (see module docstring).
    #
    # The derivative carries the _deriv_tf roll-off, which in this velocity
    # form makes its increment recursive rather than a plain scaled second
    # difference. Applying (1 - z^-1) to D(z) = Kd(1-z^-1)/((Ts+Tf) - Tf z^-1):
    #
    #     (Ts + Tf) * dU_D,k  -  Tf * dU_D,k-1  =  Kd * dde_k
    #
    # which collapses back to dde * Kd/Ts when Tf = 0.
    Ki_d = Ki * Ts
    Tf = Kd / DERIV_FILTER_N if DERIV_FILTER_N > 0 else 0.0
    dUd = 0.0

    U = 0.0
    dist = 0.0
    if dist_b != 0.0 and rng is None:
        rng = np.random.default_rng()

    for n in range(N):
        y[n] = buf[p] + dist
        if dist_b != 0.0:
            dist = dist_a * dist + dist_b * rng.standard_normal()

        e = ys2 - rs2
        de = (ys2 - rs2) - (ys1 - rs1)
        if dtype == 'y':
            dde = ys2 - 2.0 * ys1 + ys0
        else:
            dde = (ys2 - rs2) - 2.0 * (ys1 - rs1) + (ys0 - rs0)

        dUd = (Tf * dUd + Kd * dde) / (Ts + Tf)
        U = U - (de * Kp + e * Ki_d + dUd)
        U = float(np.clip(U, minu, maxu))

        s = A @ s + B * U
        buf[p] = K * s[ns - 1]
        p = (p + 1) % (nd + 1)

        # Read after p advances: that is the slot the old code's post-shift
        # Y[nd] referred to.
        ys0, ys1, ys2 = ys1, ys2, buf[p]
        rs0, rs1, rs2 = rs1, rs2, 1.0

        u_out[n] = U

    return y, u_out, t, Kp, Ki, Kd


def action_components(y: np.ndarray, Kp: float, Ki: float, Kd: float, Ts: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose total control action into P, I, D contributions.

    Ki and Kd are physical, so the realization carries the same Ki*Ts factor
    and the same filtered _deriv_tf as _controller_tf and pid_response_awup --
    the three channels are meant to be the ones that produced u, so a different
    derivative here would draw a D trace no controller ever applied.

    The grid comes from y itself rather than from Tsim/Ts, which stays the
    cheaper invariant now that params.time_grid keeps the two in agreement.
    """
    e = np.asarray(y, dtype=float).ravel() - 1.0
    num_d, den_d = _deriv_tf(Kd, Ts)

    uP = -Kp * e
    uI = lfilter([Ki * Ts, 0.0], [1.0, -1.0], -e)
    uD = lfilter(num_d, den_d, -e)

    return uP, uI, uD
