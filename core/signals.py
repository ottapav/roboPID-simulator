"""
Signal utilities: assembling loop signals dict, scaling, and derivative augmentation.

`settling_index` implements the settling-anchored window guard of
RoboPID_JPC_paper/main.tex (Definition 4, the delta guard restoring window
independence of the encirclement counts). `find_index` implements the
maximum-likelihood stability screen from the same paper.
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from .params import HORIZON_SPANS, K1_PADDING, K2_PADDING, N_POINTS
from .pid import pid_response_linear, pid_response_awup, action_components


def auto_grid(tau, L: float, N: int = N_POINTS) -> tuple[float, float]:
    """
    Simulation horizon and sampling period, proposed together from the plant.

    Under a fixed sample count the two are not two decisions but one. The
    horizon is set by the plant's own timescale and the period simply follows,
    so every simulation costs the same N samples and the transient occupies the
    same fraction of the record whether the plant settles in 2 s or 2000 s:

      span = sum(tau) + L
      Tsim = HORIZON_SPANS * span     long enough for the response to settle, so
                                      settling_index has something to anchor to
      Ts   = Tsim / (N - 1)           = span / 49.9 at the default N

    Both halves are load-bearing, and the GUI lets the user override either, so
    it is worth being explicit about what each one buys:

    *The horizon* must outlive the transient. Truncating it ends the record
    before the response settles, which pins settling_index's k_delta onto k2 and
    degenerates Definition 4's guard into exactly the unguarded raw-window count
    it exists to replace (test_settling_guard_active_on_a_full_record pins this).
    That is why callbacks._clamp_grid answers an unreasonable hand-entered pair
    by moving Ts and never Tsim.

    *The period* must resolve the transient past the fixed analysis padding: at
    Ts = span, a tau=1 plant settles in about as many samples as K1_PADDING +
    K2_PADDING trims away. span/49.9 is ~5x finer than the span/10 that needs,
    with the margin left over for the dead-time quantization nd = round(L/Ts).

    There is deliberately no floor on Tsim. An absolute minimum horizon would,
    under a fixed N, spend most of the record on a fast plant's dead time --
    tau=[0.1] at a 50 s floor resolves its own span with a single sample.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    span = float(np.sum(tau)) + L
    Tsim = HORIZON_SPANS * span
    return Tsim, Tsim / (N - 1)


def loop_signals(tau, K, L, Ts, Kp, Ki, Kd, dtype: str = 'y',
                 Tsim: float | None = None,
                 simtype: int = 0,
                 minu: float = -1.0, maxu: float = 1.0,
                 dist_a: float = 0.0, dist_b: float = 0.0,
                 delta: float = 0.02,
                 rng: np.random.Generator | None = None) -> dict:
    """
    Compute and return a dict of all loop signals.

    rng seeds the output-noise realization, so a noisy run can be repeated.

    Keys: 'y', 'u', 'e', 'v', 'uP', 'uI', 'uD', 'vP', 'vI', 'vD', 't'
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if Tsim is None:
        Tsim, _ = auto_grid(tau, L)

    if simtype == 0:
        y, u, t, _, _, _ = pid_response_linear(
            tau, K, L, Kp, Ki, Kd, Tsim, Ts, corr_type=False, dtype=dtype,
            dist_a=dist_a, dist_b=dist_b, rng=rng)
    else:
        y, u, t, _, _, _ = pid_response_awup(
            tau, K, L, Kp, Ki, Kd, Tsim, Ts, corr_type=False, dtype=dtype,
            minu=minu, maxu=maxu, dist_a=dist_a, dist_b=dist_b, rng=rng)

    r = np.ones_like(y)
    e, v, k1, k2 = scaled_variables(y, u, r)
    k_delta = settling_index(e, k1, k2, delta)

    uP, uI, uD = action_components(y, Kp, Ki, Kd, Ts)

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
    peak = np.max(np.abs(e[:k2 + 1]))   # slicing past the end is already safe
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
        if not isinstance(s, np.ndarray):
            result[name] = s          # k1/k2/k_delta and anything else scalar
            continue
        s = np.asarray(s, dtype=float).ravel()
        N = len(s)
        mat = np.zeros((N, 2 * nd + 1))
        mat[:, 0] = s

        for i in range(1, nd + 1):
            d = np.diff(mat[:, i - 1])
            mat[:N - 1, i] = d   # last sample stays 0

        # Each cumsum column integrates the one before it, so column nd+1 is
        # the cumsum of the signal and nd+2 the cumsum of *that*. Chaining from
        # column i-1 (rather than restarting from the difference columns) is
        # what makes deg=-2 the second integral it is documented to be.
        for i in range(nd + 1, 2 * nd + 1):
            mat[:, i] = np.cumsum(mat[:, i - 1] if i > nd + 1 else mat[:, 0])

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

    A record that has overflowed to inf/nan is reported unstable outright. The
    likelihood comparison cannot rank two non-finite segments -- every
    candidate split scores nan, and the `not np.any(np.isfinite(Plog))` fallback
    below then returned "stable", which triangular_rule reads as "all quiet" and
    answers by *raising* the gains that diverged. That made the screen weakest
    on its worst input: it correctly flags a merely-huge record (peak 1e40) and
    used to wave through an infinite one. Unlike the two `return 2, False`
    fallbacks, which mean "cannot decide", this is a decisive verdict, so it
    returns the window start -- the record was already diverging there.
    """
    M = np.asarray(M, dtype=float).ravel()

    seg_all = M[m:n + 1]
    if seg_all.size and not np.all(np.isfinite(seg_all)):
        return m, True

    # NOTE: this is max(M), not max(M**2), so for an all-negative record (an
    # overdamped response, where e = y - r never crosses zero) the "variance"
    # floor comes out negative and never engages. Preserved deliberately —
    # correcting it would move the stability screen's verdicts and therefore
    # the tuned gains, so it needs reconciling against the MATLAB original
    # rather than a local fix. It is latent in practice: e is never exactly
    # zero over a whole segment, so the floor is never the binding term.
    seg = M[m + 1:n - 1]
    minvar = (float(np.max(seg)) if seg.size
              else -float(np.finfo(np.float32).max)) / 1e3

    prefix = np.concatenate(([0.0], np.cumsum(M ** 2)))  # prefix[i] = sum(M[:i] ** 2)

    # Definition 3: splits leaving >=10% of the record on each side.
    margin = max(1, round(0.1 * (n - m + 1)))
    k = np.arange(m - 1 + margin, n - margin + 1)
    if k.size == 0:
        return 2, False

    n1, n2 = k - m + 1, n - k
    sum1, sum2 = prefix[k + 1] - prefix[m], prefix[n + 1] - prefix[k + 1]
    var1 = np.maximum(minvar, sum1 / n1)
    var2 = np.maximum(minvar, sum2 / n2)

    with np.errstate(divide='ignore', invalid='ignore'):
        Plog = (-(n1 * np.log(2 * np.pi * var1) + sum1 / var1)
                - (n2 * np.log(2 * np.pi * var2) + sum2 / var2))

    # The scalar form skipped non-finite candidates implicitly: `Llog < nan` is
    # False, so they never won. -inf reproduces that, and an all-non-finite
    # window falls back to the initial values the scalar loop would have kept.
    Plog = np.where(np.isfinite(Plog), Plog, -np.inf)
    if not np.any(np.isfinite(Plog)):
        return 2, False

    # argmax returns the FIRST maximum, matching the strict `Llog < Plog`
    # comparison, which only advanced on a strict improvement.
    best = int(np.argmax(Plog))
    return int(k[best]), bool(var2[best] > var1[best])


# Swappable instability-detection algorithms: (m, n, M) -> (ind, unstable).
# Register a new key here to make an alternative screen selectable via
# pid_tuning(..., stability_screen=<fn>) without editing tuning.py.
StabilityScreen = Callable[[int, int, np.ndarray], tuple[int, bool]]

STABILITY_SCREENS: dict[str, StabilityScreen] = {
    'bayesian_changepoint': find_index,
}
