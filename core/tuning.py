"""
Iterative PID gain tuning via the triangular rule of
docs/JPC26_basic/main.tex (Table "the tuning rule at a glance", eq.
"triangular"):
- Fp, Fi, Fd are multipliers applied on top of the base gains
- Each iteration evaluates the stability screen and the three Gamma
  encirclement counts and nudges gains by one gamma = 1/(1-beta) notch
- An unstable record cuts every gain at once, far more coarsely than the
  gamma notch the count-based rows use, and deepening with the band's
  frequency (Ki /2, Kp /4, Kd /8); otherwise the lowest violated band is
  cut and every band below it is raised; if none is violated, all bands
  are raised
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from .features import (
    loop_response_features, FeatureDescription, DELTA, EPSILON,
    encirc, EncirclementMetric,
)
from .signals import find_index, StabilityScreen


# Table 2's rightmost column, keyed by which row fired this iteration.
MANUAL_READING = {
    'unstable': 'runaway: back off hard',
    'N0': 'slow cycling: less reset',
    'N1': 'ringing: less gain',
    'N2': 'buzzing: less rate',
    'none': 'all quiet: tighten',
}


# Swappable gain-update decisions: given the current features/gains, decide
# the next multipliers and which MANUAL_READING row fired. Register a new
# key here to make an alternative rule selectable via
# pid_tuning(..., rule=<fn>) without editing the iteration loop below.
TuningRule = Callable[
    [list[dict], tuple[float, float, float], bool,
     float, float, float, float,
     float, float, float, float, float, float],
    tuple[float, float, float, str]
]


def triangular_rule(
    feats: list[dict],
    Nbar: tuple[float, float, float],
    unstable: bool,
    Fp_cur: float, Fi_cur: float, Fd_cur: float,
    gamma: float,
    Fp_min: float, Fp_max: float,
    Fi_min: float, Fi_max: float,
    Fd_min: float, Fd_max: float,
) -> tuple[float, float, float, str]:
    """
    Table 2's triangular tuning rule (paper eq. "triangular").

    feats must be the 3-element list produced against standard_pid_features()
    ordering: feats[0]=Gamma0 (indicts Ki), feats[1]=Gamma1 (indicts Kp),
    feats[2]=Gamma2 (indicts Kd). Returns (Fp_new, Fi_new, Fd_new, row); row
    must be a MANUAL_READING key, since callers such as callbacks.py's
    on_iteration dereference MANUAL_READING[row] directly with no fallback.
    """
    Fp_new, Fi_new, Fd_new = Fp_cur, Fi_cur, Fd_cur

    if unstable:
        # Record grows rather than decays. Every gain comes down at once, and
        # far more coarsely than the gamma notch the count-based rows below
        # use -- "instability is a state to be exited quickly, not corrected
        # delicately."
        #
        # The cut deepens with the band's frequency: Ki /2, Kp /4, Kd /8. A
        # runaway is driven by the fast bands, so the derivative channel is
        # pulled back hardest and the reset -- which sets the settling time
        # the user is actually after -- is disturbed least. This is the one
        # row that fires without a usable count to attribute the trouble to,
        # so the attribution is made a priori by band instead.
        row = 'unstable'
        Fi_new = max(Fi_cur * 0.5, Fi_min)
        Fp_new = max(Fp_cur * 0.25, Fp_min)
        Fd_new = max(Fd_cur * 0.125, Fd_min)

    elif feats[0]['N'] > Nbar[0]:
        row = 'N0'
        Fi_new = max(Fi_cur / gamma, Fi_min)

    elif feats[1]['N'] > Nbar[1]:
        row = 'N1'
        Fi_new = min(Fi_cur * gamma, Fi_max)
        Fp_new = max(Fp_cur / gamma, Fp_min)

    elif feats[2]['N'] > Nbar[2]:
        row = 'N2'
        Fi_new = min(Fi_cur * gamma, Fi_max)
        Fp_new = min(Fp_cur * gamma, Fp_max)
        Fd_new = max(Fd_cur / gamma, Fd_min)

    else:
        # All features within limits: raise all bands.
        row = 'none'
        Fi_new = min(Fi_cur * gamma, Fi_max)
        Fp_new = min(Fp_cur * gamma, Fp_max)
        Fd_new = min(Fd_cur * gamma, Fd_max)

    return Fp_new, Fi_new, Fd_new, row


TUNING_RULES: dict[str, TuningRule] = {
    'triangular': triangular_rule,
}


def pid_tuning(
    description: list[FeatureDescription],
    tau, K: float, L: float, Ts: float,
    Kp: float, Ki: float, Kd: float,
    dtype: str = 'y',
    Tsim: float | None = None,
    n_iter: int = 200,
    Fp_limits: tuple[float, float] = (0.01, 10.0),
    Fi_limits: tuple[float, float] = (0.01, 10.0),
    Fd_limits: tuple[float, float] = (0.01, 10.0),
    Nbar: tuple[float, ...] | None = None,
    beta: float = 0.1,
    simtype: int = 0,
    minu: float = -1.0, maxu: float = 1.0,
    dist_a: float = 0.0, dist_b: float = 0.0,
    delta: float = DELTA,
    eps: float = EPSILON,
    on_iteration: Callable[
        [int, int, float, float, float, str, list[dict], dict], None] | None = None,
    rule: TuningRule = triangular_rule,
    stability_screen: StabilityScreen = find_index,
    metric: EncirclementMetric = encirc,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run n_iter iterations of the given tuning rule (default: the triangular rule).

    Returns (Fp_hist, Fi_hist, Fd_hist): history of multipliers for each iteration.
    The final multipliers are the last elements.

    Nbar: tuple of 3 loop limits N̄0, N̄1, N̄2 (one per feature); defaults to
          the Nbar stored in each FeatureDescription.
    beta: step size (paper's β); the applied notch is gamma = 1/(1-beta).
    delta: settling-band guard fraction of peak error (Definition 4).
    eps: truncation disc radius for the encirclement count (Definition 1).
    on_iteration: optional callback invoked as
                  on_iteration(i, n_iter, Fp_cur, Fi_cur, Fd_cur, row, feats,
                  sigs) once per iteration with the pre-update multipliers, the
                  MANUAL_READING key for the row that fired, and the features
                  and signal dict this iteration was actually scored on — so a
                  caller that wants to display the iteration does not have to
                  re-simulate it. Invoked once more after the loop with
                  (n_iter, n_iter, <final multipliers>, row, feats, sigs).
    rule: swappable gain-update decision, see TuningRule/TUNING_RULES.
    stability_screen: swappable instability test, see StabilityScreen/STABILITY_SCREENS.
    metric: swappable trajectory-scoring algorithm used by the per-iteration
            feature evaluation, see EncirclementMetric/ENCIRCLEMENT_METRICS.
    seed: makes a run with output noise reproducible. Each iteration is given
          its own Generator derived from this seed, so iteration i always sees
          the same noise realization no matter how many iterations run --
          without it, every iteration draws a fresh realization and the same
          gains score differently each time they are visited.
    """
    if Nbar is None:
        Nbar = tuple(d.Nbar for d in description)

    n_iter = max(n_iter, 10)
    gamma = 1.0 / (1.0 - beta)
    Fp_min, Fp_max = min(Fp_limits), max(Fp_limits)
    Fi_min, Fi_max = min(Fi_limits), max(Fi_limits)
    Fd_min, Fd_max = min(Fd_limits), max(Fd_limits)

    Fp_hist = np.clip(np.ones(n_iter), Fp_min, Fp_max)
    Fi_hist = np.clip(np.ones(n_iter), Fi_min, Fi_max)
    Fd_hist = np.clip(np.ones(n_iter), Fd_min, Fd_max)

    row = 'none'
    seed_seq = None if seed is None else np.random.SeedSequence(seed)

    for i in range(1, n_iter):
        Fp_cur = Fp_hist[i - 1]
        Fi_cur = Fi_hist[i - 1]
        Fd_cur = Fd_hist[i - 1]

        feats, k1, k2, sigs = loop_response_features(
            description,
            tau, K, L, Ts,
            Fp_cur * Kp, Fi_cur * Ki, Fd_cur * Kd,
            dtype=dtype, Tsim=Tsim,
            simtype=simtype, minu=minu, maxu=maxu,
            dist_a=dist_a, dist_b=dist_b, delta=delta, eps=eps,
            metric=metric,
            rng=(None if seed_seq is None
                 else np.random.default_rng(seed_seq.spawn(1)[0])),
        )

        _, unstable = stability_screen(k1, k2, sigs['e'])

        Fp_new, Fi_new, Fd_new, row = rule(
            feats, Nbar, unstable, Fp_cur, Fi_cur, Fd_cur, gamma,
            Fp_min, Fp_max, Fi_min, Fi_max, Fd_min, Fd_max,
        )

        if on_iteration is not None:
            on_iteration(i, n_iter, float(Fp_cur), float(Fi_cur), float(Fd_cur),
                         row, feats, sigs)

        Fp_hist[i] = Fp_new
        Fi_hist[i] = Fi_new
        Fd_hist[i] = Fd_new

    if on_iteration is not None:
        on_iteration(n_iter, n_iter, float(Fp_hist[-1]), float(Fi_hist[-1]),
                     float(Fd_hist[-1]), row, feats, sigs)

    return Fp_hist, Fi_hist, Fd_hist
