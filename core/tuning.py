"""
Iterative PID gain tuning via the triangular rule of
RoboPID_JPC_paper/main.tex (Table "the tuning rule at a glance", eq.
"triangular"):
- Fp, Fi, Fd are multipliers applied on top of the base gains
- Each iteration evaluates the stability screen and the three Gamma
  encirclement counts and nudges gains by one gamma = 1/(1-beta) notch
- An unstable record halves every gain (a coarser cut than the gamma
  notch used by the count-based rows); otherwise the lowest violated
  band is cut and every band below it is raised; if none is violated, all
  bands are raised
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from .features import loop_response_features, FeatureDescription, DELTA, EPSILON
from .signals import find_index


# Table 2's rightmost column, keyed by which row fired this iteration.
MANUAL_READING = {
    'unstable': 'runaway: halve all',
    'N0': 'slow cycling: less reset',
    'N1': 'ringing: less gain',
    'N2': 'buzzing: less rate',
    'none': 'all quiet: tighten',
}


def pid_tuning(
    description: list[FeatureDescription],
    tau, K: float, L: float, Ts: float,
    Kp: float, Ki: float, Kd: float,
    dtype: str = 'y',
    T_sim: float | None = None,
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
    on_iteration: Callable[[int, int, float, float, float, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run n_iter iterations of the triangular tuning rule.

    Returns (Fp_hist, Fi_hist, Fd_hist): history of multipliers for each iteration.
    The final multipliers are the last elements.

    Nbar: tuple of 3 loop limits N̄0, N̄1, N̄2 (one per feature); defaults to
          the Nbar stored in each FeatureDescription.
    beta: step size (paper's β); the applied notch is gamma = 1/(1-beta).
    delta: settling-band guard fraction of peak error (Definition 4).
    eps: truncation disc radius for the encirclement count (Definition 1).
    on_iteration: optional callback invoked as
                  on_iteration(i, n_iter, Fp_cur, Fi_cur, Fd_cur, row) once per
                  iteration with the pre-update multipliers and the
                  MANUAL_READING key for the row that fired, plus once more
                  after the loop with (n_iter, n_iter, <final multipliers>, row).
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

    for i in range(1, n_iter):
        Fp_cur = Fp_hist[i - 1]
        Fi_cur = Fi_hist[i - 1]
        Fd_cur = Fd_hist[i - 1]

        feats, k1, k2, sigs = loop_response_features(
            description,
            tau, K, L, Ts,
            Fp_cur * Kp, Fi_cur * Ki, Fd_cur * Kd,
            dtype=dtype, T_sim=T_sim,
            simtype=simtype, minu=minu, maxu=maxu,
            dist_a=dist_a, dist_b=dist_b, delta=delta, eps=eps,
        )

        _, unstable = find_index(k1, k2, sigs['e'])

        Fp_new = Fp_cur
        Fi_new = Fi_cur
        Fd_new = Fd_cur

        if unstable:
            # Record grows rather than decays: coarse halving (Table 2,
            # unstable row: "Downarrow" = divide by 2), deliberately
            # cruder than the gamma notch used by the count-based rows
            # below -- "instability is a state to be exited quickly, not
            # corrected delicately."
            row = 'unstable'
            Fp_new = max(Fp_cur * 0.5, Fp_min)
            Fi_new = max(Fi_cur * 0.5, Fi_min)
            Fd_new = max(Fd_cur * 0.5, Fd_min)

        elif feats[0]['N'] > Nbar[0]:
            row = 'N0'
            Fi_new = max(Fi_cur / gamma, Fi_min)

        elif feats[1]['N'] > Nbar[1]:
            row = 'N1'
            Fp_new = max(Fp_cur / gamma, Fp_min)
            Fi_new = min(Fi_cur * gamma, Fi_max)

        elif feats[2]['N'] > Nbar[2]:
            row = 'N2'
            Fd_new = max(Fd_cur / gamma, Fd_min)
            Fp_new = min(Fp_cur * gamma, Fp_max)
            Fi_new = min(Fi_cur * gamma, Fi_max)

        else:
            # All features within limits: raise all bands.
            row = 'none'
            Fi_new = min(Fi_cur * gamma, Fi_max)
            Fp_new = min(Fp_cur * gamma, Fp_max)
            Fd_new = min(Fd_cur * gamma, Fd_max)

        if on_iteration is not None:
            on_iteration(i, n_iter, float(Fp_cur), float(Fi_cur), float(Fd_cur), row)

        Fp_hist[i] = Fp_new
        Fi_hist[i] = Fi_new
        Fd_hist[i] = Fd_new

    if on_iteration is not None:
        on_iteration(n_iter, n_iter, float(Fp_hist[-1]), float(Fi_hist[-1]), float(Fd_hist[-1]), row)

    return Fp_hist, Fi_hist, Fd_hist
