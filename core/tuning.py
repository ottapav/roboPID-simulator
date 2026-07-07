"""
Iterative PID gain tuning via feature-space search.

Mirrors MATLAB pidtool.pid_tuning exactly:
- Fp, Fi, Fd are multipliers applied on top of the base optimized gains
- Each iteration evaluates closed-loop features and path ratios
- Gains are nudged up or down depending on which feature limit is violated
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from .features import loop_response_features, FeatureDescription
from .signals import find_index


PR_NAMES = ['uI', 'uP', 'uD']
PR_LIMIT = float(1.5)   # path ratio limits (inf → disabled, as in MATLAB)


def pid_tuning(
    description: list[FeatureDescription],
    tau, K: float, Td: float, Ts: float,
    Kp: float, Ki: float, Kd: float,
    dtype: str = 'y',
    T: float | None = None,
    N: int = 200,
    Fp_limits: tuple[float, float] = (0.01, 5.0),
    Fi_limits: tuple[float, float] = (0.01, 5.0),
    Fd_limits: tuple[float, float] = (0.01, 5.0),
    feature_limits: tuple[float, ...] | None = None,
    step: float = 0.1,
    simtype: int = 0,
    minu: float = -1.0, maxu: float = 1.0,
    dist_a: float = 0.0, dist_b: float = 0.0,
    on_iteration: Callable[[int, int, float, float, float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run N iterations of feature-driven gain search.

    Returns (Fp_hist, Fi_hist, Fd_hist): history of multipliers for each iteration.
    The final multipliers are the last elements.

    feature_limits: tuple of 3 phase limits (one per feature); defaults to
                    the limit stored in each FeatureDescription.
    on_iteration: optional callback invoked as on_iteration(i, N, Fp_cur, Fi_cur, Fd_cur)
                  once per iteration with the pre-update multipliers, plus once more
                  after the loop with (N, N, <final multipliers>).
    """
    if feature_limits is None:
        feature_limits = tuple(d.limit for d in description)

    N = max(N, 10)
    Fp_min, Fp_max = min(Fp_limits), max(Fp_limits)
    Fi_min, Fi_max = min(Fi_limits), max(Fi_limits)
    Fd_min, Fd_max = min(Fd_limits), max(Fd_limits)

    Fp_hist = np.clip(np.ones(N), Fp_min, Fp_max)
    Fi_hist = np.clip(np.ones(N), Fi_min, Fi_max)
    Fd_hist = np.clip(np.ones(N), Fd_min, Fd_max)

    for i in range(1, N):
        Fp_cur = Fp_hist[i - 1]
        Fi_cur = Fi_hist[i - 1]
        Fd_cur = Fd_hist[i - 1]

        if on_iteration is not None:
            on_iteration(i, N, float(Fp_cur), float(Fi_cur), float(Fd_cur))

        feats, k1, k2, sigs, pr = loop_response_features(
            description, PR_NAMES,
            tau, K, Td, Ts,
            Fp_cur * Kp, Fi_cur * Ki, Fd_cur * Kd,
            dtype=dtype, T=T,
            simtype=simtype, minu=minu, maxu=maxu,
            dist_a=dist_a, dist_b=dist_b,
        )

        pr_uI = pr.get('uI', 0.0)
        pr_uP = pr.get('uP', 0.0)
        pr_uD = pr.get('uD', 0.0)
        _, unstable = find_index(k1, k2, sigs['e'])

        Fp_new = Fp_cur
        Fi_new = Fi_cur
        Fd_new = Fd_cur

        if unstable:
            # Error variance is growing late in the window rather than settling:
            # back off all three gains together instead of nudging just one.
            Fp_new = max(Fp_cur * 0.5, Fp_min)
            Fi_new = max(Fi_cur * 0.5, Fi_min)
            Fd_new = max(Fd_cur * 0.5, Fd_min)

        elif pr_uI > PR_LIMIT or pr_uP > PR_LIMIT or pr_uD > PR_LIMIT:
            # Path ratio exceeded: halve the offending gain
            if pr_uI > PR_LIMIT:
                Fi_new = max(Fi_cur * 0.5, Fi_min)
            if pr_uP > PR_LIMIT:
                Fp_new = max(Fp_cur * 0.5, Fp_min)
            if pr_uD > PR_LIMIT:
                Fd_new = max(Fd_cur * 0.5, Fd_min)

        elif feats[0]['phase'] > feature_limits[0] and Fi_cur > Fi_min:
            Fi_new = max(Fi_cur * (1.0 - step), Fi_min)

        elif feats[1]['phase'] > feature_limits[1] and Fp_cur > Fp_min:
            Fp_new = max(Fp_cur * (1.0 - step), Fp_min)
            Fi_new = min(Fi_cur / (1.0 - step), Fi_max)

        elif feats[2]['phase'] > feature_limits[2] and Fd_cur > Fd_min:
            Fd_new = max(Fd_cur * (1.0 - step), Fd_min)
            Fp_new = min(Fp_cur / (1.0 - step), Fp_max)
            Fi_new = min(Fi_cur / (1.0 - step), Fi_max)

        else:
            # All features within limits: increase all gains
            Fi_new = min(Fi_cur / (1.0 - step), Fi_max)
            Fp_new = min(Fp_cur / (1.0 - step), Fp_max)
            Fd_new = min(Fd_cur / (1.0 - step), Fd_max)

        Fp_hist[i] = Fp_new
        Fi_hist[i] = Fi_new
        Fd_hist[i] = Fd_new

    if on_iteration is not None:
        on_iteration(N, N, float(Fp_hist[-1]), float(Fi_hist[-1]), float(Fd_hist[-1]))

    return Fp_hist, Fi_hist, Fd_hist
