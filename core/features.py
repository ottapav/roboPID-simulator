"""
Encirclement-based phase-plane features: the three "Pachner plots" of
docs/JPC26_basic/main.tex (Gamma0, Gamma1, Gamma2 in eq. "plots"). Each
trajectory's winding number about the origin (Definition 1) is a
dimensionless, scale-free, band-selective damping diagnostic — N0 indicts
Ki, N1 indicts Kp, N2 indicts Kd (Section "The encirclement features").
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from .signals import loop_signals, add_derivatives

EPSILON = 0.1   # truncation disc radius (Definition 1); paper default
DELTA = 0.02    # settling-band guard (Definition 4); paper default


@dataclass
class FeatureDescription:
    name: str
    xdata: str       # signal name in signals dict
    ydata: str       # signal name in signals dict
    xname: str       # axis label
    yname: str       # axis label
    signum: int      # sign of encirclement count
    x_deg: int       # 0=value, 1=diff, 2=2nd-diff, -1=cumsum, -2=2nd-cumsum
    y_deg: int
    Nbar: float
    x_rel_to_end: bool
    y_rel_to_end: bool
    x0: list[float] = field(default_factory=lambda: [0.0])
    y0: list[float] = field(default_factory=lambda: [0.0])


def standard_pid_features(Nbar=(0.5, 0.75, 1.00)) -> list[FeatureDescription]:
    """Return the three Pachner-plot descriptors Gamma0, Gamma1, Gamma2 (eq. "plots")."""
    return [
        FeatureDescription(
            name='Gamma0', xdata='e', ydata='e',
            xname='e', yname='E',
            signum=1, x_deg=0, y_deg=-1,
            Nbar=Nbar[0],
            x_rel_to_end=False, y_rel_to_end=True,
        ),
        FeatureDescription(
            name='Gamma1', xdata='e', ydata='e',
            xname='Δe', yname='e',
            signum=1, x_deg=1, y_deg=0,
            Nbar=Nbar[1],
            x_rel_to_end=False, y_rel_to_end=False,
        ),
        FeatureDescription(
            name='Gamma2', xdata='e', ydata='e',
            xname='Δ²e', yname='Δe',
            signum=1, x_deg=2, y_deg=1,
            Nbar=Nbar[2],
            x_rel_to_end=False, y_rel_to_end=False,
        ),
    ]


def _deg_to_col(deg: int, nd: int) -> int:
    """
    Convert a degree index to a column index in the extended signal matrix.

    Columns: [0=signal, 1..nd=diffs, nd+1..2*nd=cumsums]
    Negative deg means cumulative sum: -1 → col nd+1, -2 → col nd+2, etc.
    """
    if deg >= 0:
        return deg
    return nd + (-deg)


def find_disc_entrypoint(x: np.ndarray, y: np.ndarray, eps: float) -> int:
    """
    Find the last index (0-based) where the trajectory enters the eps-disc.

    Searches backwards from the end. Returns len(x) if never found (i.e., the
    trajectory never enters the disc, so we use the full length).
    """
    n = len(x)
    if n < 2:
        return n

    a = np.stack([x[:-1], y[:-1]], axis=1)
    b = np.stack([x[1:], y[1:]], axis=1)
    ab = b - a
    ab2 = np.sum(ab ** 2, axis=1)
    safe_ab2 = np.where(ab2 == 0.0, 1.0, ab2)
    t = np.where(ab2 == 0.0, 1.0,
                np.clip(-np.sum(a * ab, axis=1) / safe_ab2, 0.0, 1.0))
    c = a + ab * t[:, None]
    dist = np.linalg.norm(c, axis=1)

    hits = np.nonzero(dist <= eps)[0]
    return int(hits.max()) if hits.size else n


def encirc(x: np.ndarray, y: np.ndarray, eps: float = EPSILON) -> float:
    """
    Compute the signed encirclement count of the trajectory around the origin.

    Implements Definition 1 (Encirclement count) of docs/JPC26_basic/main.tex:
    per-axis peak normalization, truncation at the last entry into the
    epsilon-disc, then winding number as the endpoint angle difference
    corrected by signed crossings of the negative horizontal semi-axis.
    Returns a float (partial revolutions count fractionally).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    mx = np.max(np.abs(x))
    my = np.max(np.abs(y))
    if mx < 1e-12 or my < 1e-12:
        return 0.0

    x = x / mx
    y = y / my

    j = min(find_disc_entrypoint(x, y, eps), len(x) - 1)

    if j <= 1:
        return 0.0

    # Angle swept from start to j
    angle = np.arctan2(y[j], x[j]) - np.arctan2(y[0], x[0])

    # Count negative-x-axis crossings (full winding counts)
    xp = x[:j - 1]
    xn = x[1:j]
    yp = y[:j - 1]
    yn = y[1:j]

    in_neg_x = (xp < 0) & (xn < 0)
    n1 = float(np.sum(in_neg_x & (yp < 0) & (yn >= 0)))   # CCW crossing
    n2 = float(np.sum(in_neg_x & (yp >= 0) & (yn < 0)))   # CW crossing

    return angle / (2.0 * np.pi) - n1 + n2


# Swappable trajectory-scoring algorithms: (x, y, eps) -> N. Register a new
# key here to make an alternative metric selectable via
# compute_features(..., metric=<fn>) / loop_response_features(..., metric=<fn>)
# without editing this module.
EncirclementMetric = Callable[[np.ndarray, np.ndarray, float], float]

ENCIRCLEMENT_METRICS: dict[str, EncirclementMetric] = {
    'winding_number': encirc,
}


def compute_features(description: list[FeatureDescription],
                     ext_signals: dict,
                     k1: int, k2: int,
                     eps: float = EPSILON,
                     metric: EncirclementMetric = encirc,
                     nd: int = 2) -> list[dict]:
    """
    Compute the encirclement count N for each feature descriptor.

    nd must match the order the extended signal matrix was built with, since
    it is what maps a descriptor's x_deg/y_deg onto a column.

    Returns a list of dicts with keys: name, N, xdata, ydata, xname, yname, Nbar.
    """
    results = []

    for desc in description:
        xs_mat = ext_signals.get(desc.xdata)
        ys_mat = ext_signals.get(desc.ydata)

        if xs_mat is None or ys_mat is None or not isinstance(xs_mat, np.ndarray):
            results.append({'name': desc.name, 'N': 0.0,
                            'xdata': np.array([0.0]), 'ydata': np.array([0.0]),
                            'xname': desc.xname, 'yname': desc.yname,
                            'Nbar': desc.Nbar})
            continue

        if xs_mat.ndim == 1:
            xs_mat = xs_mat[:, np.newaxis]
        if ys_mat.ndim == 1:
            ys_mat = ys_mat[:, np.newaxis]

        ix = _deg_to_col(desc.x_deg, nd)
        iy = _deg_to_col(desc.y_deg, nd)

        x = xs_mat[:, ix].ravel()
        y = ys_mat[:, iy].ravel()

        if desc.x_rel_to_end:
            x = x - x[k2]
        if desc.y_rel_to_end:
            y = y - y[k2]

        xw = x[k1:k2 + 1]
        yw = y[k1:k2 + 1]

        mx = np.max(np.abs(xw))
        my = np.max(np.abs(yw))
        if not (np.all(np.isfinite(xw)) and np.all(np.isfinite(yw))):
            # A diverged run. Scoring it 0.0 -- which is what the peak
            # normalization below produces from inf, since xw/inf is 0 or nan
            # and the winding count of that is nothing -- reads as the quietest
            # possible trajectory, and the tuning rule answers "all quiet" by
            # raising the very gains that blew up. inf rather than nan because
            # every band test is `N > Nbar`: inf makes any rule in TUNING_RULES
            # cut a gain, while nan compares False and lands back on the same
            # "all quiet" branch. find_index is the primary guard (it reports
            # unstable, which triangular_rule checks first); this keeps the
            # feature value itself from lying to a rule that reads it directly.
            #
            # Guarded here rather than inside encirc because desc.signum is
            # applied to the metric's result: signum is +1 for all three
            # standard descriptors, but a -1 descriptor would turn a
            # maximally-bad inf into a maximally-good -inf. Short-circuiting
            # before the call also covers any swapped-in metric=.
            N = float('inf')
            xw_norm = xw
            yw_norm = yw
        elif mx < 1e-12 or my < 1e-12:
            N = 0.0
            xw_norm = xw
            yw_norm = yw
        else:
            xw_norm = xw / mx
            yw_norm = yw / my
            N = -np.inf
            for x0, y0 in zip(desc.x0, desc.y0):
                N = max(N, desc.signum * metric(
                    xw_norm - x0, yw_norm - y0, eps))

        results.append({
            'name': desc.name,
            'N': float(N),
            'xdata': xw_norm,
            'ydata': yw_norm,
            'xname': desc.xname,
            'yname': desc.yname,
            'Nbar': desc.Nbar,
        })

    return results


def loop_response_features(description, tau, K, L, Ts,
                           Kp, Ki, Kd, dtype='y', Tsim=None,
                           simtype=0, minu=-1.0, maxu=1.0,
                           dist_a=0.0, dist_b=0.0, delta=DELTA,
                           eps=EPSILON, metric: EncirclementMetric = encirc,
                           nd: int = 2, rng=None):
    """
    Full pipeline: simulate → signals → derivatives → encirclement features.

    The encirclement windows are truncated at the settling-anchored k_delta
    (Definition 4), not the raw k2, so the winding counts stay window-length
    independent (Proposition "well-posedness"). Returns (features, k1, k2, signals).
    """
    sigs = loop_signals(tau, K, L, Ts, Kp, Ki, Kd, dtype, Tsim,
                        simtype, minu, maxu, dist_a, dist_b, delta, rng)
    k1 = sigs['k1']
    k2 = sigs['k2']
    k_delta = sigs['k_delta']

    # Only the signals the descriptors actually name get differentiated and
    # integrated. The standard three all read 'e', so augmenting the whole
    # 11-signal dict was building ten unused N x (2*nd+1) matrices per call.
    named = {d.xdata for d in description} | {d.ydata for d in description}
    ext = add_derivatives({k: v for k, v in sigs.items() if k in named}, nd=nd)
    features = compute_features(description, ext, k1, k_delta, eps, metric, nd)

    return features, k1, k2, sigs
