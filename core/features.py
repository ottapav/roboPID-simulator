"""
Performance feature computation using encirclement-based phase metrics.

Three features track how many times normalized signal trajectories wrap
around the origin in different phase planes — a Nyquist-like robustness metric.

Mirrors MATLAB pidtool: encirc, extended_features, standard_pid_features.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .signals import loop_signals, add_derivatives, pathratio

EPSILON = 0.1   # disc radius for encirclement detection


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
    limit: float
    x_rel_to_end: bool
    y_rel_to_end: bool
    x0: list[float] = field(default_factory=lambda: [0.0])
    y0: list[float] = field(default_factory=lambda: [0.0])


def standard_pid_features(limits=(0.5, 0.75, 1.00)) -> list[FeatureDescription]:
    """Return the three standard PID feature descriptors."""
    return [
        FeatureDescription(
            name='e E', xdata='e', ydata='e',
            xname='e', yname='E',
            signum=1, x_deg=0, y_deg=-1,
            limit=limits[0],
            x_rel_to_end=False, y_rel_to_end=True,
        ),
        FeatureDescription(
            name="e' e", xdata='e', ydata='e',
            xname='Δe', yname='e',
            signum=1, x_deg=1, y_deg=0,
            limit=limits[1],
            x_rel_to_end=False, y_rel_to_end=False,
        ),
        FeatureDescription(
            name="e'' e'", xdata='e', ydata='e',
            xname='Δ²e', yname='Δe',
            signum=1, x_deg=2, y_deg=1,
            limit=limits[2],
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

    Equivalent to MATLAB pidtool.encirc. Returns a float (can be fractional
    for partial encirclements).
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


def compute_features(description: list[FeatureDescription],
                     ext_signals: dict,
                     k1: int, k2: int) -> list[dict]:
    """
    Compute phase encirclement for each feature descriptor.

    Returns a list of dicts with keys: name, phase, xdata, ydata, xname, yname, limit.
    """
    nd = 2  # fixed derivative order used in add_derivatives
    results = []

    for desc in description:
        xs_mat = ext_signals.get(desc.xdata)
        ys_mat = ext_signals.get(desc.ydata)

        if xs_mat is None or ys_mat is None or not isinstance(xs_mat, np.ndarray):
            results.append({'name': desc.name, 'phase': 0.0,
                            'xdata': np.array([0.0]), 'ydata': np.array([0.0]),
                            'xname': desc.xname, 'yname': desc.yname,
                            'limit': desc.limit})
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
        if mx < 1e-12 or my < 1e-12:
            phase = 0.0
            xw_norm = xw
            yw_norm = yw
        else:
            xw_norm = xw / mx
            yw_norm = yw / my
            phase = -np.inf
            for x0, y0 in zip(desc.x0, desc.y0):
                phase = max(phase, desc.signum * encirc(
                    xw_norm - x0, yw_norm - y0, EPSILON))

        results.append({
            'name': desc.name,
            'phase': float(phase),
            'xdata': xw_norm,
            'ydata': yw_norm,
            'xname': desc.xname,
            'yname': desc.yname,
            'limit': desc.limit,
        })

    return results


def loop_response_features(description, pr_names, tau, K, Td, Ts,
                           Kp, Ki, Kd, dtype='y', T=None,
                           simtype=0, minu=-1.0, maxu=1.0,
                           dist_a=0.0, dist_b=0.0):
    """
    Full pipeline: simulate → signals → derivatives → features + path ratios.

    Returns (features, k1, k2, signals, pr) where pr is the path-ratio dict.
    """
    sigs = loop_signals(tau, K, Td, Ts, Kp, Ki, Kd, dtype, T,
                        simtype, minu, maxu, dist_a, dist_b)
    k1 = sigs['k1']
    k2 = sigs['k2']

    ext = add_derivatives(sigs, nd=2)
    features = compute_features(description, ext, k1, k2)
    pr = pathratio(pr_names, sigs, k1, k2)

    return features, k1, k2, sigs, pr
