"""
Can a regularized winding index stand in for the published hard count?

The ECC draft analyses the rule through N_eps = (1/2pi) * integral of
(x dy - y dx) / (rho^2 + eps^2), i.e. the winding integrand weighted by
rho^2 / (rho^2 + eps^2). On the current algorithm that index leads to other,
more aggressive destinations than the hard count the rule runs
(ORBIT_UNIQUENESS_NOTES.md). This script tests the family

    N_{eps,p} = (1/2pi) * integral of (x dy - y dx)/rho^2 * rho^p/(rho^p + eps^p),

continuous in the gains for every finite p (p = 2 is the draft's index), whose
weight tends to the indicator of |rho| > eps as p grows, on the delta-guarded
window [k1, k_delta] and on the full window [k1, k2].

Two measurements per plant, with the current rule (core.tuning) and the app's
defaults (as orbit_uniqueness.py):
  agreement    the share of the states visited by the hard-count runs (27
               starts) at which N_{eps,p} makes the same move as the hard count;
  destination  Ki, M_s of the terminal orbits reached with N_{eps,p} in place of
               the hard count, against those of the hard count.

Run from the repository root:  python docs/scripts/regindex_sharpness.py
Writes docs/ECC27_limit_cycle/regindex_sharpness.json.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import orbit_uniqueness as O                                          # noqa: E402
from core.features import compute_features, EPSILON                  # noqa: E402
from core.signals import loop_signals, add_derivatives, find_index    # noqa: E402

P_VALUES = (2, 16)


def encirc_p(p):
    def metric(x, y, eps=EPSILON):
        x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
        mx, my = np.max(np.abs(x)), np.max(np.abs(y))
        if mx < 1e-12 or my < 1e-12 or len(x) < 3:
            return 0.0
        x, y = x / mx, y / my
        xm, ym = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
        r2 = xm * xm + ym * ym
        w = r2 ** (p / 2) / (r2 ** (p / 2) + eps ** p)
        with np.errstate(divide='ignore', invalid='ignore'):
            ang = np.where(r2 > 0, (xm * np.diff(y) - ym * np.diff(x)) / r2, 0.0)
        return float(np.sum(ang * w) / (2 * np.pi))
    return metric


class PlantP(O.Plant):
    def __init__(self, name, p, window):
        super().__init__(name, 'hard', window)
        self.metric, self.hard = encirc_p(p), False


def visited_states(P_hard, starts):
    pts = []
    for th0 in starts:
        th = np.clip(th0, O.LO, O.HI)
        for _ in range(200):
            pts.append(th.copy())
            th, _ = O.step(P_hard, th)
    return np.unique(np.round(np.array(pts), 9), axis=0)


def destinations(P, starts):
    out = []
    for th0 in starts:
        orb, _ = O.run_fixed(P, th0)
        if orb is None:
            continue
        out.append((float(np.exp(orb.mean(axis=0))[0]), max(P.Ms(t) for t in orb)))
    return np.array(out)


if __name__ == '__main__':
    res = {}
    for name in O.PLANTS:
        Ph = O.Plant(name, 'hard', 'guard')
        c0 = O.run_fixed(Ph, np.zeros(3))[0].mean(axis=0)
        starts = [c0 + np.log(8) * np.array(s) for s in itertools.product((-1, 0, 1), repeat=3)]
        pts = visited_states(Ph, starts)
        hard_moves = [O.step(Ph, t)[1] for t in pts]
        dh = destinations(Ph, starts)
        res[name] = {'hard': {'Ki': [dh[:, 0].min(), dh[:, 0].max()],
                              'Ms': [dh[:, 1].min(), dh[:, 1].max()]}, 'n_states': len(pts)}
        print(f'{name}: {len(pts)} states; hard count: Ki {dh[:,0].min():.4f}-{dh[:,0].max():.4f}, '
              f'worst M_s on orbit {dh[:,1].min():.2f}-{dh[:,1].max():.2f}')
        for window in ('guard', 'full'):
            for p in P_VALUES:
                Pp = PlantP(name, p, window)
                agree = np.mean([O.step(Pp, t)[1] == m for t, m in zip(pts, hard_moves)])
                d = destinations(Pp, starts)
                res[name][f'{window}/p{p}'] = {'agreement': float(agree),
                                              'Ki': [d[:, 0].min(), d[:, 0].max()],
                                              'Ms': [d[:, 1].min(), d[:, 1].max()]}
                print(f'   {window:5s} p={p:2d}: same move {100*agree:5.1f}%   '
                      f'Ki {d[:,0].min():.4f}-{d[:,0].max():.4f}, worst M_s {d[:,1].min():.2f}-{d[:,1].max():.2f}')
    path = ROOT / 'docs' / 'ECC27_limit_cycle' / 'regindex_sharpness.json'
    path.write_text(json.dumps(res, indent=1, default=float))
    print('wrote', path)
