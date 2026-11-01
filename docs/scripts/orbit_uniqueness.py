"""
Is the terminal orbit of fixed-step SPIN unique?

The ECC draft (Corollary 1) states that the orbit circles one triple point
theta*, N_k(theta*) = Nbar_k, fixed by the plant alone. accel_schedule.py found
fixed-step runs from different starts closing different orbits. This script
maps the terminal orbits and tests three explanations:

  H1  several roots of N_eps(theta) = Nbar (several triple points);
  H2  one root, but ill-conditioned, so orbits sit anywhere along a weak
      direction with small residual;
  H3  an artefact of the ECC grid Ts = 1 for every plant, against the app's
      own grid Ts = span/49.9 (core.signals.auto_grid);
  H4  the window: regindex.patch() swaps only the metric, so the regularized
      index is still taken over the delta-guarded window [k1, k_delta]. A jump
      of k_delta (one more lobe crossing the delta band) is then a jump of the
      index, which the ECC draft (Sec. 3) says the regularized index does not
      have. 'full' takes it over [k1, k2] instead, as Sec. 3 defines it.

For each plant and grid: fixed-step runs (published rule, regularized index,
dtype='y', box [0.01,10]) from a 3x3x3 grid of starts in log-gain space around
the unit-gain orbit, each until the first exact state revisit; orbit centroids
are clustered (distance < 2h), every cluster is refined to a root by least
squares, and the residual and Jacobian singular values are reported.

Run from the repository root:  python docs/scripts/orbit_uniqueness.py
Writes docs/ECC27_limit_cycle/orbit_uniqueness.json.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import warnings

import numpy as np
from scipy.optimize import least_squares

warnings.filterwarnings('ignore')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import regindex                                                       # noqa: E402
from core.features import standard_pid_features, compute_features  # noqa: E402
from core.signals import loop_signals, add_derivatives  # noqa: E402
from core.signals import find_index, auto_grid                        # noqa: E402
from core.certificate import true_Ms                                  # noqa: E402

NBAR = np.array([0.5, 0.75, 1.0])
LO, HI = np.log(0.01), np.log(10.0)
H = np.log(1 / 0.9)
BACKOFF = np.array([1.0, 2.0, 3.0]) * np.log(2)
DESC = standard_pid_features()
MOVES = {'e': np.array([1., 1., 1.]), 'd0': np.array([-1., 0., 0.]),
         'd1': np.array([1., -1., 0.]), 'd2': np.array([1., 1., -1.])}
PLANTS = {
    'P1': ([10., 1., 1., 1.], 1.0, 1.0),
    'P2': ([5., 5., 5., 5.], 1.25, 8.0),
    'P3': ([2., 1., 1., 1.], 1.0, 10.0),
    'P4': ([8.] * 6, 1.0, 4.0),
}


class Plant:
    def __init__(self, name, grid, window='guard'):
        self.window = window          # 'guard': [k1, k_delta] as regindex.patch runs; 'full': [k1, k2]
        self.tau, self.K, self.L = PLANTS[name]
        self.tau = np.array(self.tau)
        Tsim, Ts_auto = auto_grid(self.tau, self.L)
        self.Ts = 1.0 if grid == 'ecc' else Ts_auto
        self.Tsim = None if grid == 'ecc' else Tsim
        self.cache = {}

    def counts(self, th):
        key = tuple(np.round(th, 10))
        if key not in self.cache:
            Ki, Kp, Kd = np.exp(th)
            sigs = loop_signals(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd, 'y', self.Tsim)
            k1, k2 = sigs['k1'], sigs['k2']
            kend = sigs['k_delta'] if self.window == 'guard' else k2
            feats = compute_features(DESC, add_derivatives({'e': sigs['e']}), k1, kend,
                                     metric=regindex.encirc_reg)
            _, unst = find_index(k1, k2, sigs['e'])
            self.cache[key] = (np.array([f['N'] for f in feats]), bool(unst))
        return self.cache[key]

    def Ms(self, th):
        Ki, Kp, Kd = np.exp(th)
        return float(true_Ms(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd))


def move_of(N):
    viol = [k for k in range(3) if N[k] > NBAR[k]]
    return 'e' if not viol else f'd{min(viol)}'


def run_fixed(P, th0, n_iter=600):
    th = np.clip(np.array(th0, float), LO, HI)
    seen, states = {}, []
    for k in range(n_iter):
        key = tuple(np.round(th / H, 6))
        if key in seen:
            return np.array(states[seen[key]:]), k
        seen[key] = k
        states.append(th.copy())
        N, unst = P.counts(th)
        th = np.clip(th - BACKOFF if unst else th + H * MOVES[move_of(N)], LO, HI)
    return None, n_iter


def refine(P, th0):
    r = least_squares(lambda th: P.counts(th)[0] - NBAR, th0, diff_step=1e-3,
                      bounds=(LO, HI), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    J = np.array([(P.counts(r.x + 1e-3 * e)[0] - P.counts(r.x - 1e-3 * e)[0]) / 2e-3
                  for e in np.eye(3)]).T
    return r.x, float(np.max(np.abs(r.fun))), np.linalg.svd(J, compute_uv=False)


def cluster(cents, tol=2 * H):
    groups = []
    for c in cents:
        for g in groups:
            if np.linalg.norm(c - g['c']) < tol:
                g['members'].append(c); g['c'] = np.mean(g['members'], axis=0); break
        else:
            groups.append({'c': c.copy(), 'members': [c]})
    return groups


def survey(name, grid, window):
    P = Plant(name, grid, window)
    orb0, _ = run_fixed(P, np.zeros(3))
    c0 = orb0.mean(axis=0)
    cents, costs, box = [], [], 0
    for s in itertools.product((-1, 0, 1), repeat=3):
        orb, n = run_fixed(P, c0 + np.log(8) * np.array(s))
        if orb is None:
            continue
        if np.any(np.isclose(orb, LO) | np.isclose(orb, HI)):
            box += 1
            continue
        cents.append(orb.mean(axis=0)); costs.append(n)
    groups = cluster(cents)
    rows = []
    for g in sorted(groups, key=lambda g: -len(g['members'])):
        root, res, sv = refine(P, g['c'])
        rows.append(dict(n_starts=len(g['members']), centroid=g['c'].tolist(),
                         spread_h=float(np.max(np.ptp(np.array(g['members']), axis=0)) / H),
                         gains=np.exp(g['c']).tolist(), Ms=P.Ms(g['c']),
                         root=root.tolist(), residual=res, sv=sv.tolist(),
                         centroid_to_root_h=float(np.linalg.norm(g['c'] - root) / H)))
    return dict(Ts=P.Ts, n_box=box, n_interior=len(cents), clusters=rows)


if __name__ == '__main__':
    out = {}
    for grid, window in (('ecc', 'guard'), ('auto', 'guard'), ('ecc', 'full'), ('auto', 'full')):
        tag = f'{grid}/{window}'
        out[tag] = {}
        for name in PLANTS:
            r = survey(name, grid, window)
            out[tag][name] = r
            print(f'[{tag}] {name} Ts={r["Ts"]:.3f}: {r["n_interior"]} interior orbits, '
                  f'{r["n_box"]} on the box, {len(r["clusters"])} cluster(s)')
            for c in r['clusters']:
                print(f'     {c["n_starts"]:2d} starts  spread {c["spread_h"]:5.2f}h  '
                      f'gains(Ki,Kp,Kd)={np.round(c["gains"], 4)}  Ms={c["Ms"]:.2f}  '
                      f'root residual {c["residual"]:.1e}  centroid-root {c["centroid_to_root_h"]:.2f}h  '
                      f'sv(J)={np.round(c["sv"], 3)}')
    path = ROOT / 'docs' / 'ECC27_limit_cycle' / 'orbit_uniqueness.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)
