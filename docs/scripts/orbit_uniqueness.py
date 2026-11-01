"""
Is the terminal orbit of fixed-step SPIN unique?

The ECC draft (Corollary 1) states that the orbit circles one triple point
theta*, N_k(theta*) = Nbar_k, fixed by the plant alone. accel_schedule.py found
fixed-step runs from different starts closing different orbits. This script
maps the terminal orbits of the algorithm as it runs today and tests:

  H1/H2  one or several roots of N(theta) = Nbar, and how well conditioned;
  H4     the window: regindex.patch() swaps only the metric, so the
         regularized index is still taken over the delta-guarded window
         [k1, k_delta]; a jump of k_delta (one more lobe crossing the delta
         band) is then a jump of the index. 'reg-full' takes it over [k1, k2],
         as Sec. 3 of the draft defines it.

(An earlier version also tested the old ECC grid Ts = 1 against the app grid;
it changed nothing qualitatively and is in the git history.)

Configuration: see CONFIGS below. Per plant: fixed-step runs from a 3x3x3 grid
of starts (factor 8) around the unit-gain orbit, each until the first exact
state revisit; interior orbit centroids are clustered (distance < 2h) and, for
the regularized index, refined to a root by least squares with the residual
and Jacobian singular values reported.

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


# Configurations. 'current' is the algorithm as the app runs it today
# (core.params defaults, robopid.config): 500-sample auto grid, actuator
# saturation +-10 (simtype 1), derivative on the measurement, box [0.001, 10],
# beta = 0.1, published hard count over the delta-guarded window. The two
# regularized variants are the analysis index of the ECC draft over the guarded
# window (what regindex.patch() gives) and over the full window [k1, k2] (what
# Sec. 3 defines).
CONFIGS = {
    'current/hard':      dict(metric='hard', window='guard'),
    'current/reg-guard': dict(metric='reg', window='guard'),
    'current/reg-full':  dict(metric='reg', window='full'),
}
BOX = (0.001, 10.0)
SIMTYPE, MINU, MAXU = 1, -10.0, 10.0
LO, HI = np.log(BOX[0]), np.log(BOX[1])


class Plant:
    def __init__(self, name, metric='hard', window='guard'):
        self.metric = regindex._HARD if metric == 'hard' else regindex.encirc_reg
        self.hard = metric == 'hard'
        self.window = window
        self.tau, self.K, self.L = PLANTS[name]
        self.tau = np.array(self.tau)
        self.Tsim, self.Ts = auto_grid(self.tau, self.L)
        self.cache = {}

    def counts(self, th):
        key = tuple(np.round(th, 10))
        if key not in self.cache:
            Ki, Kp, Kd = np.exp(th)
            sigs = loop_signals(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd, 'y', self.Tsim,
                                SIMTYPE, MINU, MAXU)
            k1, k2 = sigs['k1'], sigs['k2']
            kend = sigs['k_delta'] if self.window == 'guard' else k2
            feats = compute_features(DESC, add_derivatives({'e': sigs['e']}), k1, kend,
                                     metric=self.metric)
            _, unst = find_index(k1, k2, sigs['e'])
            self.cache[key] = (np.array([f['N'] for f in feats]), bool(unst))
        return self.cache[key]

    def Ms(self, th):
        Ki, Kp, Kd = np.exp(th)
        return float(true_Ms(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd, simtype=SIMTYPE))


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


def jacobian(P, th, d=1e-3):
    return np.array([(P.counts(th + d * e)[0] - P.counts(th - d * e)[0]) / (2 * d)
                     for e in np.eye(3)]).T


def refine(P, th0):
    """Root of N = Nbar near th0 (regularized index only; the hard count is
    piecewise constant, so for it only the centroid residual is reported)."""
    if P.hard:
        return th0, float(np.max(np.abs(P.counts(th0)[0] - NBAR))), None
    r = least_squares(lambda th: P.counts(th)[0] - NBAR, th0, diff_step=1e-3,
                      bounds=(LO, HI), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return r.x, float(np.max(np.abs(r.fun))), np.linalg.svd(jacobian(P, r.x), compute_uv=False)


def cluster(cents, tol=2 * H):
    groups = []
    for c in cents:
        for g in groups:
            if np.linalg.norm(c - g['c']) < tol:
                g['members'].append(c); g['c'] = np.mean(g['members'], axis=0); break
        else:
            groups.append({'c': c.copy(), 'members': [c]})
    return groups


def survey(name, cfg):
    P = Plant(name, **cfg)
    orb0, _ = run_fixed(P, np.zeros(3))
    c0 = orb0.mean(axis=0)
    cents, orbits, n_box_by_gain = [], [], np.zeros(3, int)
    for s in itertools.product((-1, 0, 1), repeat=3):
        orb, n = run_fixed(P, c0 + np.log(8) * np.array(s))
        if orb is None:
            continue
        orbits.append(orb)
        atb = (np.isclose(orb, LO) | np.isclose(orb, HI)).any(axis=0)
        if atb.any():
            n_box_by_gain += atb
            continue
        cents.append(orb.mean(axis=0))
    allc = np.array([o.mean(axis=0) for o in orbits])
    allg = np.exp(allc)
    ms = np.array([P.Ms(c) for c in allc])
    rows = []
    for g in sorted(cluster(cents), key=lambda g: -len(g['members'])):
        root, res, sv = refine(P, g['c'])
        rows.append(dict(n_starts=len(g['members']), centroid=g['c'].tolist(),
                         gains=np.exp(g['c']).tolist(), Ms=P.Ms(g['c']),
                         residual=res, sv=None if sv is None else sv.tolist(),
                         centroid_to_root_h=float(np.linalg.norm(g['c'] - root) / H)))
    return dict(Ts=P.Ts, n_orbits=len(orbits), n_box=int(len(orbits) - len(cents)),
                n_box_by_gain=n_box_by_gain.tolist(),
                ranges={k: [float(allg[:, i].min()), float(allg[:, i].max())]
                        for i, k in enumerate(('Ki', 'Kp', 'Kd'))},
                Ms_range=[float(ms.min()), float(ms.max())], clusters=rows)


if __name__ == '__main__':
    out = {}
    for tag, cfg in CONFIGS.items():
        out[tag] = {}
        for name in PLANTS:
            r = survey(name, cfg)
            out[tag][name] = r
            rg = r['ranges']
            print(f'[{tag}] {name} Ts={r["Ts"]:.3f}: {r["n_orbits"]} orbits, {len(r["clusters"])} interior '
                  f'cluster(s), {r["n_box"]} on the box (Ki,Kp,Kd at box: {r["n_box_by_gain"]}); '
                  f'Ki x{rg["Ki"][1]/rg["Ki"][0]:.2f}, Kp x{rg["Kp"][1]/rg["Kp"][0]:.2f}, '
                  f'Kd {rg["Kd"][0]:.3f}-{rg["Kd"][1]:.3f}, Ms {r["Ms_range"][0]:.2f}-{r["Ms_range"][1]:.2f}')
            for c in r['clusters'][:4]:
                sv = '' if c['sv'] is None else f'  sv(J)={np.round(c["sv"], 3)}'
                print(f'     {c["n_starts"]:2d} starts  gains(Ki,Kp,Kd)={np.round(c["gains"], 4)}  '
                      f'Ms={c["Ms"]:.2f}  residual {c["residual"]:.1e}{sv}')
    path = ROOT / 'docs' / 'ECC27_limit_cycle' / 'orbit_uniqueness.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)
