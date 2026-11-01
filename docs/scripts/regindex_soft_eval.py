"""
How well does the smooth count (regindex_soft.soft_counts) explain the rule?

For several sharpness values p (window end and disc test; window start fixed at
200 per unit of e; origin radius eps/100), against the published hard count of
the current rule (core.tuning), app defaults as in orbit_uniqueness.py:

  agreement    share of the states visited by hard-count runs (27 starts x 120
               iterations per plant) where the smooth count makes the same move;
               and the median and 99th percentile of max_k |N_k - N_k,hard|;
  destination  terminal orbits reached with the smooth count driving the rule,
               from the same 27 starts: Ki range and worst M_s on the orbit,
               against the hard count's;
  continuity   on P1, the line through the gain vector where the
               delta-guarded index jumped (ORBIT_UNIQUENESS_NOTES.md, Sec. 1),
               sampled at 1/400 of a rule step: the largest change of N between
               neighbouring samples, for the hard count and the smooth one.

Run from the repository root:  python docs/scripts/regindex_soft_eval.py
Writes docs/ECC27_limit_cycle/regindex_soft_eval.json.
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
from core.signals import loop_signals, add_derivatives, find_index    # noqa: E402
from regindex_soft import soft_counts                                # noqa: E402
from regindex_soft_check import visited, record                      # noqa: E402

SHARP = (8.0, 32.0, 128.0)


class PlantSoft(O.Plant):
    def __init__(self, name, sharp):
        super().__init__(name, 'hard', 'guard')
        self.sharp, self.hard = sharp, False

    def counts(self, th):
        key = tuple(np.round(th, 10))
        if key not in self.cache:
            Ki, Kp, Kd = np.exp(th)
            s = loop_signals(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd, 'y', self.Tsim,
                             O.SIMTYPE, O.MINU, O.MAXU)
            ext = add_derivatives({'e': s['e']})['e']
            _, unst = find_index(s['k1'], s['k2'], s['e'])
            self.cache[key] = (soft_counts(ext, s['e'], s['k1'], s['k2'], sharp=self.sharp), bool(unst))
        return self.cache[key]


def move(N, unst):
    if unst:
        return 'U'
    v = [k for k in range(3) if N[k] > O.NBAR[k]]
    return 'e' if not v else f'd{v[0]}'


def destinations(P, starts):
    d = []
    for th0 in starts:
        orb, _ = O.run_fixed(P, th0)
        if orb is not None:
            d.append((float(np.exp(orb.mean(axis=0))[0]), max(P.Ms(t) for t in orb)))
    return np.array(d)


def continuity_P1():
    """The P1 cliff line of the notes: uniform direction through the low-Ki
    orbit of the delta-guarded regularized index, +-0.5 h, 400 samples per h."""
    P = O.Plant('P1', 'hard', 'guard')
    J = json.loads((ROOT / 'docs' / 'ECC27_limit_cycle' / 'orbit_uniqueness.json').read_text())
    cl = J['current/reg-guard']['P1']['clusters']
    c = np.array(min(cl, key=lambda c: c['gains'][0])['centroid'])
    s = np.linspace(-0.5, 0.5, 401) * O.H
    out = {'hard': []}
    out.update({f'p{int(p)}': [] for p in SHARP})
    for si in s:
        th = c + si * np.ones(3)
        ext, e, k1, k2 = record(P, th)
        out['hard'].append(P.counts(th)[0].tolist())
        for p in SHARP:
            out[f'p{int(p)}'].append(soft_counts(ext, e, k1, k2, sharp=p).tolist())
    return {k: float(np.max(np.abs(np.diff(np.array(v), axis=0)))) for k, v in out.items()}


if __name__ == '__main__':
    res = {}
    for name in O.PLANTS:
        Ph = O.Plant(name, 'hard', 'guard')
        pts = visited(Ph)
        recs = [record(Ph, t) for t in pts]
        hard = [Ph.counts(t) for t in pts]
        hm = [move(*h) for h in hard]
        c0 = O.run_fixed(Ph, np.zeros(3))[0].mean(axis=0)
        starts = [c0 + np.log(8) * np.array(s) for s in itertools.product((-1, 0, 1), repeat=3)]
        dh = destinations(Ph, starts)
        res[name] = {'n_states': len(pts),
                     'hard': {'Ki': [dh[:, 0].min(), dh[:, 0].max()], 'Ms': [dh[:, 1].min(), dh[:, 1].max()]}}
        print(f'{name}: {len(pts)} states; hard: Ki {dh[:,0].min():.4f}-{dh[:,0].max():.4f}, '
              f'worst M_s {dh[:,1].min():.2f}-{dh[:,1].max():.2f}', flush=True)
        for p in SHARP:
            Ns = [soft_counts(*r, sharp=p) for r in recs]
            same = np.mean([move(n, h[1]) == m for n, h, m in zip(Ns, hard, hm)])
            dN = np.array([np.max(np.abs(n - h[0])) for n, h in zip(Ns, hard)])
            d = destinations(PlantSoft(name, p), starts)
            res[name][f'p{int(p)}'] = {'agreement': float(same), 'dN_median': float(np.median(dN)),
                                       'dN_p99': float(np.percentile(dN, 99)),
                                       'Ki': [d[:, 0].min(), d[:, 0].max()],
                                       'Ms': [d[:, 1].min(), d[:, 1].max()]}
            print(f'   p={p:5.0f}: same move {100*same:5.1f}%  |dN| median {np.median(dN):.4f}, '
                  f'99% {np.percentile(dN, 99):.3f}  Ki {d[:,0].min():.4f}-{d[:,0].max():.4f}  '
                  f'worst M_s {d[:,1].min():.2f}-{d[:,1].max():.2f}', flush=True)
    res['continuity_P1'] = continuity_P1()
    print('P1 cliff line, largest change of N between samples 1/400 h apart:', res['continuity_P1'])
    path = ROOT / 'docs' / 'ECC27_limit_cycle' / 'regindex_soft_eval.json'
    path.write_text(json.dumps(res, indent=1, default=float))
    print('wrote', path)
