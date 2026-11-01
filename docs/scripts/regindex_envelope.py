"""
A continuous index that agrees with the published hard count.

Finding behind it: core.features.find_disc_entrypoint returns the last segment
that touches the eps-disc, and a portrait that ends inside the disc touches it
to the very end, so the published count truncates (almost) nothing: it is the
whole winding over the delta-guarded window [k1, k_delta]. The truncation that
matters is k_delta (Definition 4), and it is k_delta, not the disc, that makes
the count jump with the gains.

The regularized index of the ECC draft weights by the radius instead, and so
measures something else: it discards the small-radius winding that the hard
count includes. This script smooths k_delta rather than the disc. Each segment
k of every portrait is weighted by the future envelope of the error,

    E_k = max_{m >= k} |e_m| / max |e|,     w_k = E_k^p / (E_k^p + delta^p),

so w_k ~ 1 while the error will still leave the delta band and ~ 0 once it
never will: a smooth version of "k < k_delta". E_k is continuous in the gains,
and so is the index; as p grows it tends to the hard count over [k1, k_delta].
The portraits are taken over [k1, k2] with the same per-axis peak
normalization; Gamma0's integral axis is taken relative to its value at k2.

Measured, against the hard count of the current rule (core.tuning) with the
app's defaults, as in regindex_sharpness.py: share of identical moves on the
states visited by hard-count runs, and destinations (Ki, worst M_s on orbit).

Run from the repository root:  python docs/scripts/regindex_envelope.py
Writes docs/ECC27_limit_cycle/regindex_envelope.json.
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
from core.features import DELTA                                      # noqa: E402
from core.signals import loop_signals, add_derivatives, find_index    # noqa: E402

AXES = ((0, 3), (1, 0), (2, 1))      # (x col, y col) of Gamma0, Gamma1, Gamma2 in add_derivatives
P_VALUES = (8, 32, 128)


def envelope_counts(e, k1, k2, p, delta=DELTA):
    ext = add_derivatives({'e': e})['e']
    peak = np.max(np.abs(e[:k2 + 1]))
    E = np.maximum.accumulate(np.abs(e[k1:k2 + 1])[::-1])[::-1] / peak
    w = E ** p / (E ** p + delta ** p)
    w = w[1:]                                   # segment k -> k+1 weighted by the envelope ahead of it
    N = []
    for b, (ix, iy) in enumerate(AXES):
        x = ext[k1:k2 + 1, ix].copy(); y = ext[k1:k2 + 1, iy].copy()
        if b == 0:
            y = y - ext[k2, iy]
        x /= np.max(np.abs(x)); y /= np.max(np.abs(y))
        dphi = np.arctan2(x[:-1] * y[1:] - y[:-1] * x[1:], x[:-1] * x[1:] + y[:-1] * y[1:])
        N.append(float(np.sum(w * dphi) / (2 * np.pi)))
    return np.array(N)


class PlantEnv(O.Plant):
    def __init__(self, name, p):
        super().__init__(name, 'hard', 'guard')
        self.p, self.hard = p, False

    def counts(self, th):
        key = tuple(np.round(th, 10))
        if key not in self.cache:
            Ki, Kp, Kd = np.exp(th)
            s = loop_signals(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd, 'y', self.Tsim,
                             O.SIMTYPE, O.MINU, O.MAXU)
            _, unst = find_index(s['k1'], s['k2'], s['e'])
            self.cache[key] = (envelope_counts(s['e'], s['k1'], s['k2'], self.p), bool(unst))
        return self.cache[key]


if __name__ == '__main__':
    res = {}
    for name in O.PLANTS:
        Ph = O.Plant(name, 'hard', 'guard')
        c0 = O.run_fixed(Ph, np.zeros(3))[0].mean(axis=0)
        starts = [c0 + np.log(8) * np.array(s) for s in itertools.product((-1, 0, 1), repeat=3)]
        pts = []
        for th0 in starts:
            th = np.clip(th0, O.LO, O.HI)
            for _ in range(150):
                pts.append(th.copy()); th, _ = O.step(Ph, th)
        pts = np.unique(np.round(np.array(pts), 9), axis=0)
        hm = [O.step(Ph, t)[1] for t in pts]

        def dest(P):
            d = []
            for th0 in starts:
                orb, _ = O.run_fixed(P, th0)
                if orb is not None:
                    d.append((float(np.exp(orb.mean(axis=0))[0]), max(P.Ms(t) for t in orb)))
            return np.array(d)

        dh = dest(Ph)
        res[name] = {'n_states': len(pts), 'hard': {'Ki': [dh[:, 0].min(), dh[:, 0].max()],
                                                    'Ms': [dh[:, 1].min(), dh[:, 1].max()]}}
        print(f'{name}: {len(pts)} states; hard: Ki {dh[:,0].min():.4f}-{dh[:,0].max():.4f}, '
              f'worst M_s {dh[:,1].min():.2f}-{dh[:,1].max():.2f}')
        for p in P_VALUES:
            Pe = PlantEnv(name, p)
            agree = float(np.mean([O.step(Pe, t)[1] == m for t, m in zip(pts, hm)]))
            dN = float(np.median([np.max(np.abs(Pe.counts(t)[0] - Ph.counts(t)[0])) for t in pts]))
            d = dest(Pe)
            res[name][f'p{p}'] = {'agreement': agree, 'median_max_dN': dN,
                                  'Ki': [d[:, 0].min(), d[:, 0].max()],
                                  'Ms': [d[:, 1].min(), d[:, 1].max()]}
            print(f'   p={p:3d}: same move {100*agree:5.1f}%  median max|N-N_hard| {dN:.3f}  '
                  f'Ki {d[:,0].min():.4f}-{d[:,0].max():.4f}  worst M_s {d[:,1].min():.2f}-{d[:,1].max():.2f}')
    path = ROOT / 'docs' / 'ECC27_limit_cycle' / 'regindex_envelope.json'
    path.write_text(json.dumps(res, indent=1, default=float))
    print('wrote', path)
