"""
Check that regindex_soft.soft_counts is the published count in its hard limit.

States: every gain vector visited by 120 iterations of the current rule (hard
count, app defaults, as orbit_uniqueness.py) from the 27 starts of each of
P1-P4. At each state the published counts (core.features via
orbit_uniqueness.Plant) are compared with soft_counts(sharp=None), and with a
corrected encirc that also checks the last segment for a crossing of the
negative x-axis. Mismatches are classified.

Run from the repository root:  python docs/scripts/regindex_soft_check.py
"""
from __future__ import annotations

import itertools
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import orbit_uniqueness as O                                          # noqa: E402
import core.features as F                                             # noqa: E402
from core.signals import loop_signals, add_derivatives                # noqa: E402
from regindex_soft import soft_counts                                # noqa: E402


def encirc_last_segment(x, y, eps=F.EPSILON):
    """core.features.encirc with the crossing correction run over all segments
    0..j-1; the published loop stops at j-2 although the angle is read at j."""
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    mx, my = np.max(np.abs(x)), np.max(np.abs(y))
    if mx < 1e-12 or my < 1e-12:
        return 0.0
    x, y = x / mx, y / my
    j = min(F.find_disc_entrypoint(x, y, eps), len(x) - 1)
    if j <= 1:
        return 0.0
    angle = np.arctan2(y[j], x[j]) - np.arctan2(y[0], x[0])
    xp, xn, yp, yn = x[:j], x[1:j + 1], y[:j], y[1:j + 1]
    inn = (xp < 0) & (xn < 0)
    return angle / (2 * np.pi) - float(np.sum(inn & (yp < 0) & (yn >= 0))) \
        + float(np.sum(inn & (yp >= 0) & (yn < 0)))


def record(P, th):
    Ki, Kp, Kd = np.exp(th)
    s = loop_signals(P.tau, P.K, P.L, P.Ts, Kp, Ki, Kd, 'y', P.Tsim, O.SIMTYPE, O.MINU, O.MAXU)
    return add_derivatives({'e': s['e']})['e'], s['e'], s['k1'], s['k2']


def visited(Ph, n_steps=120):
    c0 = O.run_fixed(Ph, np.zeros(3))[0].mean(axis=0)
    pts = []
    for s in itertools.product((-1, 0, 1), repeat=3):
        th = np.clip(c0 + np.log(8) * np.array(s), O.LO, O.HI)
        for _ in range(n_steps):
            pts.append(th.copy()); th, _ = O.step(Ph, th)
    return np.unique(np.round(np.array(pts), 9), axis=0)


if __name__ == '__main__':
    tot = {'states': 0, 'exact': 0, 'last_segment': 0, 'origin': 0, 'j_le_1': 0, 'other': 0}
    for name in O.PLANTS:
        Ph = O.Plant(name, 'hard', 'guard')
        Pf = O.Plant(name, 'hard', 'guard'); Pf.metric = encirc_last_segment
        pts = visited(Ph)
        c = dict.fromkeys(tot, 0); c['states'] = len(pts)
        for t in pts:
            ext, e, k1, k2 = record(Ph, t)
            Ns, parts = soft_counts(ext, e, k1, k2, sharp=None, return_parts=True)
            Nh, Nf = Ph.counts(t)[0], Pf.counts(t)[0]
            if np.max(np.abs(Nh - Ns)) <= 1e-6:
                c['exact'] += 1
            elif np.max(np.abs(Nf - Ns)) <= 1e-6:
                c['last_segment'] += 1
            else:
                b = int(np.argmax(np.abs(Nf - Ns)))
                pr = parts[b]
                big = pr is not None and np.max(np.abs(pr['dphi']) * pr['S'] * pr['V']) > 3.0
                if big and abs(abs(Nf[b] - Ns[b]) - 1) < 1e-6:
                    c['origin'] += 1
                elif Nf[b] == 0.0 and abs(Ns[b]) < 0.05:
                    c['j_le_1'] += 1
                else:
                    c['other'] += 1
        for k in tot:
            tot[k] += c[k]
        print(name, c)
    print('total', tot)
