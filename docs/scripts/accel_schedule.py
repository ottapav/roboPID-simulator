"""
SPIN as a sign-based equation solver: accelerating the travel phase.

The triangular rule solves N_k(theta) = Nbar_k, k = 0,1,2 (the triple point,
eq. (triplepoint) of the ECC draft) from the signs of the residuals alone, with
one fixed step h. Like any fixed-step iterative solver it travels at constant
speed and the cost of reaching the cycle grows linearly with the distance D:
about D/h experiments.

The simplest acceleration doubles the step while the rule repeats its move.
This script compares

  fixed        the published rule, constant h
  double       double the step on every repeat of the move, back to h on a change
  double-safe  double only once the run of a move is longer than any closed
               orbit allows. By Corollary 2 (Minimal Period) a closed orbit
               uses the moves (e, d0, d1, d2) exactly n_e*(1, 4, 2, 1) times
               per period, so a run longer than n_e*c_v of move v proves the
               iterate is travelling, not cycling. n_e is taken as 2 (the
               largest multiplicity in the battery, P3).

The step is always h * 2^m with integer m, so every iterate stays on the
lattice of the fixed-step rule. A doubled step that lands on an unstable record
is reverted to the last stable iterate (not backed off, which would change the
gain ratios), and the step returns to h.

Cost is counted in experiments (closed-loop step tests) until the iteration
closes its terminal orbit, detected exactly as the first revisit of a state
(theta, previous move, multiplier) -- the iteration is deterministic. The
quality check is where it closes: the centroid of the terminal orbit against
the centroid of the fixed-step orbit, in units of h.

Setup as in the ECC draft: regularized index (regindex.py), Ts = 1,
dtype='y', box [0.01, 10], plants P1-P4. Starts: the fixed-step orbit centroid
divided by 4, 16 and 64 in every gain (a timid start, the recommended practice),
plus the 6 mixed-sign starts at factor 16.

Run from the repository root:  python docs/scripts/accel_schedule.py
Writes docs/ECC27_limit_cycle/accel_schedule.json.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import regindex                                                       # noqa: E402
from core.features import standard_pid_features, loop_response_features  # noqa: E402
from core.signals import find_index                                   # noqa: E402

TS = 1.0
NBAR = np.array([0.5, 0.75, 1.0])
LO, HI = np.log(0.01), np.log(10.0)
BETA = 0.1
H = np.log(1 / (1 - BETA))
BACKOFF = np.array([1.0, 2.0, 3.0]) * np.log(2)      # graded screen backoff
MMAX = 3                                              # step cap 2^3 h = 8h
N_ITER = 400
N_E = 2
CYCLE_RUN = {'e': 1 * N_E, 'd0': 4 * N_E, 'd1': 2 * N_E, 'd2': 1 * N_E}
DESC = standard_pid_features()

PLANTS = {
    'P1': ([10., 1., 1., 1.], 1.0, 1.0),
    'P2': ([5., 5., 5., 5.], 1.25, 8.0),
    'P3': ([2., 1., 1., 1.], 1.0, 10.0),
    'P4': ([8.] * 6, 1.0, 4.0),
}
MOVES = {'e': np.array([1., 1., 1.]), 'd0': np.array([-1., 0., 0.]),
         'd1': np.array([1., -1., 0.]), 'd2': np.array([1., 1., -1.])}
SCHEDULES = ('fixed', 'double', 'double-safe')


class Plant:
    def __init__(self, name):
        self.tau, self.K, self.L = PLANTS[name]
        self.cache = {}

    def experiment(self, th):
        key = tuple(np.round(th, 9))
        if key not in self.cache:
            Ki, Kp, Kd = np.exp(th)
            feats, k1, k2, sigs = loop_response_features(
                DESC, np.array(self.tau), self.K, self.L, TS, Kp, Ki, Kd,
                dtype='y', metric=regindex.encirc_reg)
            _, unst = find_index(k1, k2, sigs['e'])
            self.cache[key] = (np.array([f['N'] for f in feats]), bool(unst))
        return self.cache[key]


def move_of(N):
    viol = [k for k in range(3) if N[k] > NBAR[k]]
    return 'e' if not viol else f'd{min(viol)}'


def run(P, th0, schedule, n_iter=N_ITER):
    """Iterate until the first revisit of a state. Returns a dict with the
    experiments spent, the terminal orbit, the move string and screen events."""
    th = np.clip(np.array(th0, float), LO, HI)
    prev, runlen, m = None, 0, 0
    last_ok = None
    seen, states, moves = {}, [], []
    n_unstable = n_revert = 0
    for k in range(n_iter):
        key = (tuple(np.round(th / H, 6)), prev, runlen if schedule != 'fixed' else 0, m)
        if key in seen:
            orbit = np.array(states[seen[key]:])
            return dict(n=k, orbit=orbit, moves=''.join(moves), closed=True,
                        n_unstable=n_unstable, n_revert=n_revert,
                        orbit_moves=moves[seen[key]:])
        seen[key] = k
        states.append(th.copy())
        N, unst = P.experiment(th)
        if unst:
            n_unstable += 1
            if m > 0 and last_ok is not None:            # doubled step overshot
                th, n_revert = last_ok.copy(), n_revert + 1
                moves.append('R')
            else:
                th = np.clip(th - BACKOFF, LO, HI)
                moves.append('U')
            prev, runlen, m = None, 0, 0
            continue
        last_ok = th.copy()
        mv = move_of(N)
        runlen = runlen + 1 if mv == prev else 1
        if schedule == 'double':
            m = min(runlen - 1, MMAX)
        elif schedule == 'double-safe':
            m = min(max(runlen - CYCLE_RUN[mv], 0), MMAX)
        else:
            m = 0
        prev = mv
        moves.append({'e': '+', 'd0': '0', 'd1': '1', 'd2': '2'}[mv] if m == 0 else
                     {'e': 'E', 'd0': 'O', 'd1': 'I', 'd2': 'Z'}[mv])
        th = np.clip(th + H * 2 ** m * MOVES[mv], LO, HI)
    return dict(n=n_iter, orbit=np.array(states[-16:]), moves=''.join(moves), closed=False,
                n_unstable=n_unstable, n_revert=n_revert, orbit_moves=[])


def starts(c):
    out = [('timid/4', c - np.log(4)), ('timid/16', c - np.log(16)), ('timid/64', c - np.log(64))]
    for s in itertools.product((-1, 1), repeat=3):
        if len(set(s)) > 1:
            out.append((f'mixed{"".join("+" if x > 0 else "-" for x in s)}16', c + np.log(16) * np.array(s)))
    return out


if __name__ == '__main__':
    out = {}
    for name in PLANTS:
        P = Plant(name)
        ref = run(P, np.zeros(3), 'fixed')
        c = ref['orbit'].mean(axis=0)
        print(f'{name}: fixed-step orbit from unit gains closes after {ref["n"]} experiments, '
              f'period {len(ref["orbit"])}, centroid gains (Ki,Kp,Kd) = {np.round(np.exp(c), 4)}')
        out[name] = {'centroid': c.tolist(), 'period': len(ref['orbit']), 'runs': []}
        for lbl, th0 in starts(c):
            row = {'start': lbl}
            line = f'  {lbl:14s}'
            for sch in SCHEDULES:
                r = run(P, th0, sch)
                off = (np.linalg.norm(r['orbit'].mean(axis=0) - c) / H) if r['closed'] else np.nan
                row[sch] = dict(n=r['n'], closed=r['closed'], offset_h=float(off),
                                period=len(r['orbit']) if r['closed'] else None,
                                n_unstable=r['n_unstable'], n_revert=r['n_revert'],
                                moves=r['moves'])
                line += (f' | {sch}: {r["n"]:3d} exp, off {off:5.2f}h, '
                         f'per {len(r["orbit"]) if r["closed"] else "-":>2}, U{r["n_unstable"]}')
            out[name]['runs'].append(row)
            print(line)
    path = ROOT / 'docs' / 'ECC27_limit_cycle' / 'accel_schedule.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)


# ---------------------------------------------------------------------------
# Certified extrapolation: "if the rule repeats, double the step" for blocks of
# moves. A block whose displacement exceeds the excursion bound (Corollary 3,
# span n_e*(4,2,1)*h per coordinate) cannot lie inside any closed orbit, so the
# iterate is provably travelling. The block's displacement is then replayed
# 2^j times in one jump (j = consecutive jumps, capped), which is the doubling
# rule applied to the drift d = h M^T p of the drift meter instead of to a
# single move. Balanced blocks (the cycle) are never extrapolated.
SPAN = np.array([4.0, 2.0, 1.0]) * N_E
JMAX = 3


def run_extrap(P, th0, n_iter=N_ITER):
    th = np.clip(np.array(th0, float), LO, HI)
    block = np.zeros(3)          # displacement since the last jump, in units of h
    j, last_dir, pre_jump, jumped = 0, None, None, False
    seen, states, moves = {}, [], []
    n_unstable = n_jump = 0
    for k in range(n_iter):
        key = (tuple(np.round(th / H, 6)), tuple(block), j)
        if key in seen:
            return dict(n=k, orbit=np.array(states[seen[key]:]), closed=True,
                        n_unstable=n_unstable, n_jump=n_jump, moves=''.join(moves))
        seen[key] = k
        states.append(th.copy())
        N, unst = P.experiment(th)
        if unst:
            n_unstable += 1
            if jumped and pre_jump is not None:           # the jump overshot: revert
                th = pre_jump.copy(); moves.append('R')
            else:
                th = np.clip(th - BACKOFF, LO, HI); moves.append('U')
            block[:] = 0; j = 0; last_dir = None; jumped = False
            continue
        jumped = False
        mv = move_of(N)
        moves.append({'e': '+', 'd0': '0', 'd1': '1', 'd2': '2'}[mv])
        th = np.clip(th + H * MOVES[mv], LO, HI)
        block += MOVES[mv]
        if np.any(np.abs(block) > SPAN):                  # certified travel
            d = tuple(np.sign(block))
            j = min(j + 1, JMAX) if d == last_dir else 0
            last_dir = d
            pre_jump = th.copy()
            th = np.clip(th + H * (2 ** j) * block, LO, HI)
            moves.append('J')
            n_jump += 1; jumped = True
            block[:] = 0
    return dict(n=n_iter, orbit=np.array(states[-16:]), closed=False,
                n_unstable=n_unstable, n_jump=n_jump, moves=''.join(moves))
