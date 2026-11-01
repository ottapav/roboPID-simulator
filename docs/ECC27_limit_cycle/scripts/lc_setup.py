"""
Shared setup for the limit-cycle paper (docs/ECC27_limit_cycle).

The setting of the theory: a linear loop (no actuator saturation, no noise)
with the derivative on the error, so that the error record after a unit
setpoint step is the step response of the sensitivity S (the companion ECC
paper's Lemma 1). Everything else is the app's default: the 500-sample grid
of core.signals.auto_grid, the box [0.001, 10] on the gains (unit base gains,
so multipliers are gains), beta = 0.1, Nbar = (0.5, 0.75, 1.0), eps = 0.1,
delta = 0.02. The rule is core.tuning.triangular_rule, unmodified.

Two indices drive it: 'hard', the published count (core.features), and
'soft', the smooth count of regindex_soft.py with sharpness p.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs' / 'scripts'))

from core.features import standard_pid_features, compute_features    # noqa: E402
from core.signals import loop_signals, add_derivatives, find_index, auto_grid  # noqa: E402
from core.certificate import true_Ms                                  # noqa: E402
from core.tuning import triangular_rule                               # noqa: E402
from core.amigo import amigo_design                                   # noqa: E402
from regindex_soft import soft_counts                                # noqa: E402

PLANTS = {
    'P1': ([10., 1., 1., 1.], 1.0, 1.0),
    'P2': ([5., 5., 5., 5.], 1.25, 8.0),
    'P3': ([2., 1., 1., 1.], 1.0, 10.0),
    'P4': ([8.] * 6, 1.0, 4.0),
}
NBAR = np.array([0.5, 0.75, 1.0])
BOX = (0.001, 10.0)
LO, HI = np.log(BOX[0]), np.log(BOX[1])
DESC = standard_pid_features()
MOVES = {'e': np.array([1., 1., 1.]), 'd0': np.array([-1., 0., 0.]),
         'd1': np.array([1., -1., 0.]), 'd2': np.array([1., 1., -1.])}


def h_of(beta):
    return float(np.log(1.0 / (1.0 - beta)))


class Loop:
    """Closed-loop step experiments on one plant; theta = log(Ki, Kp, Kd)."""

    def __init__(self, name, index='hard', p=32.0, dtype='e', simtype=0):
        self.name = name
        tau, self.K, self.L = PLANTS[name]
        self.tau = np.array(tau)
        self.Tsim, self.Ts = auto_grid(self.tau, self.L)
        self.index, self.p, self.dtype, self.simtype = index, p, dtype, simtype
        self.cache = {}
        self.n_exp = 0

    def signals(self, th):
        Ki, Kp, Kd = np.exp(th)
        return loop_signals(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd, self.dtype,
                            self.Tsim, self.simtype, -10.0, 10.0)

    def counts(self, th):
        key = tuple(np.round(th, 11))
        if key not in self.cache:
            s = self.signals(th)
            _, unst = find_index(s['k1'], s['k2'], s['e'])
            ext = add_derivatives({'e': s['e']})
            if self.index == 'hard':
                N = np.array([f['N'] for f in compute_features(DESC, ext, s['k1'], s['k_delta'])])
            else:
                N = soft_counts(ext['e'], s['e'], s['k1'], s['k2'], sharp=self.p)
            self.cache[key] = (N, bool(unst))
        return self.cache[key]

    def Ms(self, th):
        Ki, Kp, Kd = np.exp(th)
        return float(true_Ms(self.tau, self.K, self.L, self.Ts, Kp, Ki, Kd, simtype=self.simtype))

    def amigo(self):
        d = amigo_design(self.tau, self.K, self.L, 'PID')
        return np.log([d.Ki, d.Kp, d.Kd])


def step(loop, th, beta=0.1):
    """One iteration of core.tuning.triangular_rule. Returns (theta', row)."""
    N, unst = loop.counts(th)
    loop.n_exp += 1
    Ki, Kp, Kd = np.exp(th)
    Fp, Fi, Fd, row = triangular_rule([{'N': n} for n in N], tuple(NBAR), unst,
                                      Kp, Ki, Kd, 1.0 / (1.0 - beta), *BOX, *BOX, *BOX)
    return np.log([Fi, Fp, Fd]), row


def applied_move(th, th_new, h):
    """Name of the move actually applied: a table move, 'hold', or 'other'."""
    d = (th_new - th) / h
    if np.allclose(d, 0, atol=1e-9):
        return 'hold'
    for k, v in MOVES.items():
        if np.allclose(d, v, atol=1e-6):
            return k
    return 'other'


def run_to_orbit(loop, th0, beta=0.1, n_iter=6000):
    """Iterate until the first exact revisit of a state; the iteration is
    deterministic, so the states from the first visit on form the terminal
    orbit. Returns dict(n, orbit, moves, rows) or None if no revisit."""
    h = h_of(beta)
    th = np.clip(np.array(th0, float), LO, HI)
    seen, states, moves, rows = {}, [], [], []
    for k in range(n_iter):
        key = tuple(np.round(th / h, 7))
        if key in seen:
            i = seen[key]
            return dict(n=k, orbit=np.array(states[i:]), moves=moves[i:], rows=rows[i:],
                        transient_moves=moves[:i])
        seen[key] = k
        states.append(th.copy())
        th_new, row = step(loop, th, beta)
        moves.append('U' if row == 'unstable' else applied_move(th, th_new, h))
        rows.append(row)
        th = th_new
    return None


def grid_starts(loop, factor=8.0):
    """27 starts: the AMIGO gains multiplied by factor^{-1,0,1} in each gain."""
    a = loop.amigo()
    import itertools
    return [a + np.log(factor) * np.array(s) for s in itertools.product((-1, 0, 1), repeat=3)]


def root(loop_soft, th0):
    """A zero of N_p(theta) - Nbar near th0, its residual, Jacobian and the
    Jacobian's singular values (central differences, step 1e-3 in log gain)."""
    r = least_squares(lambda th: loop_soft.counts(th)[0] - NBAR, th0, diff_step=1e-4,
                      bounds=(LO, HI), xtol=1e-13, ftol=1e-13, gtol=1e-13)
    J = np.array([(loop_soft.counts(r.x + 1e-3 * e)[0] - loop_soft.counts(r.x - 1e-3 * e)[0]) / 2e-3
                  for e in np.eye(3)]).T
    return r.x, float(np.max(np.abs(r.fun))), J
