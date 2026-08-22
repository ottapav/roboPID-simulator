"""
Measure the duty cycle of the terminal limit cycle.

Uses the reference implementation (core/tuning.py) unmodified: the
on_iteration callback reports which row of the triangular rule fired at each
iteration, and the shares of those rows over the terminal window are the
duty cycle predicted by Proposition 1.

Row -> move:  'none' -> e,  'N0' -> d0,  'N1' -> d1,  'N2' -> d2
('unstable' rows are screen events, not moves of the cycle; they only occur
in the initial transient and are reported separately.)
"""
from __future__ import annotations
import numpy as np
from collections import Counter

from core.features import standard_pid_features
from core.tuning import pid_tuning

Ts = 1.0                       # app default
N_ITER = 400                   # long enough to reach and sample the cycle
TAIL = 160                     # terminal window used for the duty cycle

# battery presets from callbacks.BATTERY_PRESETS (tau list, K, L)
PLANTS = {
    'P1': ([10., 1., 1., 1.],           1.0, 1.0),
    'P2': ([5., 5., 5., 5.],            1.25, 8.0),
    'P3': ([2., 1., 1., 1.],            1.0, 10.0),
    'P4': ([8., 8., 8., 8., 8., 8.],    1.0, 4.0),
}

ROW2MOVE = {'none': 'e', 'N0': 'd0', 'N1': 'd1', 'N2': 'd2'}
PRED = {'e': 0.125, 'd0': 0.5, 'd1': 0.25, 'd2': 0.125}


def run(tau, K, L, beta, Kp=1.0, Ki=1.0, Kd=1.0, n_iter=N_ITER):
    desc = standard_pid_features()
    rows: list[str] = []

    def cb(i, n, Fp, Fi, Fd, row):
        rows.append(row)

    Fp, Fi, Fd = pid_tuning(
        desc, np.array(tau), K, L, Ts,
        Kp, Ki, Kd, dtype='y',
        n_iter=n_iter, beta=beta,
        on_iteration=cb,
    )
    return rows, (Fp, Fi, Fd)


def duty(rows, tail=TAIL):
    seq = rows[-tail:]
    c = Counter(seq)
    n_unstable = c.get('unstable', 0)
    moves = [ROW2MOVE[r] for r in seq if r in ROW2MOVE]
    cm = Counter(moves)
    tot = len(moves)
    return {m: cm.get(m, 0) / tot for m in ('e', 'd0', 'd1', 'd2')}, tot, n_unstable


def period_check(rows, tail=TAIL):
    """Report the shortest p in 1..16 for which the tail is p-periodic."""
    seq = [r for r in rows[-tail:] if r in ROW2MOVE]
    for p in range(1, 17):
        if all(seq[i] == seq[i + p] for i in range(len(seq) - p)):
            return p
    return None


if __name__ == '__main__':
    print(f'{"plant":5s} {"beta":>5s} {"w_e":>7s} {"w_d0":>7s} {"w_d1":>7s} '
          f'{"w_d2":>7s} {"period":>7s}  terminal multipliers')
    print('-' * 86)
    for beta in (0.10, 0.03):
        for name, (tau, K, L) in PLANTS.items():
            rows, (Fp, Fi, Fd) = run(tau, K, L, beta)
            d, n, nu = duty(rows)
            p = period_check(rows)
            print(f'{name:5s} {beta:5.2f} '
                  f'{d["e"]:7.3f} {d["d0"]:7.3f} {d["d1"]:7.3f} {d["d2"]:7.3f} '
                  f'{str(p):>7s}  '
                  f'({Fi[-1]:.3f}, {Fp[-1]:.3f}, {Fd[-1]:.3f})')
        print()
    print('predicted', {k: v for k, v in PRED.items()})
