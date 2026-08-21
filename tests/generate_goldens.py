"""Regenerate tests/goldens/*.npz.

Run from the repo root:  py -3 tests/generate_goldens.py

This is not a test. It records what the code currently does so later changes can
be diffed against it. Only re-run it when a behaviour change is *intended*, and
review the resulting diff before committing — regenerating goldens to make a
failing test pass defeats the entire point of having them.

Noise is left off everywhere so every recorded quantity is deterministic.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from conftest import BATTERY, GAINS, GAINS_SAT, GOLDEN_DIR, sim_grid  # noqa: E402

from core.features import standard_pid_features, loop_response_features  # noqa: E402
from core.pid import pid_response_linear, pid_response_awup  # noqa: E402
from core.plant import plant_step_response  # noqa: E402
from core.signals import loop_signals, find_index  # noqa: E402
from core.tuning import pid_tuning  # noqa: E402

N_ITER = 50          # per plant
N_ITER_LONG = 200    # P2 only, matching the app's PID default


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    desc = standard_pid_features()

    for name, (tau_list, K, L) in sorted(BATTERY.items()):
        tau = np.asarray(tau_list, dtype=float)
        Tsim, Ts = sim_grid(tau, L)
        out: dict[str, np.ndarray] = {'tau': tau, 'K': K, 'L': L,
                                      'Tsim': Tsim, 'Ts': Ts}

        y, u, t, Kp, Ki, Kd = pid_response_linear(tau, K, L, *GAINS, Tsim, Ts)
        out |= {'lin_y': y, 'lin_u': u, 'lin_t': t,
                'gains': np.array([Kp, Ki, Kd])}

        ya, ua, _, _, _, _ = pid_response_awup(
            tau, K, L, *GAINS_SAT, Tsim, Ts, minu=-1.0, maxu=1.0)
        out |= {'awup_y': ya, 'awup_u': ua}

        ps, _ = plant_step_response(tau, K, L, Tsim, Ts)
        out['plant_step'] = ps

        sigs = loop_signals(tau, K, L, Ts, *GAINS, Tsim=Tsim)
        out |= {f'sig_{k}': np.asarray(sigs[k], dtype=float)
                for k in ('e', 'v', 'uP', 'uI', 'uD')}
        out |= {'k1': sigs['k1'], 'k2': sigs['k2'], 'k_delta': sigs['k_delta']}

        ind, unstable = find_index(sigs['k1'], sigs['k2'], sigs['e'])
        out |= {'find_ind': ind, 'find_unstable': unstable}

        feats, _, _, _ = loop_response_features(
            desc, tau, K, L, Ts, *GAINS, Tsim=Tsim)
        out['feat_N'] = np.array([f['N'] for f in feats])
        for i, f in enumerate(feats):
            out[f'feat{i}_x'] = f['xdata']
            out[f'feat{i}_y'] = f['ydata']

        n_iter = N_ITER_LONG if name == 'P2' else N_ITER
        Fp, Fi, Fd = pid_tuning(desc, tau, K, L, Ts, 1.0, 1.0, 1.0,
                                Tsim=Tsim, n_iter=n_iter)
        out |= {'tune_Fp': Fp, 'tune_Fi': Fi, 'tune_Fd': Fd,
                'tune_n_iter': n_iter}

        np.savez_compressed(GOLDEN_DIR / f'{name}.npz', **out)
        print(f'{name}: N={len(y)} Ts={Ts:g} Tsim={Tsim:g} '
              f'feat_N={np.round(out["feat_N"], 4)} '
              f'F_final=({Fp[-1]:.6f}, {Fi[-1]:.6f}, {Fd[-1]:.6f})')

    print(f'\nwrote {len(BATTERY)} goldens to {GOLDEN_DIR}')


if __name__ == '__main__':
    main()
