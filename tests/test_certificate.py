"""
Certified robustness from one setpoint step (core.certificate) and the certified
/ safe SPIN variants of pid_tuning (docs/ECC27_certified_SPIN).
"""
import numpy as np
import pytest

from core.certificate import (certificate, certify_ahead, predict_Ms, true_Ms,
                              check_record, increments, tau_for)
from core.features import standard_pid_features
from core.signals import loop_signals, auto_grid
from core.tuning import pid_tuning

LOOPS = [  # tau, K, L, (Kp, Ki, Kd)
    ([5, 5, 5, 5], 1.25, 8.0, (0.3, 0.02, 1.0)),
    ([5, 5, 5, 5], 1.25, 8.0, (0.6, 0.05, 3.0)),
    ([10, 1, 1, 1], 1.0, 1.0, (2.0, 0.3, 2.0)),
    ([10.0], 1.0, 1.0, (3.0, 0.4, 0.0)),
    ([2.0], 1.0, 5.0, (0.0, 0.08, 0.0)),
]


def _record(tau, K, L, gains, simtype):
    Tsim, Ts = auto_grid(tau, L)
    s = loop_signals(tau, K, L, Ts, *gains, dtype='e', Tsim=Tsim, simtype=simtype,
                     minu=-1e3, maxu=1e3)
    return -s['e'], Ts, s


@pytest.mark.parametrize('simtype', [0, 1])
@pytest.mark.parametrize('loop', LOOPS)
def test_certificate_brackets_true_Ms(loop, simtype):
    tau, K, L, g = loop
    e_r, Ts, s = _record(tau, K, L, g, simtype)
    assert check_record(e_r, s['u'], -1e3, 1e3) is None
    c = certificate(e_r, tau_for(e_r))
    Ms = true_Ms(tau, K, L, Ts, *g, simtype=simtype)
    assert c['lower'] - 1e-9 <= Ms <= c['upper'] + 1e-9
    assert c['upper'] - c['lower'] < 0.02


def test_record_is_impulse_response_of_S():
    tau, K, L, g = LOOPS[0]
    e_r, Ts, _ = _record(tau, K, L, g, 0)
    a = increments(e_r)
    assert a[0] == pytest.approx(1.0)                  # e_0 = 1: L strictly proper
    assert a.sum() == pytest.approx(e_r[-1])            # the sum of increments is e_N
    assert abs(e_r[-1]) < 1e-3                          # S(1) = 0: integral action settles e


@pytest.mark.parametrize('mult', [(1.11, 1.11, 1.11), (0.9, 1.11, 1.0), (1.0, 1.0, 0.9),
                                  (1.5, 1.5, 1.5), (0.5, 0.5, 0.5)])
def test_certify_ahead_bounds_the_next_loop(mult):
    tau, K, L, g = LOOPS[0]
    e_r, Ts, _ = _record(tau, K, L, g, 1)
    g2 = tuple(x * m for x, m in zip(g, mult))
    A = certify_ahead(e_r, Ts, g, g2, tau_for(e_r))
    Ms2 = true_Ms(tau, K, L, Ts, *g2, simtype=1)
    assert A['stable']
    assert Ms2 <= A['U'] + 1e-9
    assert predict_Ms(e_r, Ts, g, g2) == pytest.approx(Ms2, rel=1e-3)


def test_check_record_flags_saturation_and_noise():
    tau, K, L, g = LOOPS[1]
    Tsim, Ts = auto_grid(tau, L)
    s = loop_signals(tau, K, L, Ts, *g, dtype='e', Tsim=Tsim, simtype=1, minu=-0.5, maxu=0.5)
    assert check_record(-s['e'], s['u'], -0.5, 0.5) == 'actuator saturated'
    assert check_record(-s['e'], noisy=True) == 'measurement noise'


def _run(variant, n_iter=80):
    tau, K, L = [5, 5, 5, 5], 1.25, 8.0
    Kp, Ki, Kd = 0.3, 0.02, 1.0
    Tsim, Ts = auto_grid(tau, L)
    applied = []

    def cb(i, n, Fp, Fi, Fd, row, feats, sigs):
        applied.append((i, true_Ms(tau, K, L, Ts, Fp * Kp, Fi * Ki, Fd * Kd, simtype=1),
                        sigs.get('cert')))

    Fp, Fi, Fd = pid_tuning(standard_pid_features(), tau, K, L, Ts, Kp, Ki, Kd,
                            dtype='e', Tsim=Tsim, n_iter=n_iter, simtype=1,
                            minu=-10, maxu=10, variant=variant, on_iteration=cb,
                            Fp_limits=(0.01, 10), Fi_limits=(0.01, 10), Fd_limits=(0.01, 10))
    final = true_Ms(tau, K, L, Ts, Fp[-1] * Kp, Fi[-1] * Ki, Fd[-1] * Kd, simtype=1)
    return applied[:-1], final


def test_certified_spin_ends_below_Mstar():
    applied, final = _run('certified')
    assert final <= 1.7
    assert all(c is not None and 'upper' in c for _, _, c in applied)


def test_safe_spin_never_applies_above_Mstar():
    applied, final = _run('safe')
    first = next(i for i, (_, _, c) in enumerate(applied) if c['accepted'])
    assert all(ms <= 1.7 for _, ms, _ in applied[first:])
    assert final <= 1.7


def test_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        pid_tuning(standard_pid_features(), [5.0], 1.0, 1.0, 0.1, 1.0, 1.0, 0.0,
                   n_iter=10, variant='bold')
