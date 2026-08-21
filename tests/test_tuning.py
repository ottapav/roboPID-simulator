"""The triangular tuning rule and the full tuning loop."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ATOL, RTOL, golden

from core.features import standard_pid_features
from core.tuning import MANUAL_READING, pid_tuning, triangular_rule


def _feats(n0, n1, n2):
    return [{'N': n0}, {'N': n1}, {'N': n2}]


NBAR = (0.5, 0.75, 1.0)
LIMITS = dict(Fp_min=0.01, Fp_max=10.0, Fi_min=0.01, Fi_max=10.0,
              Fd_min=0.01, Fd_max=10.0)


def _rule(feats, unstable, cur=(1.0, 1.0, 1.0), gamma=1.0 / 0.9):
    return triangular_rule(feats, NBAR, unstable, *cur, gamma,
                           LIMITS['Fp_min'], LIMITS['Fp_max'],
                           LIMITS['Fi_min'], LIMITS['Fi_max'],
                           LIMITS['Fd_min'], LIMITS['Fd_max'])


def test_rule_unstable_halves_every_gain():
    Fp, Fi, Fd, row = _rule(_feats(0, 0, 0), unstable=True)
    assert row == 'unstable'
    assert (Fp, Fi, Fd) == (0.5, 0.5, 0.5)


def test_rule_N0_cuts_reset_only():
    Fp, Fi, Fd, row = _rule(_feats(9.0, 9.0, 9.0), unstable=False)
    assert row == 'N0'
    assert Fi < 1.0 and Fp == 1.0 and Fd == 1.0


def test_rule_N1_cuts_gain_and_raises_reset():
    Fp, Fi, Fd, row = _rule(_feats(0.0, 9.0, 9.0), unstable=False)
    assert row == 'N1'
    assert Fp < 1.0 < Fi and Fd == 1.0


def test_rule_N2_cuts_rate_and_raises_the_bands_below():
    Fp, Fi, Fd, row = _rule(_feats(0.0, 0.0, 9.0), unstable=False)
    assert row == 'N2'
    assert Fd < 1.0 < Fp and Fi > 1.0


def test_rule_none_raises_every_band():
    Fp, Fi, Fd, row = _rule(_feats(0.0, 0.0, 0.0), unstable=False)
    assert row == 'none'
    assert Fp > 1.0 and Fi > 1.0 and Fd > 1.0


def test_rule_precedence_is_lowest_violated_band_first():
    """All three violated: Gamma0 wins, and instability outranks all of them."""
    assert _rule(_feats(9, 9, 9), unstable=False)[3] == 'N0'
    assert _rule(_feats(9, 9, 9), unstable=True)[3] == 'unstable'


def test_every_rule_row_has_a_manual_reading():
    """callbacks.on_iteration dereferences MANUAL_READING[row] with no fallback."""
    rows = {_rule(_feats(9, 9, 9), True)[3], _rule(_feats(9, 9, 9), False)[3],
            _rule(_feats(0, 9, 9), False)[3], _rule(_feats(0, 0, 9), False)[3],
            _rule(_feats(0, 0, 0), False)[3]}
    assert rows == set(MANUAL_READING)


def test_rule_respects_its_bounds():
    Fp, Fi, Fd, _ = _rule(_feats(0, 0, 0), False, cur=(10.0, 10.0, 10.0))
    assert (Fp, Fi, Fd) == (10.0, 10.0, 10.0)          # clamped at Fmax
    Fp, Fi, Fd, _ = _rule(_feats(0, 0, 0), True, cur=(0.01, 0.01, 0.01))
    assert (Fp, Fi, Fd) == (0.01, 0.01, 0.01)          # clamped at Fmin


def test_pid_tuning_trajectory_golden(plant):
    name, tau, K, L = plant
    g = golden(name)
    desc = standard_pid_features()
    Fp, Fi, Fd = pid_tuning(desc, tau, K, L, float(g['Ts']), 1.0, 1.0, 1.0,
                            Tsim=float(g['Tsim']), n_iter=int(g['tune_n_iter']))

    np.testing.assert_allclose(Fp, g['tune_Fp'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Fi, g['tune_Fi'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Fd, g['tune_Fd'], rtol=RTOL, atol=ATOL)


def test_pid_tuning_reports_every_iteration(plant):
    _, tau, K, L = plant
    desc = standard_pid_features()
    seen = []
    pid_tuning(desc, tau, K, L, 1.0, 1.0, 1.0, 1.0, Tsim=140.0, n_iter=12,
               on_iteration=lambda *a: seen.append(a))

    assert [s[0] for s in seen] == list(range(1, 12)) + [12]
    assert all(row in MANUAL_READING for *_, row, _f, _s in seen)


def test_pid_tuning_hands_the_callback_what_it_scored(plant):
    """B4: the callback receives the features and signals of the iteration it
    is reporting, so a display caller never has to re-simulate the same gains."""
    _, tau, K, L = plant
    desc = standard_pid_features()
    seen = []
    pid_tuning(desc, tau, K, L, 1.0, 1.0, 1.0, 1.0, Tsim=140.0, n_iter=11,
               on_iteration=lambda *a: seen.append(a))

    for *_, feats, sigs in seen:
        assert [f['name'] for f in feats] == ['Gamma0', 'Gamma1', 'Gamma2']
        assert {'y', 'u', 'e', 't', 'k1', 'k2', 'k_delta'} <= set(sigs)
        assert len(sigs['t']) == len(sigs['y'])

    # The reported features must be the ones the reported multipliers produce.
    i, n, Fp, Fi, Fd, row, feats, _ = seen[3]
    from core.features import loop_response_features
    again, _, _, _ = loop_response_features(
        desc, tau, K, L, 1.0, Fp * 1.0, Fi * 1.0, Fd * 1.0, Tsim=140.0)
    np.testing.assert_allclose([f['N'] for f in feats], [f['N'] for f in again])


def test_pid_tuning_enforces_a_minimum_iteration_count():
    desc = standard_pid_features()
    Fp, _, _ = pid_tuning(desc, [5.0] * 4, 1.25, 8.0, 1.0, 1.0, 1.0, 1.0,
                          Tsim=280.0, n_iter=3)
    assert len(Fp) == 10


def test_pid_tuning_multipliers_stay_within_limits():
    desc = standard_pid_features()
    Fp, Fi, Fd = pid_tuning(desc, [5.0] * 4, 1.25, 8.0, 1.0, 1.0, 1.0, 1.0,
                            Tsim=280.0, n_iter=40,
                            Fp_limits=(0.5, 2.0), Fi_limits=(0.5, 2.0),
                            Fd_limits=(0.5, 2.0))
    for hist in (Fp, Fi, Fd):
        assert hist.min() >= 0.5 - 1e-12 and hist.max() <= 2.0 + 1e-12


def test_tuning_with_noise_is_reproducible_under_a_seed():
    """D2: a seeded tuning run must repeat exactly, and differ from another
    seed. Unseeded, every iteration draws a fresh noise realization, so the
    same gains score differently each time they are visited."""
    desc = standard_pid_features()
    kw = dict(Tsim=280.0, n_iter=25, dist_a=0.9, dist_b=0.02)

    a = pid_tuning(desc, [5.0] * 4, 1.25, 8.0, 1.0, 1.0, 1.0, 1.0, seed=11, **kw)
    b = pid_tuning(desc, [5.0] * 4, 1.25, 8.0, 1.0, 1.0, 1.0, 1.0, seed=11, **kw)
    c = pid_tuning(desc, [5.0] * 4, 1.25, 8.0, 1.0, 1.0, 1.0, 1.0, seed=12, **kw)

    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert any(not np.array_equal(x, z) for x, z in zip(a, c))


# The simulation is driven to overflow on purpose here; numpy's warnings about
# it are the point of the test, not a problem with it.
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_tuner_backs_out_of_divergence_instead_of_climbing_further():
    """The tuner must never answer a diverged run by raising the gains.

    Reported on tau=1, K=1, L=0. A record that overflows to inf was scored
    Gamma = [0, 0, 0] with no instability flag, which triangular_rule reads as
    "all quiet: tighten" — so once the search stepped past the divergence
    threshold it raised Kp every iteration forever (400 -> 1275 over twelve
    iterations, peak|y| = inf throughout). A one-way trap.

    The limits are opened wide on purpose: the box ceiling must not be what
    stops the climb, or the test would pass on the broken code too.
    """
    tau, K, L = np.array([1.0]), 1.0, 0.0
    Tsim = 10.0
    Ts = Tsim / 999
    wide = dict(Fp_limits=(0.01, 1e6), Fi_limits=(0.01, 1e6), Fd_limits=(0.01, 1e6))

    rows = []
    pid_tuning(standard_pid_features(), tau, K, L, Ts, 400.0, 1.0, 1.0,
               Tsim=Tsim, n_iter=12,
               on_iteration=lambda i, n, fp, fi, fd, row, f, s: rows.append(
                   (400.0 * fp, row, np.max(np.abs(s['y'])))),
               **wide)

    assert not np.isfinite(rows[0][2]), 'setup must actually diverge to be a regression'
    assert rows[0][1] == 'unstable', f'diverged run scored {rows[0][1]!r}'
    assert MANUAL_READING[rows[0][1]] == 'runaway: halve all'
    assert rows[1][0] < rows[0][0], 'gain must come down off a diverged run'

    # And it must not sit there: within a few halvings the record is finite again.
    assert any(np.isfinite(pk) for _, _, pk in rows[:4])


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_stability_screen_still_catches_a_merely_huge_record():
    """The non-finite guard is for overflowed records only. A record that is
    enormous but still finite (peak ~1e40) must keep being caught by the
    change-point comparison it was always caught by, not fall through to the
    new shortcut."""
    from core.signals import find_index
    from core.features import loop_response_features

    tau, K, L = np.array([1.0]), 1.0, 0.0
    Tsim = 10.0
    Ts = Tsim / 999

    _, k1, k2, sigs = loop_response_features(
        standard_pid_features(), tau, K, L, Ts, 200.0, 1.0, 1.0, Tsim=Tsim)

    assert np.all(np.isfinite(sigs['y'])), 'this case must stay finite'
    assert np.max(np.abs(sigs['y'])) > 1e30, 'and must still be enormous'
    assert find_index(k1, k2, sigs['e'])[1] is True
