"""Encirclement counts and the Pachner-plot feature pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import GAINS, SETTLING_GAINS, SIM_ATOL, SIM_RTOL, golden, sim_grid

from core.features import (
    EPSILON, compute_features, encirc, find_disc_entrypoint,
    loop_response_features, standard_pid_features,
)
from core.signals import add_derivatives


def _spiral(turns: float, n: int = 4000, decay: float = 0.0):
    """A trajectory winding `turns` times counter-clockwise about the origin."""
    th = np.linspace(0.0, 2.0 * np.pi * turns, n)
    r = np.exp(-decay * np.linspace(0.0, 1.0, n))
    return r * np.cos(th), r * np.sin(th)


@pytest.mark.parametrize('turns', [1.0, 2.0, 3.0, 0.5])
def test_encirc_counts_known_windings(turns):
    x, y = _spiral(turns)
    assert encirc(x, y, eps=1e-9) == pytest.approx(turns, abs=1e-3)


def test_encirc_sign_follows_direction():
    x, y = _spiral(2.0)
    assert encirc(x, y, eps=1e-9) == pytest.approx(-encirc(x, -y, eps=1e-9), abs=1e-6)


def test_encirc_is_scale_free():
    """Per-axis peak normalization: the count cannot depend on units."""
    x, y = _spiral(2.0)
    assert encirc(x, y, eps=1e-9) == pytest.approx(
        encirc(1000.0 * x, 0.001 * y, eps=1e-9), abs=1e-9)


def test_encirc_degenerate_inputs():
    assert encirc(np.zeros(10), np.zeros(10)) == 0.0
    assert encirc(np.ones(10), np.zeros(10)) == 0.0
    assert encirc(np.array([1.0]), np.array([1.0])) == 0.0


def test_find_disc_entrypoint_truncates_at_last_entry():
    x, y = _spiral(3.0, decay=4.0)          # spirals inward
    j = find_disc_entrypoint(x, y, EPSILON)
    assert 0 < j <= len(x)
    assert np.hypot(x[j - 1], y[j - 1]) <= max(EPSILON * 2, np.hypot(x[-1], y[-1]) * 2)


def test_feature_counts_golden(plant):
    name, tau, K, L = plant
    g = golden(name)
    desc = standard_pid_features()
    feats, _, _, _ = loop_response_features(
        desc, tau, K, L, float(g['Ts']), *GAINS, Tsim=float(g['Tsim']))

    np.testing.assert_allclose([f['N'] for f in feats], g['feat_N'],
                               rtol=SIM_RTOL, atol=SIM_ATOL)
    for i, f in enumerate(feats):
        np.testing.assert_allclose(f['xdata'], g[f'feat{i}_x'], rtol=SIM_RTOL, atol=SIM_ATOL)
        np.testing.assert_allclose(f['ydata'], g[f'feat{i}_y'], rtol=SIM_RTOL, atol=SIM_ATOL)


def test_counts_are_window_independent_when_guarded(plant):
    """Paper's well-posedness proposition: on a record that settles, doubling
    the horizon must not move the counts."""
    name, tau, K, L = plant
    Tsim, Ts = sim_grid(tau, L)
    gains = SETTLING_GAINS[name]
    desc = standard_pid_features()

    def counts(horizon, delta):
        feats, _, _, sigs = loop_response_features(
            desc, tau, K, L, Ts, *gains, Tsim=horizon, delta=delta)
        return np.array([f['N'] for f in feats]), sigs

    short, s_sigs = counts(Tsim, 0.02)
    long_, l_sigs = counts(2 * Tsim, 0.02)

    assert s_sigs['k_delta'] < s_sigs['k2'], 'precondition: record must settle'
    assert l_sigs['k_delta'] == s_sigs['k_delta'], 'guard anchors at the same sample'
    np.testing.assert_allclose(short, long_, atol=SIM_ATOL)


@pytest.mark.parametrize('name', ['P2', 'P3', 'P4'])
def test_counts_drift_without_the_guard(name):
    """The converse of the proposition: delta=0 is the unguarded raw-window
    count, and it drifts with the horizon — which is what the guard exists to
    prevent. Measured drift over a 4x horizon is 11.5 (P2), 4.9 (P3) and 10.4
    (P4) turns.

    P1 is excluded deliberately rather than by tolerance-fudging: under GAINS
    its counts are ~(0.12, -0.02, -0.03), i.e. the response barely encircles
    the origin at all, so an unguarded window has nothing to accumulate and
    drifts by only ~0.004. The guard is a no-op where there is no winding.
    """
    from conftest import BATTERY

    tau_list, K, L = BATTERY[name]
    tau = np.asarray(tau_list, dtype=float)
    Tsim, Ts = sim_grid(tau, L)
    desc = standard_pid_features()

    def counts(horizon):
        feats, _, _, _ = loop_response_features(
            desc, tau, K, L, Ts, *GAINS, Tsim=horizon, delta=0.0)
        return np.array([f['N'] for f in feats])

    assert np.max(np.abs(counts(Tsim) - counts(4 * Tsim))) > 1.0


def test_guard_degenerates_on_an_unsettled_record():
    """Documented limitation, not a regression: when the response has not
    settled inside the record, k_delta pins to k2 and the count goes back to
    being window-dependent. P4 under GAINS is still ~13% off setpoint at 4x
    the app's chosen horizon, and nothing in the UI signals this."""
    tau, K, L = np.array([8.0] * 6), 1.0, 4.0
    Tsim, Ts = sim_grid(tau, L)
    desc = standard_pid_features()

    def counts(horizon):
        feats, _, _, sigs = loop_response_features(
            desc, tau, K, L, Ts, *GAINS, Tsim=horizon, delta=0.02)
        return np.array([f['N'] for f in feats]), sigs

    short, s_sigs = counts(Tsim)
    long_, l_sigs = counts(2 * Tsim)

    assert s_sigs['k_delta'] == s_sigs['k2'], 'guard has nothing to anchor to'
    assert l_sigs['k_delta'] == l_sigs['k2']
    assert np.all(np.abs(long_ - short) > 1.0), 'counts scale with the window'


# inf arithmetic is the input under test, so numpy's complaints about it are
# the expected noise of exercising this path, not a signal.
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('bad', [np.inf, -np.inf, np.nan])
def test_non_finite_window_scores_maximally_bad_not_quiet(bad):
    """A diverged window must not come out as the quietest possible trajectory.

    The peak normalization turns inf into 0-or-nan and the winding count of
    that is nothing, so these used to score 0.0 — which every band test reads
    as "quiet" and the tuning rule answers by raising the gains that blew up.
    inf rather than nan so that `N > Nbar` is True and any rule in
    TUNING_RULES cuts a gain instead of falling through the same branch.
    """
    e = np.linspace(-1.0, 0.0, 400)
    e[300:] = bad
    ext = add_derivatives({'e': e}, nd=2)

    feats = compute_features(standard_pid_features(), ext, 5, 380, EPSILON)

    assert [f['N'] for f in feats] == [np.inf] * 3
    for f in feats:
        assert f['N'] > f['Nbar'], 'must trip its band, not slip under it'


def test_finite_windows_are_untouched_by_the_non_finite_guard(plant):
    """The guard must not perturb ordinary records: the golden counts stand."""
    name, tau, K, L = plant
    g = golden(name)
    Tsim, Ts = sim_grid(tau, L)

    feats, _, _, _ = loop_response_features(
        standard_pid_features(), tau, K, L, Ts, *GAINS, Tsim=Tsim)

    np.testing.assert_allclose([f['N'] for f in feats], g['feat_N'],
                               rtol=SIM_RTOL, atol=SIM_ATOL)


def test_descriptor_limits_are_carried_through():
    desc = standard_pid_features(Nbar=(0.3, 0.6, 0.9))
    assert [d.Nbar for d in desc] == [0.3, 0.6, 0.9]
    assert [d.name for d in desc] == ['Gamma0', 'Gamma1', 'Gamma2']
