"""Grid selection, derivative augmentation and the stability screen."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import GAINS, SIM_ATOL, SIM_RTOL, golden, sim_grid

from core.params import K1_PADDING, K2_PADDING, N_POINTS, time_grid
from core.signals import (
    add_derivatives, auto_grid, find_index, loop_signals, scaled_variables,
    settling_index,
)

# (Tsim, Ts) proposed for each battery plant. The horizons are unchanged from
# the variable-N grid that preceded this; only the spacing moved, from a flat
# Ts=1 to Tsim/499.
EXPECTED_GRID = {
    'P1': (140.0, 140.0 / 499),
    'P2': (280.0, 280.0 / 499),
    'P3': (150.0, 150.0 / 499),
    'P4': (520.0, 520.0 / 499),
}


def test_grid_for_battery(plant):
    name, tau, K, L = plant
    Tsim, Ts = sim_grid(tau, L)
    exp_T, exp_Ts = EXPECTED_GRID[name]

    assert Tsim == pytest.approx(exp_T)
    assert Ts == pytest.approx(exp_Ts)


@pytest.mark.parametrize('tau,L', [
    ([1e-3], 0.0), ([0.1], 0.0), ([5.0], 0.0),
    ([1000.0], 0.0), ([1e5], 0.0), ([500.0], 200.0), ([10.0, 1, 1, 1], 1.0),
])
def test_grid_is_exactly_N_points(tau, L):
    """The whole contract: every plant costs the same N samples, however fast
    or slow it is. The grid that preceded this ranged from 141 to 10001."""
    Tsim, Ts = auto_grid(np.asarray(tau, dtype=float), L)
    assert len(time_grid(Tsim, Ts)) == N_POINTS


@pytest.mark.parametrize('tau,L', [([1e-3], 0.0), ([5.0], 0.0), ([500.0], 200.0)])
def test_grid_is_scale_free(tau, L):
    """Tsim and Ts both scale with the plant's span, so the transient occupies
    the same fraction of the record at every timescale — that is what makes a
    fixed N enough for a 1 ms plant and a 1000 s one alike."""
    tau = np.asarray(tau, dtype=float)
    span = float(np.sum(tau)) + L
    Tsim, Ts = auto_grid(tau, L)

    assert Tsim / span == pytest.approx(10.0)
    assert Tsim / Ts == pytest.approx(N_POINTS - 1)


@pytest.mark.parametrize('tau,L', [([1e-3], 0.0), ([5.0], 0.0), ([500.0], 200.0)])
def test_grid_resolves_past_the_analysis_padding(tau, L):
    """auto_grid dropped its explicit resolution rule when N became fixed; this
    is the assertion that the fixed N satisfies it anyway. Ts coarser than
    span/(K1+K2) would let the window padding eat the whole transient."""
    tau = np.asarray(tau, dtype=float)
    span = float(np.sum(tau)) + L
    _, Ts = auto_grid(tau, L)

    assert Ts <= span / (K1_PADDING + K2_PADDING)


def test_grid_has_no_horizon_floor():
    """A fast plant must not be stretched over an absolute minimum horizon: at
    the old 50 s floor, tau=[0.1] resolved its own span with one sample."""
    Tsim, Ts = auto_grid(np.array([0.1]), 0.0)
    assert Tsim == pytest.approx(1.0)
    assert Ts < 0.1


def test_settling_guard_active_on_a_full_record():
    """The guard is only meaningful when the record outlives the transient."""
    tau, K, L = np.array([5.0] * 4), 1.25, 8.0
    sigs = loop_signals(tau, K, L, 1.0, *GAINS, Tsim=280.0)
    assert sigs['k1'] < sigs['k_delta'] < sigs['k2']

    short = loop_signals(tau, K, L, 1.0, *GAINS, Tsim=70.0)
    assert short['k_delta'] == short['k2'], 'truncation degenerates the guard'


@pytest.mark.parametrize('bad', [np.inf, -np.inf, np.nan])
def test_find_index_calls_a_non_finite_record_unstable(bad):
    """A record that has overflowed must be reported unstable.

    Every candidate split of a non-finite window scores nan, and the
    all-non-finite fallback used to return "stable" — which triangular_rule
    reads as "all quiet" and answers by raising the gains that diverged. The
    screen was weakest on its worst input: it correctly flags a merely-huge
    record and waved through an infinite one.
    """
    e = np.linspace(-1.0, 0.0, 200)
    e[150:] = bad

    ind, unstable = find_index(5, 194, e)
    assert unstable is True
    assert ind == 5, 'a decisive verdict returns the window start, not the "cannot decide" 2'


def test_find_index_guard_does_not_over_trigger(plant):
    """The non-finite guard must not disturb ordinary records: every golden
    verdict has to survive it unchanged."""
    name, tau, K, L = plant
    g = golden(name)

    ind, unstable = find_index(int(g['k1']), int(g['k2']), g['sig_e'])
    assert ind == int(g['find_ind'])
    assert unstable == bool(g['find_unstable'])


def test_add_derivatives_columns():
    """A4: columns are s, then nd successive diffs, then nd successive cumsums."""
    s = np.arange(1.0, 6.0)
    mat = add_derivatives({'s': s}, nd=2)['s']

    np.testing.assert_allclose(mat[:, 0], s)
    np.testing.assert_allclose(mat[:-1, 1], np.diff(s))
    np.testing.assert_allclose(mat[:-2, 2], np.diff(np.diff(s)))
    np.testing.assert_allclose(mat[:, 3], np.cumsum(s))
    np.testing.assert_allclose(mat[:, 4], np.cumsum(np.cumsum(s)))


def test_add_derivatives_passes_through_scalars():
    out = add_derivatives({'e': np.ones(5), 'k1': 3, 'k2': 4, 'k_delta': 4}, nd=2)
    assert out['k1'] == 3 and out['k2'] == 4 and out['k_delta'] == 4
    assert out['e'].shape == (5, 5)


def test_find_index_golden(plant):
    name, tau, K, L = plant
    g = golden(name)
    sigs = loop_signals(tau, K, L, float(g['Ts']), *GAINS, Tsim=float(g['Tsim']))
    ind, unstable = find_index(sigs['k1'], sigs['k2'], sigs['e'])

    assert ind == int(g['find_ind'])
    assert bool(unstable) == bool(g['find_unstable'])


def test_find_index_flags_a_growing_record():
    """A settling record front-loads its energy; a diverging one back-loads it."""
    n = 300
    decaying = np.exp(-np.linspace(0, 6, n)) * np.sin(np.linspace(0, 40, n))
    growing = np.exp(+np.linspace(0, 6, n)) * np.sin(np.linspace(0, 40, n))

    assert not find_index(0, n - 1, decaying)[1]
    assert find_index(0, n - 1, growing)[1]


def test_loop_signals_arrays_are_all_the_same_length(plant):
    """A5: everything in the dict must share one time base."""
    _, tau, K, L = plant
    Tsim, Ts = sim_grid(tau, L)
    sigs = loop_signals(tau, K, L, Ts, *GAINS, Tsim=Tsim)

    n = len(sigs['t'])
    for key, val in sigs.items():
        if isinstance(val, np.ndarray):
            assert len(val) == n, f'{key} has length {len(val)}, expected {n}'


def test_loop_signals_goldens(plant):
    name, tau, K, L = plant
    g = golden(name)
    sigs = loop_signals(tau, K, L, float(g['Ts']), *GAINS, Tsim=float(g['Tsim']))

    for key in ('e', 'v', 'uP', 'uI', 'uD'):
        np.testing.assert_allclose(sigs[key], g[f'sig_{key}'], rtol=SIM_RTOL, atol=SIM_ATOL,
                                   err_msg=f'signal {key} moved')
    assert (sigs['k1'], sigs['k2'], sigs['k_delta']) == (
        int(g['k1']), int(g['k2']), int(g['k_delta']))


def test_scaled_variables_window_ordering(plant):
    _, tau, K, L = plant
    Tsim, Ts = sim_grid(tau, L)
    sigs = loop_signals(tau, K, L, Ts, *GAINS, Tsim=Tsim)
    e, v, k1, k2 = scaled_variables(sigs['y'], sigs['u'], np.ones_like(sigs['y']))

    assert 0 <= k1 <= k2 < len(e)
    assert settling_index(e, k1, k2, 0.02) <= k2
    assert settling_index(e, k1, k2, 0.0) == k2, 'delta=0 is the unguarded case'
