"""The GUI's proposal/override arbitration for the simulation grid.

These are pure functions pulled out of callbacks.py precisely so they can be
tested without a browser: the Dash wiring around them (propose_grid writes the
Tsim field, commit_grid reads it into grid-store, show_ts reads Ts back out)
has nothing in it but plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from callbacks import _clamp_grid, _grid_matches, _grid_sig, _propose_grid, _resolve_grid
from core.params import N_POINTS


# ── The plant signature ───────────────────────────────────────────────────────

@pytest.mark.parametrize('a,b', [
    ([10.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 10.0]),   # reordered
    ([5.0], [2.5, 2.5]),                               # split
    ([1.0, 2.0, 3.0], [6.0]),                          # collapsed
])
def test_sig_ignores_tau_rearrangements_that_do_not_move_the_grid(a, b):
    """auto_grid consumes sum(tau) + L and nothing else, so these pairs share a
    grid and must share a signature.

    Not cosmetic: update_figures_patch skips a redraw whenever the stored grid's
    signature disagrees with the current plant, and waits for commit_grid to
    catch up. But commit_grid only fires when propose_grid actually changes a
    field value, which for these pairs it does not — so a signature that moved
    here would leave the figures frozen until something else nudged them.
    """
    assert _grid_sig(np.array(a), 0.0) == _grid_sig(np.array(b), 0.0)
    assert _propose_grid(np.array(a), 0.0)['Tsim'] == pytest.approx(
        _propose_grid(np.array(b), 0.0)['Tsim'])


@pytest.mark.parametrize('tau,L', [([5.0], 1.0), ([6.0], 0.0), ([5.0], 0.5)])
def test_sig_separates_plants_with_different_grids(tau, L):
    base = _grid_sig(np.array([5.0]), 0.0)
    assert _grid_sig(np.array(tau), L) != base


# ── Resolution ────────────────────────────────────────────────────────────────

def test_resolve_returns_the_store_when_it_belongs_to_this_plant():
    tau, L = np.array([5.0] * 4), 8.0
    store = _clamp_grid('840', _propose_grid(tau, L))   # a horizon the user typed
    store['sig'] = _grid_sig(tau, L)

    assert _resolve_grid(store, tau, L)['Tsim'] == pytest.approx(840.0)


@pytest.mark.parametrize('store', [None, {}, {'sig': [1.0, 0.0], 'Tsim': 9, 'Ts': 1}])
def test_resolve_falls_back_to_the_proposal(store):
    """Covers the first paint (no store yet) and the beat after a tau edit
    before commit_grid has answered."""
    tau, L = np.array([5.0] * 4), 8.0
    grid = _resolve_grid(store, tau, L)

    assert grid['Tsim'] == pytest.approx(280.0)
    assert grid['Ts'] == pytest.approx(280.0 / (N_POINTS - 1))
    assert _grid_matches(grid, tau, L)


# ── Clamping a hand-entered horizon ───────────────────────────────────────────

@pytest.fixture
def good():
    return _propose_grid(np.array([5.0] * 4), 8.0)   # Tsim 280, Ts 0.5611


def test_clamp_passes_a_reasonable_override_through(good):
    grid = _clamp_grid('840', good)

    assert grid['Tsim'] == pytest.approx(840.0)
    assert grid['notes'] == []


@pytest.mark.parametrize('tsim_raw', ['abc', '', None, '0', '-5', 'nan'])
def test_clamp_rejects_a_bad_horizon(tsim_raw, good):
    grid = _clamp_grid(tsim_raw, good)

    assert grid['Tsim'] == pytest.approx(good['Tsim'])
    assert grid['notes']


@pytest.mark.parametrize('tsim_raw', ['840', '17', 'abc', None])
def test_clamp_always_derives_Ts_from_the_horizon(tsim_raw, good):
    """N is the constant N_POINTS, so Ts is never a decision of its own — no
    input to _clamp_grid, and no route by which the sample count can move.
    """
    grid = _clamp_grid(tsim_raw, good)

    assert grid['Ts'] == pytest.approx(grid['Tsim'] / (N_POINTS - 1))
    assert int(round(grid['Tsim'] / grid['Ts'])) + 1 == N_POINTS


def test_clamp_notes_name_the_value_actually_simulated(good):
    """Corrections are reported, not written back into the field (a Dash
    callback cannot write a property it also reads), so the note is the only
    place the user learns what ran."""
    grid = _clamp_grid('abc', good)

    assert f'{good["Tsim"]:.4g}' in ' '.join(grid['notes'])
