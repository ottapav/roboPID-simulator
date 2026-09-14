"""The AMIGO reference overlay: FOPTD reduction, the rule, and the callback helper."""

from __future__ import annotations

import numpy as np
import pytest

from callbacks import _amigo_reference, _build_time_fig, _gains_label, _patches_from
from core.amigo import amigo_gains, foptd_fit
from core.signals import auto_grid


# ── FOPTD fit ─────────────────────────────────────────────────────────────────

def test_fit_of_a_single_lag_is_the_lag_itself():
    T, L_app = foptd_fit([5.0], 0.0)
    assert T == pytest.approx(5.0, abs=0.01)
    assert L_app == pytest.approx(0.0, abs=1e-9)


def test_fit_carries_dead_time_through_unchanged():
    T, L_app = foptd_fit([5.0], 3.0)
    assert T == pytest.approx(5.0, abs=0.01)
    assert L_app == pytest.approx(3.0, abs=1e-9)


def test_fit_of_four_equal_lags_matches_the_erlang_tangent():
    """Four 5 s lags: steepest point at t = 3*tau, where y = 1 - 13e^-3 = 0.3528
    and the slope is 27e^-3/30, so the tangent crosses zero at 7.127 s; 63% is
    reached at 21.76 s."""
    T, L_app = foptd_fit([5.0] * 4, 8.0)
    assert L_app == pytest.approx(8.0 + 7.127, abs=0.05)
    assert T == pytest.approx(21.76 - 7.127, rel=0.01)


# ── The rule ──────────────────────────────────────────────────────────────────

def test_amigo_pid_gains():
    Kp, Ki, Kd = amigo_gains(2.0, 10.0, 2.0, 'PID')
    assert Kp == pytest.approx(1.225, rel=1e-6)
    assert Ki == pytest.approx(1.225 * 3.0 / 17.6, rel=1e-6)
    assert Kd == pytest.approx(1.225 * 10.0 / 10.6, rel=1e-6)


@pytest.mark.parametrize('ctype', ['PI', 'I'])
def test_amigo_pi_gains_and_i_falls_back_to_pi(ctype):
    Kp, Ki, Kd = amigo_gains(2.0, 10.0, 2.0, ctype)
    Kp_ref = (0.15 + (0.35 - 20.0 / 144.0) * 5.0) / 2.0
    Ti_ref = 0.7 + 2600.0 / 368.0
    assert Kp == pytest.approx(Kp_ref, rel=1e-6)
    assert Ki == pytest.approx(Kp_ref / Ti_ref, rel=1e-6)
    assert Kd == 0.0


# ── Label formatting ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('ctype,expected', [
    ('I', 'SPIN I: Ki=0.05'),
    ('PI', 'SPIN PI: Ki=0.05 Kp=0.60'),
    ('PID', 'SPIN PID: Ki=0.05 Kp=0.60 Kd=0.60'),
])
def test_gains_label_order_is_ki_kp_kd(ctype, expected):
    assert _gains_label('SPIN', ctype, 0.6, 0.05, 0.6) == expected


# ── Callback helper ───────────────────────────────────────────────────────────

P2 = (np.array([5.0] * 4), 1.25, 8.0)


@pytest.mark.parametrize('K', [0.0, -1.0, float('nan')])
def test_no_reference_without_a_positive_gain(K):
    tau, _, L = P2
    Tsim, Ts = auto_grid(tau, L)
    assert _amigo_reference(tau, K, L, Tsim, Ts, 'PID', {}) is None


@pytest.mark.parametrize('ctype,label', [('PID', 'AMIGO PID'), ('PI', 'AMIGO PI:'),
                                         ('I', 'AMIGO PI:')])
def test_reference_settles_on_p2(ctype, label):
    tau, K, L = P2
    Tsim, Ts = auto_grid(tau, L)
    ref = _amigo_reference(tau, K, L, Tsim, Ts, ctype, {'simtype': 0})
    assert ref['label'].startswith(label)
    assert len(ref['t']) == len(ref['y']) == len(ref['u'])
    assert ref['y'][-1] == pytest.approx(1.0, abs=0.02)
    assert ref['u'][-1] == pytest.approx(1.0 / K, rel=0.02)


def test_figure_and_patch_agree_on_trace_layout():
    tau, K, L = P2
    Tsim, Ts = auto_grid(tau, L)
    t = np.linspace(0.0, Tsim, 5)
    sigs = {'t': t, 'y': np.zeros(5), 'u': np.zeros(5)}
    ref = _amigo_reference(tau, K, L, Tsim, Ts, 'PID', {})
    spin_label = _gains_label('SPIN', 'PID', 0.6, 0.05, 0.6)

    fig = _build_time_fig(sigs, ref, spin_label)
    assert [tr.name for tr in fig.data] == ['r', 'y SPIN', 'u SPIN', 'y AMIGO', 'u AMIGO']
    assert [tr.yaxis for tr in fig.data] == [None, None, 'y2', None, 'y2']
    # One style per source: both AMIGO lines match, and r is the thickest.
    assert fig.data[3].line.dash == fig.data[4].line.dash == 'dot'
    assert fig.data[3].line.width == fig.data[4].line.width
    assert fig.data[0].line.dash == 'dash'
    assert fig.data[0].line.width > max(tr.line.width for tr in fig.data[1:])
    # Both halves land in the title, SPIN first, in that order.
    title = fig.layout.title.text
    assert title == f'Step Response · {spin_label} · {ref["label"]}'
    assert title.index(spin_label) < title.index(ref['label'])

    empty = _build_time_fig(sigs, None, None)
    assert len(empty.data) == 5 and len(empty.data[3].x) == 0
    assert empty.layout.title.text == 'Step Response'

    # The tuner's progress patch recomputes both halves every iteration: the
    # AMIGO reference is unchanged (same plant, so it hits the cache), and the
    # SPIN label tracks whatever gains this iteration actually scored.
    feats = [{'xdata': np.zeros(2), 'ydata': np.zeros(2), 'N': 0.0, 'Nbar': 1.0}] * 3
    ops = _patches_from(feats, sigs, ref, spin_label)[3].to_plotly_json()['operations']
    title_ops = [op for op in ops if op['location'] == ['layout', 'title', 'text']]
    assert title_ops and title_ops[0]['params']['value'] == title
