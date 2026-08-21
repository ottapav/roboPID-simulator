"""Simulation regressions against the recorded goldens."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import GAINS, GAINS_SAT, RTOL, SIM_ATOL, SIM_RTOL, golden, sim_grid

from core.params import time_grid
from core.pid import pid_response_linear, pid_response_awup, action_components
from core.plant import plant_step_response


def test_linear_step_response(plant):
    name, tau, K, L = plant
    g = golden(name)
    y, u, t, Kp, Ki, Kd = pid_response_linear(
        tau, K, L, *GAINS, float(g['Tsim']), float(g['Ts']))

    np.testing.assert_allclose(y, g['lin_y'], rtol=SIM_RTOL, atol=SIM_ATOL)
    np.testing.assert_allclose(u, g['lin_u'], rtol=SIM_RTOL, atol=SIM_ATOL)
    np.testing.assert_allclose(t, g['lin_t'], rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose([Kp, Ki, Kd], g['gains'], rtol=RTOL, atol=RTOL)


def test_awup_step_response(plant):
    """The saturating anti-windup path (simtype=1, what robopid.config ships)."""
    name, tau, K, L = plant
    g = golden(name)
    y, u, _, _, _, _ = pid_response_awup(
        tau, K, L, *GAINS_SAT, float(g['Tsim']), float(g['Ts']),
        minu=-1.0, maxu=1.0)

    np.testing.assert_allclose(y, g['awup_y'], rtol=SIM_RTOL, atol=SIM_ATOL)
    np.testing.assert_allclose(u, g['awup_u'], rtol=SIM_RTOL, atol=SIM_ATOL)
    assert np.any(np.isclose(np.abs(u), 1.0)), 'gains should drive u into saturation'


def test_plant_step_response(plant):
    name, tau, K, L = plant
    g = golden(name)
    y, t = plant_step_response(tau, K, L, float(g['Tsim']), float(g['Ts']))

    np.testing.assert_allclose(y, g['plant_step'], rtol=SIM_RTOL, atol=SIM_ATOL)
    assert len(t) == len(y)


def test_plant_step_settles_to_K(plant):
    """Sanity independent of the goldens: an FOPTD plant settles at K."""
    _, tau, K, L = plant
    Tsim, Ts = sim_grid(tau, L)
    y, _ = plant_step_response(tau, K, L, Tsim * 3, Ts)
    assert y[-1] == pytest.approx(K, rel=1e-3)


@pytest.mark.parametrize('Tsim,Ts', [(151.5, 1.0), (100.0, 0.66), (50.0, 0.33),
                                     (280.0, 280.0 / 499)])
def test_action_components_match_response_length(Tsim, Ts):
    """A5: the P/I/D decomposition must be the same length as the response it
    decomposes, and both must be the length params.time_grid says. The first
    three grids land Tsim/Ts on a half sample, where the awup sample count and
    a rebuilt arange grid used to disagree; the last is a proposed grid, where
    the ratio is 499 give or take float noise."""
    tau, K, L = np.array([5.0, 5.0]), 1.0, 4.0
    y, u, t, Kp, Ki, Kd = pid_response_awup(tau, K, L, *GAINS, Tsim, Ts)
    uP, uI, uD = action_components(y, Kp, Ki, Kd, Ts)

    assert len(uP) == len(uI) == len(uD) == len(y) == len(t)
    assert len(t) == len(time_grid(Tsim, Ts))


def test_gains_are_physical_not_per_sample():
    """Kp/Ki/Kd are continuous-time, so refining the grid must converge on one
    response rather than retune the loop.

    This is the invariant the whole plant-adaptive grid rests on. Ts is now
    chosen per plant and editable in the GUI, and per-sample gains would make
    every one of those a silent retune: halving Ts doubles the number of
    integrator steps, so without the Ki*Ts factor the integral action doubles
    with it. Convergence is the sharp form of the claim — a per-sample
    controller does not merely differ from this by a constant, it fails to
    converge at all, holding its error near 100% of peak at every refinement
    while the ratio asserted below stays at 1.

    L=0 deliberately: with dead time the error is not monotone at coarse Ts,
    because nd = round(L/Ts) quantizes the delay in whole samples and only
    stops jumping once nd is large. test_gains_are_physical_with_dead_time
    covers that case with a plain bound instead.
    """
    tau, K, L = np.array([5.0] * 4), 1.25, 0.0
    Tsim = 280.0

    errs = []
    for k in (1, 2, 4, 8):
        Ts = Tsim / 499 / k
        y1, _, _, *_ = pid_response_linear(tau, K, L, *GAINS, Tsim, Ts)
        y2, _, _, *_ = pid_response_linear(tau, K, L, *GAINS, Tsim, Ts / 2)
        # The fine grid contains the coarse grid's instants at every 2nd sample.
        errs.append(np.max(np.abs(y2[::2] - y1)))

    peak = np.max(np.abs(y1))
    assert errs[0] < 0.03 * peak, 'even the coarsest grid is within a few %'
    for coarse, fine in zip(errs, errs[1:]):
        assert 1.7 < coarse / fine < 2.3, (
            f'first-order convergence expected, got ratios '
            f'{[round(a / b, 2) for a, b in zip(errs, errs[1:])]}')


def test_gains_are_physical_with_dead_time():
    """As above for a delayed plant, where round(L/Ts) makes the refinement
    non-monotone: assert the bound rather than the convergence order."""
    tau, K, L = np.array([5.0] * 4), 1.25, 8.0
    Tsim, Ts = 280.0, 280.0 / 499

    y1, _, _, *_ = pid_response_linear(tau, K, L, *GAINS, Tsim, Ts)
    y2, _, _, *_ = pid_response_linear(tau, K, L, *GAINS, Tsim, Ts / 2)

    np.testing.assert_allclose(y2[::2], y1, rtol=0, atol=0.05 * np.max(np.abs(y1)))


def test_action_components_are_grid_invariant():
    """The P/I/D split carries the same Ts factors as the controller that
    produced u, so it too must survive a change of grid.

    uD is exempt at the step edge: it differences the reference jump, and
    (Kd/Ts)*Δr is a one-sample impulse whose height is inversely proportional
    to Ts by construction. Away from the edge it converges like the rest, which
    is exactly the signature of a correct 1/Ts rather than a missing one.
    """
    tau, K, L = np.array([5.0] * 4), 1.25, 0.0
    Tsim, Ts = 280.0, 280.0 / 499

    y1, _, _, Kp, Ki, Kd = pid_response_linear(tau, K, L, *GAINS, Tsim, Ts)
    y2, _, _, *_ = pid_response_linear(tau, K, L, *GAINS, Tsim, Ts / 2)
    coarse = action_components(y1, Kp, Ki, Kd, Ts)
    fine = action_components(y2, Kp, Ki, Kd, Ts / 2)

    for name, a, b in zip('PID', coarse, fine):
        np.testing.assert_allclose(
            b[::2][3:], a[3:], rtol=0, atol=0.03 * np.max(np.abs(a)),
            err_msg=f'u{name} moved with the grid')


def _nyquist_flips(u):
    """Sample-to-sample direction reversals in u. A settling response has a
    handful (its genuine turning points); ringing at the Nyquist frequency
    reverses at essentially every sample."""
    du = np.diff(np.asarray(u, dtype=float))
    return int(np.sum(np.sign(du[:-1]) * np.sign(du[1:]) < 0))


@pytest.mark.parametrize('sim,kw', [
    (pid_response_linear, {}),
    (pid_response_awup, dict(minu=-10.0, maxu=10.0)),
])
@pytest.mark.parametrize('Kd', [1.0, 1.1, 2.0, 10.0])
def test_derivative_does_not_ring_at_nyquist(sim, kw, Kd):
    """An unfiltered discrete derivative is not implementable at small Ts.

    D(z) = (Kd/Ts)(1 - z^-1) has Nyquist gain 2*Kd/Ts, unbounded as the grid is
    refined. Taken on the output it forms an inner loop P/(1 + D*P) whose
    characteristic polynomial tends to (z - 1)(z + Kd*K/tau), i.e. a pole at
    exactly -Kd*K/tau: on this plant (tau = K = 1) the default Kd = 1 sat on
    the unit circle and Kd = 1.1 diverged to 1e40. Refining the grid made it
    worse, not better, which is how it surfaced.

    Both realizations are checked because both had it: the TF path through its
    inner loop, the velocity-form loop through its raw second difference.
    """
    tau, K, L = np.array([1.0]), 1.0, 0.0
    Tsim, Ts = 10.0, 10.0 / 999

    y, u, t, *_ = sim(tau, K, L, 1.0, 1.0, Kd, Tsim, Ts, **kw)

    assert np.all(np.isfinite(u)) and np.max(np.abs(u)) < 100.0, 'control diverged'
    assert _nyquist_flips(u) < 0.05 * len(u), 'u reverses direction nearly every sample'
    assert y[-1] == pytest.approx(1.0, abs=0.3), 'response never reaches setpoint'


def test_derivative_rolloff_is_bounded_and_grid_free():
    """The roll-off caps |D| at DERIV_FILTER_N whatever Kd and Ts are — that
    bound, not mere damping, is what removes the Nyquist pole. Tf = Kd/N
    carries no Ts, so the cap does not drift when the grid changes."""
    from core.params import DERIV_FILTER_N
    from core.pid import _deriv_tf

    for Kd in (0.01, 1.0, 10.0):
        for Ts in (1.0, 0.02, 0.001):
            num, den = _deriv_tf(Kd, Ts)
            # |D(z)| at z = -1, the Nyquist frequency.
            gain = abs(np.polyval(num, -1.0) / np.polyval(den, -1.0))
            assert gain <= DERIV_FILTER_N + 1e-9, (Kd, Ts, gain)


def test_deriv_filter_disabled_recovers_the_raw_difference():
    """N <= 0 turns the roll-off off, which must reproduce (Kd/Ts)(1 - z^-1)
    exactly — the property that makes the filter a strict generalization of
    what came before rather than a different derivative."""
    from core.pid import _deriv_tf

    num, den = _deriv_tf(2.0, 0.5, N=0.0)
    np.testing.assert_allclose(num / den[0], [2.0 / 0.5, -2.0 / 0.5])
    np.testing.assert_allclose(den / den[0], [1.0, 0.0])


def test_noise_is_reproducible_with_a_seeded_rng():
    """D2: the same generator seed must give the same noisy record. Without
    this, no run with Output noise enabled can be repeated or tested."""
    tau, K, L, Tsim, Ts = np.array([5.0] * 4), 1.25, 8.0, 280.0, 1.0
    kw = dict(dist_a=0.9, dist_b=0.05)

    a, _, _, _, _, _ = pid_response_linear(
        tau, K, L, *GAINS, Tsim, Ts, rng=np.random.default_rng(7), **kw)
    b, _, _, _, _, _ = pid_response_linear(
        tau, K, L, *GAINS, Tsim, Ts, rng=np.random.default_rng(7), **kw)
    c, _, _, _, _, _ = pid_response_linear(
        tau, K, L, *GAINS, Tsim, Ts, rng=np.random.default_rng(8), **kw)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), 'a different seed must give a different record'


def test_awup_noise_is_reproducible_with_a_seeded_rng():
    tau, K, L, Tsim, Ts = np.array([5.0] * 4), 1.25, 8.0, 280.0, 1.0
    kw = dict(dist_a=0.9, dist_b=0.05, minu=-1.0, maxu=1.0)

    a, _, _, _, _, _ = pid_response_awup(
        tau, K, L, *GAINS, Tsim, Ts, rng=np.random.default_rng(3), **kw)
    b, _, _, _, _, _ = pid_response_awup(
        tau, K, L, *GAINS, Tsim, Ts, rng=np.random.default_rng(3), **kw)

    np.testing.assert_array_equal(a, b)


def test_noiseless_runs_need_no_seed():
    """dist_b=0 must stay deterministic without anyone passing an rng."""
    tau, K, L = np.array([5.0] * 4), 1.25, 8.0
    a, _, _, _, _, _ = pid_response_linear(tau, K, L, *GAINS, 280.0, 1.0)
    b, _, _, _, _, _ = pid_response_linear(tau, K, L, *GAINS, 280.0, 1.0)
    np.testing.assert_array_equal(a, b)
