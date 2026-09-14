"""The admissibility gates and the runtime detectors."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import BATTERY

from core import admissibility as adm
from core.admissibility import (
    ACTIVE_GAINS, LAGS_REQUIRED, PHI_BY_CTYPE, check_bands, check_phase,
    check_plant, check_reachability, check_sign, diagnose_run, relative_degree,
)
from core.amigo import amigo_design

CTYPES = ('I', 'PI', 'PID', 'PID_UNFILTERED')
GUI_CTYPES = ('I', 'PI', 'PID')
BOX = (0.01, 10.0)
UNIT = {'Kp': 1.0, 'Ki': 1.0, 'Kd': 1.0}

# The app's own defaults (app.py --tau/--K/--L), i.e. what a first-time visitor
# clicks TUNE on.
APP_DEFAULT = ([5.0, 5.0, 5.0, 5.0], 1.25, 8.0)


def _target(tau, K, L, ctype):
    return amigo_design(tau, K, L, ctype)


def _feats(n0, n1, n2, nbar=(0.5, 0.75, 1.0)):
    return [{'N': n, 'Nbar': b} for n, b in zip((n0, n1, n2), nbar)]


# ── Gate 1: phase ─────────────────────────────────────────────────────────────

def test_the_degree_table_is_derived_not_typed():
    """LAGS_REQUIRED is built from -(pi/2)*r + phi_inf < -pi. Pinning the
    values here is what stops the formula and the documented table drifting
    apart if one of them is edited."""
    assert LAGS_REQUIRED == {'I': 2, 'PI': 3, 'PID': 3, 'PID_UNFILTERED': 4}
    for ctype, r_req in LAGS_REQUIRED.items():
        phi = PHI_BY_CTYPE[ctype]
        assert -np.pi / 2 * r_req + phi < -np.pi, 'requirement must satisfy it'
        assert not -np.pi / 2 * (r_req - 1) + phi < -np.pi, 'and be the least'


@pytest.mark.parametrize('ctype', CTYPES)
@pytest.mark.parametrize('n', (1, 2, 3, 4, 5, 6))
def test_phase_gate_matches_the_degree_table(ctype, n):
    """A delay-free chain of n lags passes iff its relative degree meets the
    requirement. This is the whole gate."""
    finding = check_phase([5.0] * n, 0.0, ctype)
    assert (finding is None) == (n >= LAGS_REQUIRED[ctype])


@pytest.mark.parametrize('ctype', CTYPES)
def test_the_marginal_band_is_refused(ctype):
    """One degree short is the case where the asymptote lands exactly on -pi.
    The crossing is then decided by O(1/w) terms the method does not evaluate,
    so it is refused rather than guessed at -- and the message must say that
    rather than claim the plant has no boundary."""
    finding = check_phase([1.0] * (LAGS_REQUIRED[ctype] - 1), 0.0, ctype)
    assert finding is not None
    text = ' '.join(finding.detail)
    assert 'conservative' in text
    assert 'may well be tunable' in text


def test_relative_degree_counts_lags():
    assert relative_degree([5.0]) == 1
    assert relative_degree([1.0, 2.0, 3.0]) == 3
    assert relative_degree(np.array([8.0] * 6)) == 6


@pytest.mark.parametrize('tau', ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0],
                                 [1e-3, 1.0, 1e3]))
def test_only_the_count_matters_not_the_values(tau):
    """Relative degree is structural: the same number of lags gives the same
    verdict however fast or disparate they are. The sweep this replaced could
    not offer that -- its answer moved with where the corners landed."""
    assert check_phase(tau, 0.0, 'PID') is None
    assert check_phase(tau[:2], 0.0, 'PID') is not None


@pytest.mark.parametrize('ctype', CTYPES)
@pytest.mark.parametrize('L', (1e-12, 1e-9, 1e-3, 1.0, 100.0))
def test_any_dead_time_satisfies_every_controller(ctype, L):
    """A dead time makes the phase unbounded below, so the asymptote question
    does not arise and one lag is enough for every row. Structural, so even an
    L far too small for a frequency grid to resolve is decided correctly."""
    assert check_phase([5.0], L, ctype) is None


def test_the_gate_takes_no_grid():
    """No Ts, no Tsim, no sweep resolution: the verdict cannot move with the
    simulation grid, and cannot be contaminated by a discretized model's own
    ZOH lag. That independence is the reason for the structural form."""
    import inspect
    params = set(inspect.signature(check_phase).parameters)
    assert params == {'tau', 'L', 'ctype'}


@pytest.mark.parametrize('name', sorted(BATTERY))
@pytest.mark.parametrize('ctype', GUI_CTYPES)
def test_battery_is_admissible(name, ctype):
    tau, K, L = BATTERY[name]
    assert check_plant(tau, K, L, ctype, UNIT, BOX).ok


@pytest.mark.parametrize('ctype', GUI_CTYPES)
def test_app_default_plant_is_admissible(ctype):
    """The shipped landing state must never open the modal."""
    tau, K, L = APP_DEFAULT
    assert check_plant(tau, K, L, ctype, UNIT, BOX).ok


def test_two_lags_delay_free_is_refused_although_it_does_tune():
    """tau=[1,2], K=1, L=0 -- the case that exposed the sweep.

    This plant has a real, reachable stability boundary. Routh on
    2s^3 + 3s^2 + (1+K*Kp)s + K*Ki gives instability for K*Ki > 1.5*(1+K*Kp),
    which the simulation confirms: Kp=Kd=0.01 is stable at Ki=1 and diverges at
    Ki=2, well inside the default box. Run the tuner on it under PID and it
    converges to F=(10, 1.372, 9.139) with peak|y| = 1.026 and band 2 active at
    N2 = 1.4.

    It is refused anyway: r = 2 against the 3 that filtered PID requires. That
    is the conservative band doing its job -- the asymptote sits exactly on
    -180 deg and the method declines to rest a verdict on the O(1/w) terms that
    decide it there.

    So this test asserts a refusal that is *known to be over-strict*, on
    purpose. If a later change makes it pass, that is a decision about what the
    method claims, not a bug fix -- which is exactly why it is pinned here with
    the evidence attached.
    """
    tau, K, L = [1.0, 2.0], 1.0, 0.0
    assert relative_degree(tau) == 2
    assert check_phase(tau, L, 'PID') is not None
    assert check_phase(tau, L, 'PI') is not None
    assert check_phase(tau, L, 'I') is None          # I needs only 2

    verdict = check_plant(tau, K, L, 'PID', UNIT, BOX)
    assert not verdict.ok
    assert [f.gate for f in verdict.findings] == ['phase']


# ── Gate 0: sign ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('K', (0.0, -1.0, -1e-9, float('nan'), float('inf')))
def test_non_positive_or_unreadable_K_blocks(K):
    verdict = check_plant([5.0, 5.0, 5.0], K, 1.0, 'PID', UNIT, BOX)
    assert not verdict.ok
    assert [f.gate for f in verdict.findings] == ['sign']


def test_sign_gate_runs_before_the_phase_gate():
    """A negative K adds +-pi to arg P. Were the sweep to see it first, a
    one-lag plant would clear the phase requirement on a sign flip rather than
    on real lag, so the order here is load-bearing, not cosmetic."""
    verdict = check_plant([5.0], -1.0, 0.0, 'PID', UNIT, BOX)
    assert [f.gate for f in verdict.findings] == ['sign']


@pytest.mark.parametrize('K', (0.01, 1.0, 1000.0))
def test_positive_K_passes_the_sign_gate(K):
    assert check_sign(K) is None


# ── The AMIGO target ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', sorted(BATTERY))
@pytest.mark.parametrize('ctype', GUI_CTYPES)
def test_every_battery_plant_has_a_target(name, ctype):
    tau, K, L = BATTERY[name]
    t = _target(tau, K, L, ctype)
    assert t is not None
    assert t.T > 0 and t.L > 0
    assert t.structure == ('PID' if ctype == 'PID' else 'PI')


@pytest.mark.parametrize('ctype', CTYPES)
def test_delay_free_plants_passing_gate_1_still_get_a_target(ctype):
    """The AMIGO rule divides by L. Gate 1 lets a delay-free plant through only
    with at least two lags, and such a chain always fits a positive apparent
    dead time -- so the target exists wherever check_plant asks for it."""
    tau = [5.0] * LAGS_REQUIRED[ctype]
    assert check_phase(tau, 0.0, ctype) is None
    assert _target(tau, 1.0, 0.0, ctype) is not None


def test_no_target_where_the_rule_is_undefined():
    assert _target([5.0], 1.0, 0.0, 'PID') is None     # L_app = 0
    assert _target([5.0], -1.0, 1.0, 'PID') is None    # K not positive


def test_target_gains_scale_inversely_with_plant_gain():
    tau, L = [5.0, 5.0, 5.0], 1.0
    t1, t4 = _target(tau, 1.0, L, 'PID'), _target(tau, 4.0, L, 'PID')
    for name in ('Kp', 'Ki', 'Kd'):
        assert getattr(t4, name) == pytest.approx(getattr(t1, name) / 4.0, rel=1e-12)
    assert (t4.T, t4.L) == (t1.T, t1.L)   # the fit is gain-independent


# ── Gate 2: reachability ──────────────────────────────────────────────────────

def test_reachability_boundary_on_a_first_order_plant():
    """Unit start gains, default box: the AMIGO PID target leaves the box below
    L/tau ~ 0.108 (Ki = 8.5 at L = 0.12, 11.2 at L = 0.10) and sits inside it
    above. The binding gain is Ki -- AMIGO's reset grows as L shrinks and is the
    first to run off the top of the box."""
    inside = check_reachability(_target([1.0], 1.0, 0.12, 'PID'), 'PID', UNIT, BOX)
    outside = check_reachability(_target([1.0], 1.0, 0.10, 'PID'), 'PID', UNIT, BOX)

    assert inside == []
    assert [f.gate for f in outside] == ['reach']
    assert outside[0].title.startswith('Ki')


def test_reachability_is_advisory_not_blocking():
    verdict = check_plant([1.0], 1.0, 0.05, 'PID', UNIT, BOX)
    assert verdict.ok
    assert verdict.warnings and not verdict.blocking


def test_widening_the_box_clears_a_reach_finding():
    tau, K, L = [1.0], 1.0, 0.05
    assert check_reachability(_target(tau, K, L, 'PID'), 'PID', UNIT, BOX)
    assert check_reachability(_target(tau, K, L, 'PID'), 'PID', UNIT,
                              (0.01, 1000.0)) == []


@pytest.mark.parametrize('start', (0.02, 0.5, 1.0, 7.0, 500.0))
def test_reachability_does_not_depend_on_where_the_slider_starts(start):
    """The multiplier bounds are (Kmin/g_start, Kmax/g_start), so the reachable
    set is exactly the absolute gain box however the sliders are placed. The
    message says as much, and would be giving bad advice if this ever moved."""
    target = _target([1.0], 1.0, 0.05, 'PID')
    gains = {'Kp': start, 'Ki': start, 'Kd': start}
    reported = {f.title.split()[0]
                for f in check_reachability(target, 'PID', gains, BOX)}
    expected = {n for n in ACTIVE_GAINS['PID']
                if not BOX[0] <= getattr(target, n) <= BOX[1]}
    assert reported == expected


@pytest.mark.parametrize('ctype', GUI_CTYPES)
def test_reachability_only_reports_gains_the_structure_uses(ctype):
    findings = check_reachability(_target([1.0], 1.0, 0.001, ctype), ctype,
                                  UNIT, BOX)
    reported = {f.title.split()[0] for f in findings}
    assert reported <= set(ACTIVE_GAINS[ctype])


# ── Gate 3: band non-degeneracy ───────────────────────────────────────────────

def test_band_gate_passes_at_the_shipped_constants():
    """Structural: band 2 spans exactly nu = DERIV_FILTER_N, so on a shipped
    build the gate cannot fire."""
    assert adm.DERIV_FILTER_N >= adm.NU_MIN
    assert check_bands('PID') == []


def test_band_gate_fires_when_the_roll_off_is_lowered(monkeypatch):
    monkeypatch.setattr(adm, 'DERIV_FILTER_N', 2.0)
    findings = check_bands('PID')
    assert [f.gate for f in findings] == ['filter']


@pytest.mark.parametrize('ctype', ('I', 'PI'))
def test_band_gate_is_silent_without_a_derivative(ctype, monkeypatch):
    monkeypatch.setattr(adm, 'DERIV_FILTER_N', 2.0)
    assert check_bands(ctype) == []


# ── Runtime detectors ─────────────────────────────────────────────────────────

def _lim(F=(0.01, 10.0)):
    return {'Kp': F, 'Ki': F, 'Kd': F}


def test_detector_A_ceiling_with_every_band_quiet():
    findings = diagnose_run('PID', {'Kp': 10.0, 'Ki': 10.0, 'Kd': 10.0}, _lim(),
                            {'Kp': 10.0, 'Ki': 10.0, 'Kd': 10.0},
                            _feats(0.0, 0.0, 0.0))
    assert [f.gate for f in findings] == ['ceiling']


def test_detector_A_sends_the_user_to_the_box_not_to_the_plant():
    """Only a plant that passed Gate 1 can reach this detector, and passing
    means a boundary exists — so it is above Kmax, not absent. The message used
    to say the opposite ("widening will not help"), which the structural gate
    made false."""
    finding = diagnose_run('PID', {'Kp': 10.0, 'Ki': 10.0, 'Kd': 10.0}, _lim(),
                           {'Kp': 10.0, 'Ki': 10.0, 'Kd': 10.0},
                           _feats(0.0, 0.0, 0.0))[0]
    text = ' '.join(finding.detail)
    assert 'will not help' not in text
    assert 'does exist' in text
    assert any('Kmax' in f for f in finding.fixes)


def test_detector_A_needs_every_multiplier_at_the_ceiling():
    findings = diagnose_run('PID', {'Kp': 10.0, 'Ki': 3.0, 'Kd': 10.0}, _lim(),
                            {'Kp': 10.0, 'Ki': 3.0, 'Kd': 10.0},
                            _feats(0.0, 0.0, 0.0))
    assert findings == []


def test_detector_B_names_the_pinned_gain_and_its_band():
    findings = diagnose_run('PID', {'Kp': 0.01, 'Ki': 3.0, 'Kd': 3.0}, _lim(),
                            {'Kp': 0.01, 'Ki': 3.0, 'Kd': 3.0},
                            _feats(0.0, 2.0, 0.0))
    assert [f.gate for f in findings] == ['bound']
    assert findings[0].title.startswith('Kp')


def test_detector_B_ignores_a_ringing_band_that_is_not_pinned():
    findings = diagnose_run('PID', {'Kp': 1.0, 'Ki': 1.0, 'Kd': 1.0}, _lim(),
                            {'Kp': 1.0, 'Ki': 1.0, 'Kd': 1.0},
                            _feats(0.0, 2.0, 0.0))
    assert findings == []


def test_detectors_are_silent_on_an_ordinary_run():
    assert diagnose_run('PID', {'Kp': 1.3, 'Ki': 0.4, 'Kd': 0.9}, _lim(),
                        {'Kp': 1.3, 'Ki': 0.4, 'Kd': 0.9},
                        _feats(0.1, 0.2, 0.3)) == []


@pytest.mark.parametrize('ctype', GUI_CTYPES)
def test_detectors_ignore_gains_the_structure_does_not_tune(ctype):
    """Kp and Kd are held at base 0 under I, so their multipliers are
    meaningless -- reporting one would send the user after a slider the run
    never touched."""
    mult = {'Kp': 10.0, 'Ki': 1.0, 'Kd': 10.0}
    findings = diagnose_run(ctype, mult, _lim(), mult, _feats(0.0, 2.0, 2.0))
    reported = {f.title.split()[0] for f in findings}
    assert reported <= set(ACTIVE_GAINS[ctype])


def test_detector_B_fires_on_an_infinite_count():
    """compute_features scores a diverged record as inf so any rule cuts; the
    detector has to read that as ringing rather than choke on it."""
    findings = diagnose_run('PI', {'Kp': 0.01, 'Ki': 1.0, 'Kd': 0.0}, _lim(),
                            {'Kp': 0.01, 'Ki': 1.0, 'Kd': 0.0},
                            _feats(0.0, float('inf'), 0.0))
    assert [f.gate for f in findings] == ['bound']


def test_diagnose_run_tolerates_a_run_with_no_features():
    assert diagnose_run('PID', {'Kp': 1.0, 'Ki': 1.0, 'Kd': 1.0}, _lim(),
                        {'Kp': 1.0, 'Ki': 1.0, 'Kd': 1.0}, []) == []


# ── The messages themselves ───────────────────────────────────────────────────
# These gates exist to explain a refusal. A message that has decayed into
# "invalid input" is a regression even though every branch still fires
# correctly, so the content is asserted rather than just the control flow.

def _all_findings():
    yield check_sign(-1.0, K_raw='-1')
    yield check_sign(float('nan'), K_raw='abc')
    yield check_phase([5.0], 0.0, 'PID')
    yield from check_reachability(_target([1.0], 1.0, 0.05, 'PID'), 'PID',
                                  UNIT, BOX)
    yield from diagnose_run('PID', {'Kp': 10.0, 'Ki': 10.0, 'Kd': 10.0}, _lim(),
                            {'Kp': 10.0, 'Ki': 10.0, 'Kd': 10.0},
                            _feats(0.0, 0.0, 0.0))
    yield from diagnose_run('PID', {'Kp': 0.01, 'Ki': 3.0, 'Kd': 3.0}, _lim(),
                            {'Kp': 0.01, 'Ki': 3.0, 'Kd': 3.0},
                            _feats(0.0, 2.0, 0.0))


@pytest.mark.parametrize('finding', list(_all_findings()),
                         ids=lambda f: f.gate + ('!' if f.blocking else ''))
def test_every_finding_says_what_and_why(finding):
    assert finding.title.strip()
    assert finding.detail, 'a finding with no explanation is a bare refusal'
    assert all(p.strip() for p in finding.detail)
    assert finding.status.strip()


@pytest.mark.parametrize('finding', [f for f in _all_findings() if f.blocking],
                         ids=lambda f: f.gate)
def test_every_blocking_finding_offers_a_way_out(finding):
    assert finding.fixes


def test_phase_message_quotes_the_structural_numbers():
    finding = check_phase([5.0, 5.0], 0.0, 'PID')
    text = ' '.join(finding.detail)
    assert 'relative degree 3' in text                # what was required
    assert 'r = 2' in text                            # what the plant has
    assert '−180°' in text                            # the requirement itself
    assert '−180°' in text.split('tends to')[1]       # and the asymptote reached
    assert any('relative degree ≥ 3' in f for f in finding.fixes)


def test_phase_message_does_not_claim_the_plant_is_uncontrollable():
    """The refusal is a statement about the method's guarantee, not about the
    plant. In the marginal band the message must not assert there is no
    boundary, because there may well be one."""
    marginal = ' '.join(check_phase([1.0, 2.0], 0.0, 'PID').detail)
    assert 'may well be tunable' in marginal
    assert 'SPIN does not claim it' in marginal

    # Below the marginal band the stronger claim is fair game.
    clear = ' '.join(check_phase([1.0], 0.0, 'PI').detail)
    assert 'no target to reach' in clear


def test_reach_message_quotes_the_target_and_its_provenance():
    finding = check_reachability(_target([1.0], 1.0, 0.05, 'PID'), 'PID',
                                 UNIT, BOX)[0]
    text = ' '.join(finding.detail)
    assert 'AMIGO PID' in text and 'FOPTD' in text
    assert 'T = ' in text and 'L = ' in text
    assert 'still valid' in text
    assert finding.fixes


def test_sign_message_quotes_the_offending_field():
    assert 'wobble' in ' '.join(check_sign(float('nan'), K_raw='wobble').detail)
    assert '-2' in check_sign(-2.0).title
