"""Plant-parameter parsing and the shared defaults."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.params import (
    DEFAULT_TAU, GAIN_BOX, N_ITER_BY_CTYPE, NBAR, TAU_MIN,
    fmt2, gain_slider_marks, parse_tau,
)
from core.plant import plant_tf


@pytest.mark.parametrize('text,expected', [
    ('[5,5,5,5]', [5.0, 5.0, 5.0, 5.0]),
    ('10', [10.0]),
    (' [2, 1] ', [2.0, 1.0]),
    ('(3, 4)', [3.0, 4.0]),
])
def test_parse_tau_accepts_valid_input(text, expected):
    tau, notes = parse_tau(text)
    np.testing.assert_allclose(tau, expected)
    assert notes == []


@pytest.mark.parametrize('text', ['', 'abc', '[', 'None', '[]'])
def test_parse_tau_falls_back_and_explains(text):
    """A3: unusable input must produce the default plus a visible note, never
    an exception and never a silent substitution."""
    tau, notes = parse_tau(text)
    np.testing.assert_allclose(tau, DEFAULT_TAU)
    assert notes, f'{text!r} should have produced a warning note'


@pytest.mark.parametrize('text', ['0', '[0, 5]', '-3', '[-1, 2]'])
def test_parse_tau_clamps_non_positive(text):
    """A3: tau=0 divides by zero in plant_tf; tau<0 puts a pole outside the
    unit circle. Both are clamped, and the clamp is reported."""
    tau, notes = parse_tau(text)
    assert np.all(tau >= TAU_MIN)
    assert any('clamped' in n for n in notes)


def test_parse_tau_drops_non_finite():
    tau, notes = parse_tau('[1, 2]')
    assert notes == []
    tau, notes = parse_tau("[float('nan')]") if False else parse_tau('[1e400, 2]')
    assert np.all(np.isfinite(tau))
    assert any('non-finite' in n for n in notes)


@pytest.mark.parametrize('text', ['0', '[0, 5]', '-3', '[]', 'abc', '1e400'])
def test_parsed_tau_always_builds_a_usable_plant(text):
    """The whole point of A3: whatever the user types, plant_tf must not raise
    and must not produce NaN."""
    tau, _ = parse_tau(text)
    num, den = plant_tf(tau, 1.0, 8.0, 1.0)
    assert np.all(np.isfinite(num)) and np.all(np.isfinite(den))
    assert np.max(np.abs(np.roots(den))) <= 1.0 + 1e-9, 'no poles outside the unit circle'


@pytest.mark.parametrize('value,expected', [
    (1.0, '1.00'), (0.5, '0.50'), (280.0, '280.00'), (1.25, '1.25'),
    (0.5611222, '0.56'), (0.0, '0.00'), (-8.0, '-8.00'),
    (0.005, '0.01'),            # the boundary rounds, it does not escape
    (0.0001, '1.00e-04'),       # Kmin's floor, which .2f would show as 0.00
    (0.001, '1.00e-03'),        # TAU_MIN, likewise
])
def test_fmt2_always_shows_two_digits_after_the_point(value, expected):
    """Every number the GUI prints goes through here, so the contract is two
    decimals — and, for magnitudes that would round away entirely, two decimals
    in the exponential form rather than a false zero."""
    assert fmt2(value) == expected


def test_fmt2_output_parses_back_to_what_it_rendered():
    """The header's Tsim/Ts fields are read back from what fmt2 wrote into them,
    so the rendering has to survive a float() round trip."""
    for value in (280.0, 0.5611, 2e-4, 1e-6):
        assert float(fmt2(value)) == pytest.approx(value, rel=0.01)


def test_gain_slider_marks_span_the_box():
    marks = gain_slider_marks(*GAIN_BOX)
    lo, hi = math.log10(GAIN_BOX[0]), math.log10(GAIN_BOX[1])
    assert marks[lo] == fmt2(GAIN_BOX[0]) and marks[hi] == fmt2(GAIN_BOX[1])
    assert all(lo <= float(k) <= hi for k in marks)


def test_gain_slider_marks_on_a_partial_decade():
    marks = gain_slider_marks(0.03, 3.0)
    assert marks[math.log10(0.03)] == '0.03' and marks[math.log10(3.0)] == '3.00'
    assert marks[-1] == '0.10' and marks[0] == '1.00'


def test_shared_defaults_are_self_consistent():
    assert len(NBAR) == 3 and all(n > 0 for n in NBAR)
    assert GAIN_BOX[0] < GAIN_BOX[1]
    assert set(N_ITER_BY_CTYPE) == {'I', 'PI', 'PID'}


def test_layout_defaults_come_from_params():
    """C5: the GUI must not restate the defaults it renders."""
    import layout

    assert layout.GAIN_LOG_MIN == math.log10(GAIN_BOX[0])
    assert layout.GAIN_LOG_MAX == math.log10(GAIN_BOX[1])
    assert layout.SLIDER_MARKS == gain_slider_marks(*GAIN_BOX)
