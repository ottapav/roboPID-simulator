"""Plant-parameter parsing and the shared defaults."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.params import (
    DEFAULT_TAU, GAIN_BOX, N_ITER_BY_CTYPE, NBAR, TAU_MIN,
    gain_slider_marks, parse_tau,
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


def test_gain_slider_marks_span_the_box():
    marks = gain_slider_marks(*GAIN_BOX)
    lo, hi = math.log10(GAIN_BOX[0]), math.log10(GAIN_BOX[1])
    assert marks[lo] == f'{GAIN_BOX[0]:g}' and marks[hi] == f'{GAIN_BOX[1]:g}'
    assert all(lo <= float(k) <= hi for k in marks)


def test_gain_slider_marks_on_a_partial_decade():
    marks = gain_slider_marks(0.03, 3.0)
    assert marks[math.log10(0.03)] == '0.03' and marks[math.log10(3.0)] == '3'
    assert marks[-1] == '0.1' and marks[0] == '1'


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
