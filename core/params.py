"""
Shared constants and plant-parameter parsing.

This module is the single source of truth for the values that both the
simulation core and the GUI need to agree on, and it deliberately imports
nothing from the rest of ``core`` so anything may import it without creating a
cycle (``signals`` imports ``pid``, so the analysis-window constants cannot
live in ``signals`` if ``pid`` is to see them too).
"""

from __future__ import annotations

import ast

import numpy as np

# ── Analysis window ───────────────────────────────────────────────────────────
# Samples trimmed from each end before the encirclement counts are taken.
K1_PADDING = 5
K2_PADDING = 5

# ── Simulation grid ───────────────────────────────────────────────────────────
# Samples the auto grid proposes for every plant, and the horizon it proposes
# them over. See core.signals.auto_grid for why these two numbers, together,
# are the whole grid decision.
N_POINTS = 500
HORIZON_SPANS = 10.0      # Tsim, in multiples of the plant's span = sum(tau) + L

# Bounds on a hand-edited grid. N is only exactly N_POINTS while the proposal
# stands; an override moves it, and these keep it survivable.
#
# N_MAX is set by conditioning, not by speed. plant_tf encodes the dead time as
# nd = round(L/Ts) trailing denominator zeros, so a finer Ts means a longer
# polynomial, and lfilter on a high-degree denominator near the unit circle
# eventually loses the response entirely. Measured on P2 (four 5s lags, L=8,
# Tsim=280): clean and converging through N=35000 (nd=1000), peak 1.57; visibly
# wrong at N=40000 (nd=1143), peak 2.68; diverged at N=50000 (nd=1429), peak
# 119. 20000 is half the first bad value, and P3 -- the delay-dominant plant,
# which reaches a given nd at the lowest N -- is still clean there.
N_MIN, N_MAX = 20, 20000


def time_grid(Tsim: float, Ts: float) -> np.ndarray:
    """
    Sample instants [0, Ts, 2*Ts, ..., Tsim].

    Every simulation builds its time base here, so they cannot disagree about
    the sample count. They used to: the TF paths built ``arange(0, Tsim +
    Ts/2, Ts)`` while the anti-windup loop counted ``1 + ceil(Tsim/Ts)``, and
    the two differ by one whenever Tsim/Ts lands on a half sample. Rounding
    once, here, is also what makes an exact N reachable at all -- the proposed
    grid sets Ts = Tsim/(N-1), so the ratio is N-1 give or take float noise,
    which ``ceil`` would round the wrong way about half the time.
    """
    return np.arange(int(round(Tsim / Ts)) + 1) * Ts


# ── Plant parameters ──────────────────────────────────────────────────────────
# A time constant of zero divides by zero in plant_tf's exp(-Ts/tau); a negative
# one puts a pole outside the unit circle, i.e. a "plant" that diverges by
# construction. Both are clamped away rather than allowed to reach the solver.
TAU_MIN = 1e-3

# ── Paper constants (RoboPID_JPC_paper/main.tex) ──────────────────────────────
NBAR = (0.5, 0.75, 1.0)   # loop limits N̄0, N̄1, N̄2
EPS = 0.1                 # truncation disc radius (Definition 1)
DELTA = 0.02              # settling-band guard (Definition 4)
BETA = 0.1                # tuning step size; the applied notch is 1/(1-beta)

# Derivative roll-off divisor: D(s) = Kd*s / (1 + (Kd/DERIV_FILTER_N)*s), the
# "implemented derivative filter" of main.tex's Remark on the mod-2pi
# formulation (~line 411). Kd is in seconds in this parallel form, so Kd/N is
# already a time constant and no Kp is needed -- which matters because the
# tuner can drive Kp to its 0.01 floor while Kd stays high, and the textbook
# Td/N would filter the derivative channel out of existence there.
#
# Not optional. An unfiltered discrete derivative has Nyquist gain 2*Kd/Ts,
# unbounded as Ts shrinks, and for derivative-on-output the inner loop
# P/(1 + D*P) then has characteristic polynomial tending to (z-1)(z + Kd*K/tau)
# -- a pole at exactly -Kd*K/tau. On tau=K=1 the default Kd=1 sits on the unit
# circle (undamped Nyquist ringing) and Kd=1.1 diverges. The filter caps the
# derivative's high-frequency gain at N regardless of the gains, which removes
# that pole rather than merely damping it.
DERIV_FILTER_N = 10.0

# ── GUI defaults ──────────────────────────────────────────────────────────────
GAIN_BOX = (0.01, 10.0)   # (Kmin, Kmax): gain boundary and slider range

# Tuning iteration budget per controller structure — fewer terms to search
# converge faster, so I/PI don't need as many iterations as full PID.
N_ITER_BY_CTYPE = {'I': 50, 'PI': 100, 'PID': 200}

DEFAULT_TAU = (5.0,)


def parse_tau(tau_str, default=DEFAULT_TAU) -> tuple[np.ndarray, list[str]]:
    """
    Parse the tau field into a usable array of time constants.

    Returns ``(tau, notes)``. ``notes`` holds user-facing messages for anything
    that had to be corrected, so the caller can surface the correction instead
    of silently simulating something the user did not ask for.
    """
    fallback = np.asarray(default, dtype=float)
    try:
        arr = np.atleast_1d(
            np.asarray(ast.literal_eval(str(tau_str).strip()), dtype=float))
    except Exception:
        return fallback, ["τ: couldn't parse — using [5.0]"]

    notes: list[str] = []
    finite = arr[np.isfinite(arr)]
    if finite.size < arr.size:
        notes.append('τ: non-finite entries dropped')
    if finite.size == 0:
        return fallback, notes + ['τ: no usable entries — using [5.0]']
    if np.any(finite < TAU_MIN):
        notes.append(f'τ: entries below {TAU_MIN:g} clamped')
        finite = np.maximum(finite, TAU_MIN)
    return finite, notes


def gain_slider_marks(kmin: float, kmax: float) -> dict:
    """
    Log-scale slider marks spanning [kmin, kmax]: one per whole decade inside
    the range, plus the exact endpoints.
    """
    lo, hi = np.log10(kmin), np.log10(kmax)
    start, end = int(np.ceil(lo - 1e-9)), int(np.floor(hi + 1e-9))
    marks = {i: f'{10.0 ** i:g}' for i in range(start, end + 1)}
    marks[lo] = f'{kmin:g}'
    marks[hi] = f'{kmax:g}'
    return marks
