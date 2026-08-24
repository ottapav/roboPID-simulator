"""
Admissibility gates: is this plant one the method can tune, and can the tuner
get there from where the sliders are?

The tuner is a search for a stability boundary. It raises gains until a band
starts to ring, reads which band rang, and backs that band off. Two things can
make that search meaningless, and they are not the same thing:

  * **There is no boundary.** If the plant never supplies enough phase lag, no
    gain anywhere makes the loop ring; every turn index stays at zero, the
    triangular rule reads "all quiet" forever, and the multipliers walk to the
    box ceiling having found nothing. Nothing about the box or the iteration
    budget helps -- the answer does not exist. That is Gate 1, and it is the
    only condition here that rejects a run.

    Gate 1 is a structural test on relative degree, and a deliberately
    conservative one: it refuses the marginal band, where the asymptotic loop
    phase lands exactly on -pi, rather than resting a verdict on the O(1/w)
    terms that decide it there. So a rejected plant may still be tunable. The
    gate states the limit of what this method claims, which is not the same as
    a claim about the plant. See check_phase.
  * **The boundary is outside the box.** The answer exists but the search cannot
    reach it from the current sliders. The run is still valid; it just
    terminates at a bound instead of at a turn-index limit. Those are Gates 2-4,
    which warn and let the run proceed.

Gate 0 (positive static gain) precedes both: it is required in its own right,
and it is required *before* the phase sweep, since a negative K adds a further
+-pi to arg P and would let a one-lag plant pass Gate 1 spuriously.

Not re-checked here, because they are guaranteed upstream: tau >= TAU_MIN
(core.params.parse_tau clamps it) and L >= 0 (callbacks._deadtime clamps it).
The plant form K*exp(-Ls)/prod(tau_i*s + 1) cannot express an integrating,
open-loop-unstable, non-minimum-phase or resonant process at all, so those
members of Assumption 1 need no test -- the parameterization enforces them.

Two runtime detectors live here as well (`diagnose_run`). They are not
preconditions: they read the state the tuner actually finished in and say
whether the box or the plant was the binding constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import DERIV_FILTER_N

# ── Controller structure ──────────────────────────────────────────────────────
# phi_infinity: the phase the controller contributes in the limit. Gate 1 asks
# that the *asymptotic* loop phase sit strictly below -pi,
#
#     -(pi/2)*r + phi_inf < -pi     <=>     r > 2 + (2/pi)*phi_inf
#
# for plant relative degree r, which is what LAGS_REQUIRED tabulates. The two
# are kept side by side deliberately: the requirement is derived, not chosen,
# and DEGREE_REQUIRED below re-derives it so the table cannot drift from the
# formula it came from.
#
# The app only ever uses the *filtered* PID row: the derivative roll-off in
# core.params.DERIV_FILTER_N is mandatory and unconditional, and it eats the
# +pi/2 an ideal derivative would contribute, leaving 0 from above.
# PID_UNFILTERED is here for completeness -- it is not reachable from the GUI.
#
# PI and filtered PID are 0 from opposite sides, which the strict inequality
# would resolve differently; both land on r >= 3 and the sign is carried in the
# comment rather than in a float that cannot represent it.
PHI_BY_CTYPE = {
    'I': -np.pi / 2,
    'PI': 0.0,             # 0 from below
    'PID': 0.0,            # 0 from above
    'PID_UNFILTERED': np.pi / 2,
}


def _degree_required(phi_inf: float) -> int:
    """Smallest integer relative degree with -(pi/2)*r + phi_inf < -pi."""
    r = 2.0 + (2.0 / np.pi) * phi_inf
    # Strictly greater: an integer sitting exactly on the bound is the marginal
    # case Gate 1 refuses, so it must not satisfy the test.
    return int(np.floor(r) + 1)


# {'I': 2, 'PI': 3, 'PID': 3, 'PID_UNFILTERED': 4}; pinned in the tests.
LAGS_REQUIRED = {c: _degree_required(p) for c, p in PHI_BY_CTYPE.items()}

# How each structure's phi_infinity reads in a sentence, for the Gate 1 message.
PHI_PHRASE = {
    'I': 'the integrator adds a further −90°',
    'PI': 'the controller adds nothing in the limit',
    'PID': 'the mandatory derivative roll-off leaves the controller at 0° there',
    'PID_UNFILTERED': 'an unfiltered derivative adds +90°',
}

# Which gains a structure actually searches, and which Pachner plot indicts
# each one (core.tuning.triangular_rule: feats[0]=Gamma0 -> Ki, feats[1]=Gamma1
# -> Kp, feats[2]=Gamma2 -> Kd).
ACTIVE_GAINS = {
    'I': ('Ki',),
    'PI': ('Kp', 'Ki'),
    'PID': ('Kp', 'Ki', 'Kd'),
    'PID_UNFILTERED': ('Kp', 'Ki', 'Kd'),
}

BAND_OF_GAIN = {'Ki': 0, 'Kp': 1, 'Kd': 2}

# ── Target rule ───────────────────────────────────────────────────────────────
# The target the gates measure reachability against comes from the plant's own
# ultimate point -- the same phase sweep Gate 1 runs -- rather than from a FOPTD
# reduction, so it inherits Gate 1's indifference to whether the plant is a
# chain of lags or something else with the same frequency response.
#
# Kp = Ku/3.5 is the measured ratio between the ultimate gain and an AMIGO
# design across four decades of L/tau. Ti and Td are Ziegler-Nichols off Tu.
KU_OVER_KP = 3.5
TI_OVER_TU = 0.5
TD_OVER_TU = 0.125

# Band non-degeneracy floors. Ti/Td is fixed at TI_OVER_TU/TD_OVER_TU = 4 by the
# rule above, and nu ships at 10, so *neither* of these can fire as the
# constants stand -- they are regression guards on the constants themselves, and
# their messages say so. TI_TD_MIN sits below 4 deliberately: the gate is meant
# to mean "Ti has collapsed onto Td", not to track the rule's own arithmetic.
TI_TD_MIN = 2.0
NU_MIN = 4.0

# A multiplier reaches a bound through max()/min() in triangular_rule, so it
# lands on it exactly; the tolerance is only against float round-trips.
PIN_RTOL = 1e-9


# ── Findings ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Finding:
    """One thing wrong, in a form both display surfaces can render.

    Splitting the text into title/detail/fixes is what lets the modal and the
    warning line show the same finding at different densities without the
    wording being duplicated in two places and drifting apart.
    """
    gate: str                                   # 'sign' | 'phase' | 'reach' | ...
    blocking: bool                              # True: the answer does not exist
    title: str                                  # one bolded lead line
    detail: tuple[str, ...] = ()                # paragraphs
    fixes: tuple[str, ...] = ()                 # rendered as a list
    status: str = ''                            # terse suffix for the status line


@dataclass(frozen=True)
class Verdict:
    findings: tuple[Finding, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(f.blocking for f in self.findings)

    @property
    def blocking(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.blocking)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if not f.blocking)


@dataclass(frozen=True)
class Target:
    """The gain the tuner is implicitly searching for, and where it came from."""
    wu: float
    Ku: float
    Tu: float
    Kp: float
    Ki: float
    Kd: float
    Ti: float
    Td: float

    def gain(self, name: str) -> float:
        return getattr(self, name)

    @property
    def provenance(self) -> str:
        return (f'The target comes from the plant’s ultimate point '
                f'ω_u = {self.wu:.4g} rad/s, Ku = {self.Ku:.4g}, '
                f'Tu = {self.Tu:.4g} s via Kp = Ku/{KU_OVER_KP:g}, '
                f'Ti = Tu/{1 / TI_OVER_TU:g}, Td = Tu/{1 / TD_OVER_TU:g}.')


# ── Frequency response ────────────────────────────────────────────────────────

def plant_phase(tau, L: float, w: np.ndarray) -> np.ndarray:
    """
    arg P(jw) in radians for P(s) = K*exp(-Ls)/prod(tau_i*s + 1), K > 0.

    Monotone decreasing in w, so nothing here needs unwrapping and the minimum
    over any grid is its last sample. K is absent because a positive gain
    contributes no phase -- which is exactly why Gate 0 has to run first.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    return -L * w - np.sum(np.arctan(np.outer(w, tau)), axis=1)


def plant_magnitude(tau, K: float, w: np.ndarray) -> np.ndarray:
    """|P(jw)|. The dead time is all-pass and does not appear."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    return abs(K) / np.prod(np.sqrt(1.0 + np.outer(w, tau) ** 2), axis=1)


def phase_sweep(tau, L: float, points_per_decade: int = 128
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    (w, arg P(jw)) over a grid wide enough to settle the question.

    The top of the range has to cover both corner sources. 1e3/min(tau) puts
    every lag within 0.057 deg of its own 90 deg asymptote, which is far below
    any decision margin here. The dead time needs its own term: a *small* L
    still guarantees a crossing mathematically (the lag -L*w is unbounded) but
    places it near pi/L, so a grid topping out at 1e3/min(tau) would miss it and
    report a plant with L = 1e-9 as phase-starved. Dividing by min(tau_min, L)
    covers whichever of the two is the later one to run out.

    Resolution is fixed per decade rather than in total, so widening the range
    for a tiny L does not thin out the samples near the crossing.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    span = float(np.sum(tau)) + L
    w_lo = 1e-3 / span
    w_hi = 1e3 / (min(float(np.min(tau)), L) if L > 0 else float(np.min(tau)))

    lo, hi = np.log10(w_lo), np.log10(w_hi)
    n = max(2000, int(points_per_decade * (hi - lo)))
    w = np.logspace(lo, hi, n)
    return w, plant_phase(tau, L, w)


def ultimate_point(tau, K: float, L: float, phi: float) -> tuple[float, float, float] | None:
    """
    (w_u, Ku, Tu) at the frequency where arg P crosses -pi - phi, or None when
    it never does (i.e. Gate 1 fails, and there is no ultimate point to find).

    The crossing is interpolated linearly in log w between the bracketing
    samples; the phase is smooth and monotone there, so this is exact to well
    past the precision anything downstream needs.
    """
    thr = -np.pi - phi
    w, ph = phase_sweep(tau, L)

    below = np.nonzero(ph < thr)[0]
    if below.size == 0:
        return None

    i = int(below[0])
    if i == 0:
        w_u = float(w[0])
    else:
        t = (thr - ph[i - 1]) / (ph[i] - ph[i - 1])
        w_u = float(10.0 ** (np.log10(w[i - 1])
                             + t * (np.log10(w[i]) - np.log10(w[i - 1]))))

    mag = float(plant_magnitude(tau, K, np.array([w_u]))[0])
    if not np.isfinite(mag) or mag <= 0.0:
        return None
    return w_u, 1.0 / mag, 2.0 * np.pi / w_u


def zn_target(w_u: float, Ku: float, Tu: float) -> Target:
    """Kp = Ku/3.5 with Ziegler-Nichols Ti, Td off Tu."""
    Kp = Ku / KU_OVER_KP
    Ti = TI_OVER_TU * Tu
    Td = TD_OVER_TU * Tu
    return Target(wu=w_u, Ku=Ku, Tu=Tu,
                  Kp=Kp, Ki=Kp / Ti, Kd=Kp * Td, Ti=Ti, Td=Td)


# ── Gates ─────────────────────────────────────────────────────────────────────

def _deg(rad: float, places: int = 1) -> str:
    """An angle in degrees, using the same minus sign as the prose around it."""
    return f'{np.degrees(rad):.{places}f}°'.replace('-', '−')


def check_sign(K: float, K_raw=None, gain_box=(0.01, 10.0)) -> Finding | None:
    """Gate 0: the static gain must be a positive, finite number."""
    Kmin, Kmax = gain_box

    if not np.isfinite(K):
        return Finding(
            gate='sign', blocking=True,
            title='Static gain K could not be read as a number.',
            detail=(
                f'The K field contains {K_raw!r}, which is not a number. Left '
                f'alone, the tuner falls back to K = 1.0 and tunes against a '
                f'plant you did not specify — so it is refused rather than '
                f'guessed at.',
            ),
            fixes=('Enter a positive number — for example 1.25, the default '
                   'plant’s gain.',),
            status='⚠ K unreadable',
        )

    if K <= 0.0:
        return Finding(
            gate='sign', blocking=True,
            title=f'Static gain K = {K:g} is not positive.',
            detail=(
                'The method assumes a self-regulating, positive-acting plant '
                '(Assumption 1), and the search box admits only positive Kp, '
                'Ki and Kd.',
                f'With K < 0 the loop closes as *positive* feedback, so no gain '
                f'anywhere in [{Kmin:g}, {Kmax:g}] stabilises it. With K = 0 the '
                f'plant does not respond at all and there is no error record to '
                f'score.',
            ),
            fixes=('Enter a positive K. An inverse-acting process is modelled '
                   'by flipping the sign of the measurement, not the sign of K.',),
            status='⚠ K must be positive',
        )

    return None


def relative_degree(tau) -> int:
    """
    Relative degree of P(s) = K*exp(-Ls)/prod(tau_i*s + 1): one per lag.

    This plant form has no zeros, so the count is exact. The wider statement --
    for a model class that did have them -- is to use the plain relative degree
    and take no credit for right-half-plane zeros: they add lag, so a plant that
    passes Gate 1 on r passes a fortiori once they are included. A note rather
    than code, since the GUI cannot express such a plant.
    """
    return int(np.atleast_1d(np.asarray(tau, dtype=float)).size)


def check_phase(tau, L: float, ctype: str) -> Finding | None:
    """
    Gate 1: the plant must supply enough phase lag for a stability boundary to
    exist under this controller.

    Structural, not measured. The condition is that the *asymptotic* loop phase
    sit strictly below -pi:

        -(pi/2)*r + phi_inf < -pi      <=>      r > 2 + (2/pi)*phi_inf

    which is LAGS_REQUIRED. L > 0 passes unconditionally -- a dead time makes
    the phase unbounded below, so the asymptote question does not arise.

    This deliberately replaces an earlier implementation that swept arg P(jw)
    and asked whether it dipped below -pi - phi. Two reasons, and both matter:

      * The sweep needs a tolerance exactly where the answer is marginal. The
        cases that fail are the ones whose asymptote lands *on* the threshold,
        so the verdict came down to how close to its asymptote the grid got --
        a sign-of-a-small-number decision with a step size in it. Counting is
        exact and step-size independent.
      * It must never be run on a discretized model, where the ZOH's own lag
        would manufacture a crossing that the continuous plant does not have.
        A structural count cannot be fooled that way.

    The price is conservatism, and it falls entirely in the marginal band -- PI
    at r=2, filtered PID at r=2, unfiltered PID at r=3. There the asymptote is
    exactly -pi and whether the loop truly crosses is settled by O(1/w) terms
    in Ti, Td, nu and the plant's own time constants. Such a plant may well
    have a boundary: tau=[1,2] under PID does, and the tuner converges on it.
    The method does not evaluate those terms, so it declines the case rather
    than guessing -- and the message says exactly that, because "SPIN does not
    claim this plant" and "this plant cannot be controlled" are different
    statements and only the first one is true.
    """
    if L > 0:
        return None

    r = relative_degree(tau)
    r_req = LAGS_REQUIRED[ctype]
    if r >= r_req:
        return None

    phi = PHI_BY_CTYPE[ctype]
    asym = -np.pi / 2 * r + phi
    marginal = r == r_req - 1
    return Finding(
        gate='phase', blocking=True,
        title=f'{ctype} control needs more phase lag than this plant supplies.',
        detail=(
            f'{ctype} needs a plant of relative degree {r_req} or more; this '
            f'one has r = {r} — {r} lag{"" if r == 1 else "s"}, no dead time, '
            f'no zeros — which gives −90° × {r} = {_deg(-np.pi / 2 * r, 0)}. '
            f'On top of that {PHI_PHRASE[ctype]}, so the loop phase tends to '
            f'{_deg(asym, 0)} as ω → ∞, and the requirement is that it sit '
            f'strictly below −180°.',
            'Above −180° there is no stability boundary for the search to find: '
            'no gain in any band makes the response ring, every turn index '
            'stays at zero, the triangular rule reads “all quiet” forever, and '
            'the multipliers walk to the box ceiling having found nothing.',
            ('This is a strict test on the asymptote, and deliberately '
             'conservative: your plant sits in the band where it lands exactly '
             'on −180°, and whether the loop actually crosses is then decided '
             'by O(1/ω) terms in Ti, Td, the roll-off ν and the plant’s time '
             'constants. The method does not evaluate those, so it declines '
             'the case rather than resting a verdict on the sign of a small '
             'number. Such a plant may well be tunable — by this tuner or '
             'another — but SPIN does not claim it.'
             if marginal else
             'Widening the gain box will not help, and neither will more '
             'iterations: there is no target to reach.'),
        ),
        fixes=(
            'Add dead time. Any L > 0 satisfies every controller, since a dead '
            'time makes the phase unbounded below.',
            f'Add lags. {ctype} needs relative degree ≥ {r_req}; this plant '
            f'has {r}.',
            'Choose a controller with a lower requirement: I needs 2, PI needs '
            '3, PID needs 3 (the derivative roll-off is mandatory here, which '
            'is why PID needs 3 rather than the unfiltered 4).',
        ),
        status='⚠ Not tunable — see message',
    )


def check_reachability(target: Target, ctype: str, start: dict,
                       gain_box=(0.01, 10.0)) -> list[Finding]:
    """
    Gate 2: the target must sit inside the multiplier box the tuner searches.

    The bounds are the ones callbacks.run_tune hands to pid_tuning --
    (Kmin/g_start, Kmax/g_start) -- so the test is equivalently
    Kmin <= g_target <= Kmax. Warning only: a run that ends pinned at a bound
    is still a valid run, it has just answered "the boundary is further out
    than the box goes".
    """
    Kmin, Kmax = gain_box
    findings: list[Finding] = []

    for name in ACTIVE_GAINS[ctype]:
        g_start = float(start.get(name, 0.0))
        g_target = target.gain(name)
        if g_start <= 0.0 or not np.isfinite(g_target) or g_target <= 0.0:
            continue

        F_min, F_max = Kmin / g_start, Kmax / g_start
        ratio = g_target / g_start
        if F_min <= ratio <= F_max:
            continue

        low = ratio < F_min
        findings.append(Finding(
            gate='reach', blocking=False,
            title=f'{name} target is out of reach from the current slider.',
            detail=(
                f'The tuner would need {name} ≈ {g_target:.4g}, which is '
                f'{ratio:.3g}× the starting {name} = {g_start:.4g}, but the '
                f'multiplier box only spans {F_min:.3g}×…{F_max:.3g}× '
                f'(gain box [{Kmin:g}, {Kmax:g}]).',
                target.provenance,
                f'The run is still valid — it will simply terminate with {name} '
                f'pinned at its {"lower" if low else "upper"} bound instead of '
                f'at a turn-index limit.',
                f'Moving the {name} slider will not help: the multiplier '
                f'bounds are derived from the gain box, so the reachable set is '
                f'[{Kmin:g}, {Kmax:g}] wherever the search starts.',
            ),
            fixes=(
                (f'Widen the gain box to Kmin ≤ {g_target:.3g}.' if low
                 else f'Widen the gain box to Kmax ≥ {g_target:.3g}.'),
            ),
            status=f'⚠ {name} target outside the box',
        ))

    return findings


def check_bands(target: Target, ctype: str) -> list[Finding]:
    """
    Gates 3 and 4: the two bands the derivative-side turn indices read must not
    have collapsed.

    Both are structural under the shipped constants -- Ti/Td is fixed at 4 by
    TI_OVER_TU/TD_OVER_TU and nu is fixed at 10 -- so neither can fire unless
    somebody edits those. That is the point of them, and the messages say so.
    """
    if ctype not in ('PID', 'PID_UNFILTERED'):
        return []

    findings: list[Finding] = []

    ratio = target.Ti / target.Td if target.Td > 0 else np.inf
    if ratio < TI_TD_MIN:
        findings.append(Finding(
            gate='bands', blocking=False,
            title='Band 1 has collapsed.',
            detail=(
                f'The target Ti = {target.Ti:.4g} s and Td = {target.Td:.4g} s '
                f'differ by only {ratio:.3g}× (minimum {TI_TD_MIN:g}×).',
                'Band 1 = [1/Ti, 1/Td] is the frequency range Γ1’s turn index '
                'N1 is selective over. When Ti ≈ Td that range vanishes and N1 '
                'no longer attributes ringing to Kp specifically, so the middle '
                'row of the triangular rule stops meaning what it says.',
                'With the shipped constants Ti/Td is fixed at '
                f'{TI_OVER_TU / TD_OVER_TU:g}, so this can only appear if '
                'TI_OVER_TU or TD_OVER_TU in core/admissibility.py has been '
                'changed — check those before suspecting the plant.',
            ),
            status='⚠ band 1 degenerate',
        ))

    if DERIV_FILTER_N < NU_MIN:
        findings.append(Finding(
            gate='filter', blocking=False,
            title='Band 2 has collapsed.',
            detail=(
                f'The derivative roll-off divisor ν = {DERIV_FILTER_N:g} leaves '
                f'band 2 = [1/Td, ν/Td] spanning only {DERIV_FILTER_N:g}× '
                f'(minimum {NU_MIN:g}×).',
                'That band is what Γ2’s turn index N2 reads to attribute '
                'buzzing to Kd; once it narrows toward band 1, N2 stops being '
                'selective and the rule’s bottom row misfires.',
                'ν = DERIV_FILTER_N ships at 10 — one full decade — so this can '
                'only appear if core/params.py has been changed.',
            ),
            status='⚠ band 2 degenerate',
        ))

    return findings


def check_plant(tau, K: float, L: float, ctype: str, start: dict,
                gain_box=(0.01, 10.0), K_raw=None) -> Verdict:
    """
    Every precondition, in the order they depend on each other.

    `start` maps 'Kp'/'Ki'/'Kd' to the absolute gains the sliders are sitting
    on; `gain_box` is (Kmin, Kmax). Stops at the first blocking gate, because
    neither the phase sweep nor the ultimate point means anything once an
    earlier one has failed.
    """
    sign = check_sign(K, K_raw=K_raw, gain_box=gain_box)
    if sign is not None:
        return Verdict((sign,))

    phase = check_phase(tau, L, ctype)
    if phase is not None:
        return Verdict((phase,))

    # Passing Gate 1 puts the asymptote strictly below the threshold, and arg P
    # approaches its asymptote from above, so the crossing the sweep looks for
    # always exists here -- this branch stays unreachable. It is kept because
    # "no ultimate point" is a real answer for the function itself, and the two
    # should not silently disagree if the gate is ever relaxed.
    point = ultimate_point(tau, K, L, PHI_BY_CTYPE[ctype])
    if point is None:
        return Verdict()

    target = zn_target(*point)
    return Verdict(tuple(check_reachability(target, ctype, start, gain_box)
                         + check_bands(target, ctype)))


# ── Runtime detectors ─────────────────────────────────────────────────────────

def _pinned(F: float, bound: float) -> bool:
    return bool(np.isclose(F, bound, rtol=PIN_RTOL, atol=0.0))


def diagnose_run(ctype: str, mult: dict, limits: dict, gains: dict,
                 feats: list[dict], gain_box=(0.01, 10.0)) -> list[Finding]:
    """
    What the finished run actually says, read off the final multipliers and the
    counts of the last iteration scored.

    `mult`/`limits`/`gains` map a gain name to its final multiplier, its
    (F_min, F_max) pair and its final absolute gain. `feats` is the three-element
    list from the last on_iteration call.

    The two detectors are mutually exclusive by construction: A requires every
    count under its limit, B requires one over.
    """
    Kmin, Kmax = gain_box
    active = ACTIVE_GAINS[ctype]
    if not active or len(feats) < 3:
        return []

    quiet = all(feats[k]['N'] <= feats[k]['Nbar'] for k in range(3))
    counts = ', '.join(
        f'N{k} = {feats[k]["N"]:.3g} ≤ {feats[k]["Nbar"]:g}' for k in range(3))

    # A -- every band quiet with every multiplier hard against the ceiling: the
    # search ran out of box before anything rang.
    #
    # This used to read "widening the box will not help", on the grounds that
    # it was the runtime signature of a phase-starved plant. Under the
    # structural Gate 1 that is no longer true of anything that can reach here:
    # passing the gate puts the asymptotic loop phase strictly below -pi, so a
    # boundary does exist and the only place left for it is above Kmax. A and B
    # now differ by *why* rather than by remedy -- A has no band to attribute
    # the shortfall to, B has one and no room left.
    if quiet and all(_pinned(mult[n], limits[n][1]) for n in active):
        return [Finding(
            gate='ceiling', blocking=False,
            title='Finished at the box ceiling with nothing ringing.',
            detail=(
                f'Every tuned multiplier ended pinned at its ceiling '
                f'({", ".join(f"{n} ×{limits[n][1]:.3g}" for n in active)}) '
                f'while all three turn indices stayed under their limits: '
                f'{counts}.',
                'Nothing rang, so no band can be held responsible and the rule '
                'had nothing to back off — it simply raised everything until '
                'the box stopped it. This plant passed the phase gate, which '
                'means a stability boundary does exist; it is above the current '
                'ceiling rather than absent.',
                'The gains now on the sliders are the box ceiling, not a tuned '
                'result.',
            ),
            fixes=(f'Raise Kmax above {Kmax:g} and re-run — for example to '
                   f'{Kmax * 10.0:g} — so the search can reach the boundary.',),
            status='— box ceiling, nothing rang',
        )]

    # B -- a multiplier stuck on a bound while its own band is still ringing:
    # the rule knew what to do and ran out of room to do it.
    findings: list[Finding] = []
    for name in active:
        F = mult[name]
        F_min, F_max = limits[name]
        k = BAND_OF_GAIN[name]
        N, Nbar = feats[k]['N'], feats[k]['Nbar']
        if N <= Nbar:
            continue

        at_min, at_max = _pinned(F, F_min), _pinned(F, F_max)
        if not (at_min or at_max):
            continue

        g = gains[name]
        findings.append(Finding(
            gate='bound', blocking=False,
            title=(f'{name} finished pinned at its '
                   f'{"lower" if at_min else "upper"} bound and its band is '
                   f'still ringing.'),
            detail=(
                f'{name} ended at ×{F:.3g} (gain {g:.4g}, the box '
                f'{f"floor Kmin = {Kmin:g}" if at_min else f"ceiling Kmax = {Kmax:g}"}) '
                f'while N{k} = {N:.3g} still exceeds its limit '
                f'N̄{k} = {Nbar:g} — the rule wanted to keep '
                f'{"cutting" if at_min else "raising"} {name} and ran out of '
                f'box.',
                'The gains now on the sliders are a box boundary, not a '
                'converged tuning.',
            ),
            fixes=((f'Widen the gain box — the paper’s first remedy — to '
                    f'Kmin ≤ {Kmin / 10.0:g}, then re-run.' if at_min else
                    f'Widen the gain box — the paper’s first remedy — to '
                    f'Kmax ≥ {Kmax * 10.0:g}, then re-run.'),),
            status=f'— {name} pinned at a bound',
        ))

    return findings
