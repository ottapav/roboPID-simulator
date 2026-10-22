"""
Iterative PID gain tuning via the triangular rule of
docs/JPC26_basic/main.tex (Table "the tuning rule at a glance", eq.
"triangular"):
- Fp, Fi, Fd are multipliers applied on top of the base gains
- Each iteration evaluates the stability screen and the three Gamma
  encirclement counts and nudges gains by one gamma = 1/(1-beta) notch
- An unstable record cuts every gain at once, far more coarsely than the
  gamma notch the count-based rows use, and deepening with the band's
  frequency (Ki /2, Kp /4, Kd /8); otherwise the lowest violated band is
  cut and every band below it is raised; if none is violated, all bands
  are raised

Three variants (docs/ECC27_certified_SPIN), chosen by pid_tuning(variant=...):
- 'basic':     SPIN as published -- the counts alone decide.
- 'certified': a tuning the counts accept is accepted only if the certified
               upper bound on M_s from its own record (core.certificate,
               Theorem 1) is at most M*; otherwise the gains are lowered by the
               band move predicted to lower M_s (at most N_BAND_MAX in a row),
               else by a uniform back-off. The result ends with M_s <= M*, but
               fragile tunings are still applied, and only then rejected.
- 'safe':      every move is certified *before* it is applied (Theorem 3):
               from an accepted tuning, the move of the rule is applied only if
               the record certifies that the new tuning stabilizes the plant
               with M_s <= M*; otherwise the band moves (highest band first) and
               a uniform back-off are tried, else the tuning is held. Before the
               first accepted tuning it backs off uniformly.
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from .features import (
    loop_response_features, FeatureDescription, DELTA, EPSILON,
    encirc, EncirclementMetric,
)
from .signals import find_index, StabilityScreen
from .certificate import (certificate, certify_ahead, predict_Ms, check_record,
                          tau_for, TAU_CERT, MSTAR)


# Table 2's rightmost column, keyed by which row fired this iteration.
MANUAL_READING = {
    'unstable': 'runaway: back off hard',
    'N0': 'slow cycling: less reset',
    'N1': 'ringing: less gain',
    'N2': 'buzzing: less rate',
    'none': 'all quiet: tighten',
    # certified / safe SPIN (docs/ECC27_certified_SPIN)
    'cert_reject': 'certificate: M_s above target, lower gains',
    'warm': 'not yet certified: back off',
    'fallback': 'not confirmed: back to last certified tuning',
    'hold': 'no move certified: hold',
    'cert_band': 'certified band move',
    'cert_back': 'certified back-off',
}

CERT_ROWS = frozenset({'cert_reject', 'warm', 'fallback', 'hold', 'cert_band', 'cert_back'})
VARIANTS = ('basic', 'certified', 'safe')
N_BAND_MAX = 10      # certified SPIN: band moves in a row before a uniform back-off


# Swappable gain-update decisions: given the current features/gains, decide
# the next multipliers and which MANUAL_READING row fired. Register a new
# key here to make an alternative rule selectable via
# pid_tuning(..., rule=<fn>) without editing the iteration loop below.
TuningRule = Callable[
    [list[dict], tuple[float, float, float], bool,
     float, float, float, float,
     float, float, float, float, float, float],
    tuple[float, float, float, str]
]


def triangular_rule(
    feats: list[dict],
    Nbar: tuple[float, float, float],
    unstable: bool,
    Fp_cur: float, Fi_cur: float, Fd_cur: float,
    gamma: float,
    Fp_min: float, Fp_max: float,
    Fi_min: float, Fi_max: float,
    Fd_min: float, Fd_max: float,
) -> tuple[float, float, float, str]:
    """
    Table 2's triangular tuning rule (paper eq. "triangular").

    feats must be the 3-element list produced against standard_pid_features()
    ordering: feats[0]=Gamma0 (indicts Ki), feats[1]=Gamma1 (indicts Kp),
    feats[2]=Gamma2 (indicts Kd). Returns (Fp_new, Fi_new, Fd_new, row); row
    must be a MANUAL_READING key, since callers such as callbacks.py's
    on_iteration dereference MANUAL_READING[row] directly with no fallback.
    """
    Fp_new, Fi_new, Fd_new = Fp_cur, Fi_cur, Fd_cur

    if unstable:
        # Record grows rather than decays. Every gain comes down at once, and
        # far more coarsely than the gamma notch the count-based rows below
        # use -- "instability is a state to be exited quickly, not corrected
        # delicately."
        #
        # The cut deepens with the band's frequency: Ki /2, Kp /4, Kd /8. A
        # runaway is driven by the fast bands, so the derivative channel is
        # pulled back hardest and the reset -- which sets the settling time
        # the user is actually after -- is disturbed least. This is the one
        # row that fires without a usable count to attribute the trouble to,
        # so the attribution is made a priori by band instead.
        row = 'unstable'
        Fi_new = max(Fi_cur * 0.5, Fi_min)
        Fp_new = max(Fp_cur * 0.25, Fp_min)
        Fd_new = max(Fd_cur * 0.125, Fd_min)

    elif feats[0]['N'] > Nbar[0]:
        row = 'N0'
        Fi_new = max(Fi_cur / gamma, Fi_min)

    elif feats[1]['N'] > Nbar[1]:
        row = 'N1'
        Fi_new = min(Fi_cur * gamma, Fi_max)
        Fp_new = max(Fp_cur / gamma, Fp_min)

    elif feats[2]['N'] > Nbar[2]:
        row = 'N2'
        Fi_new = min(Fi_cur * gamma, Fi_max)
        Fp_new = min(Fp_cur * gamma, Fp_max)
        Fd_new = max(Fd_cur / gamma, Fd_min)

    else:
        # All features within limits: raise all bands.
        row = 'none'
        Fi_new = min(Fi_cur * gamma, Fi_max)
        Fp_new = min(Fp_cur * gamma, Fp_max)
        Fd_new = min(Fd_cur * gamma, Fd_max)

    return Fp_new, Fi_new, Fd_new, row


TUNING_RULES: dict[str, TuningRule] = {
    'triangular': triangular_rule,
}


class _Certifier:
    """
    State of certified / safe SPIN across iterations. Multipliers are handled
    in band order (Fi, Fp, Fd) = bands 0, 1, 2; a band move k lowers band k by
    one notch and raises the active bands below it by one, as the rule does.
    """

    def __init__(self, variant, base, Ts, gamma, Mstar, tau, lo, hi,
                 simtype, minu, maxu, noisy):
        self.variant, self.base, self.Ts, self.g = variant, np.asarray(base, float), Ts, gamma
        self.Mstar, self.tau = Mstar, tau
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.simtype, self.minu, self.maxu, self.noisy = simtype, minu, maxu, noisy
        self.bands = [k for k in range(3) if self.base[k] > 0]
        self.streak = 0                  # certified: band moves in a row
        self.safe = False                # safe: an accepted tuning has been seen
        self.F_cert = None               # safe: last accepted multipliers
        self.bad = set()                 # safe: tunings that failed their own record
        self.last_accepted = None        # the result: last accepted multipliers (Fi, Fp, Fd)

    def _clip(self, F):
        return np.clip(F, self.lo, self.hi)

    def _gains(self, F):
        Ki, Kp, Kd = self.base * F
        return (Kp, Ki, Kd)              # core.certificate order

    def _band_move(self, F, k):
        Fn = F.copy()
        for b in self.bands:
            if b < k:
                Fn[b] *= self.g
        Fn[k] /= self.g
        return self._clip(Fn)

    @staticmethod
    def _key(F):
        return tuple(np.round(np.log(np.maximum(F, 1e-300)), 6))

    def step(self, sigs, unstable, row, F_cur, F_rule):
        """Decide the next multipliers; returns (Fp, Fi, Fd, row)."""
        F = np.asarray(F_cur, float)
        e_r = -np.asarray(sigs['e'], float)          # the app's e is y - r
        why = check_record(e_r, sigs.get('u') if self.simtype == 1 else None,
                           self.minu, self.maxu, self.noisy)
        if why is None and unstable:
            why = 'record not settled'
        tau = tau_for(e_r, self.tau)                  # Assumption 1, with a data-based floor
        cert = certificate(e_r, tau) if why is None else None
        upper = cert['upper'] if cert else float('inf')
        accepted = upper <= self.Mstar
        info = {'upper': upper, 'why': why, 'accepted': accepted, 'U': None, 'tau': tau}
        sigs['cert'] = info

        if self.variant == 'certified':
            Fn = np.asarray(F_rule, float)
            if row == 'none':                         # the counts accept
                if accepted:
                    self.streak = 0
                    self.last_accepted = F.copy()
                else:
                    row = 'cert_reject'
                    move = None
                    if why is None and self.streak < N_BAND_MAX:
                        # a band move that (4) predicts to lower M_s: below the peak
                        # read from the record, not below the certified bound above it
                        down = [k for k in self.bands
                                if predict_Ms(e_r, self.Ts, self._gains(F),
                                              self._gains(self._band_move(F, k))) < cert['read']]
                        move = max(down) if down else None
                    if move is None:
                        Fn, self.streak = self._clip(F / self.g), 0
                    else:
                        Fn, self.streak = self._band_move(F, move), self.streak + 1
            else:
                self.streak = 0
            return float(Fn[1]), float(Fn[0]), float(Fn[2]), row

        # safe SPIN
        if not accepted:
            if not self.safe:                         # before the first accepted tuning
                Fn = np.asarray(F_rule, float) if unstable else self._clip(F / self.g)
                return float(Fn[1]), float(Fn[0]), float(Fn[2]), ('unstable' if unstable else 'warm')
            self.bad.add(self._key(F))                # certified ahead, not by its own record
            Fn = self.F_cert.copy()
            return float(Fn[1]), float(Fn[0]), float(Fn[2]), 'fallback'
        self.safe, self.F_cert = True, F.copy()
        self.last_accepted = F.copy()
        cands = [(row, self._clip(np.asarray(F_rule, float)))]
        cands += [('cert_band', self._band_move(F, k)) for k in sorted(self.bands, reverse=True)]
        cands += [('cert_back', self._clip(F / self.g))]
        for r, Fn in cands:
            if np.allclose(Fn, F) or self._key(Fn) in self.bad:
                continue
            A = certify_ahead(e_r, self.Ts, self._gains(F), self._gains(Fn),
                              tau, self.Mstar)
            if A['stable'] and A['U'] <= self.Mstar:
                info['U'] = A['U']
                return float(Fn[1]), float(Fn[0]), float(Fn[2]), r
        return float(F[1]), float(F[0]), float(F[2]), 'hold'


def pid_tuning(
    description: list[FeatureDescription],
    tau, K: float, L: float, Ts: float,
    Kp: float, Ki: float, Kd: float,
    dtype: str = 'y',
    Tsim: float | None = None,
    n_iter: int = 200,
    Fp_limits: tuple[float, float] = (0.01, 10.0),
    Fi_limits: tuple[float, float] = (0.01, 10.0),
    Fd_limits: tuple[float, float] = (0.01, 10.0),
    Nbar: tuple[float, ...] | None = None,
    beta: float = 0.1,
    simtype: int = 0,
    minu: float = -1.0, maxu: float = 1.0,
    dist_a: float = 0.0, dist_b: float = 0.0,
    delta: float = DELTA,
    eps: float = EPSILON,
    on_iteration: Callable[
        [int, int, float, float, float, str, list[dict], dict], None] | None = None,
    rule: TuningRule = triangular_rule,
    stability_screen: StabilityScreen = find_index,
    metric: EncirclementMetric = encirc,
    seed: int | None = None,
    variant: str = 'basic',
    Mstar: float = MSTAR,
    tau_cert: float = TAU_CERT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run n_iter iterations of the given tuning rule (default: the triangular rule).

    Returns (Fp_hist, Fi_hist, Fd_hist): history of multipliers for each iteration.
    The final multipliers are the last elements.

    Nbar: tuple of 3 loop limits N̄0, N̄1, N̄2 (one per feature); defaults to
          the Nbar stored in each FeatureDescription.
    beta: step size (paper's β); the applied notch is gamma = 1/(1-beta).
    delta: settling-band guard fraction of peak error (Definition 4).
    eps: truncation disc radius for the encirclement count (Definition 1).
    on_iteration: optional callback invoked as
                  on_iteration(i, n_iter, Fp_cur, Fi_cur, Fd_cur, row, feats,
                  sigs) once per iteration with the pre-update multipliers, the
                  MANUAL_READING key for the row that fired, and the features
                  and signal dict this iteration was actually scored on — so a
                  caller that wants to display the iteration does not have to
                  re-simulate it. Invoked once more after the loop with
                  (n_iter, n_iter, <final multipliers>, row, feats, sigs).
    rule: swappable gain-update decision, see TuningRule/TUNING_RULES.
    stability_screen: swappable instability test, see StabilityScreen/STABILITY_SCREENS.
    metric: swappable trajectory-scoring algorithm used by the per-iteration
            feature evaluation, see EncirclementMetric/ENCIRCLEMENT_METRICS.
    variant: 'basic' (published SPIN), 'certified' or 'safe' (see module
             docstring). The certified variants read the error record of each
             iteration; sigs['cert'] then carries the certified bound and the
             decision for display. They need the derivative on the error
             (dtype='e'), so that the record is the step response of S.
    Mstar: target sensitivity peak of the certified variants.
    tau_cert: least assumed bound on the unrecorded tail of the record
              (Assumption 1); core.certificate.tau_for raises it when the end of
              the record has not settled.
    seed: makes a run with output noise reproducible. Each iteration is given
          its own Generator derived from this seed, so iteration i always sees
          the same noise realization no matter how many iterations run --
          without it, every iteration draws a fresh realization and the same
          gains score differently each time they are visited.
    """
    if Nbar is None:
        Nbar = tuple(d.Nbar for d in description)

    n_iter = max(n_iter, 10)
    gamma = 1.0 / (1.0 - beta)
    Fp_min, Fp_max = min(Fp_limits), max(Fp_limits)
    Fi_min, Fi_max = min(Fi_limits), max(Fi_limits)
    Fd_min, Fd_max = min(Fd_limits), max(Fd_limits)

    Fp_hist = np.clip(np.ones(n_iter), Fp_min, Fp_max)
    Fi_hist = np.clip(np.ones(n_iter), Fi_min, Fi_max)
    Fd_hist = np.clip(np.ones(n_iter), Fd_min, Fd_max)

    row = 'none'
    seed_seq = None if seed is None else np.random.SeedSequence(seed)
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}; expected one of {VARIANTS}')
    cert_state = _Certifier(
        variant, (Ki, Kp, Kd), Ts, gamma, Mstar, tau_cert,
        lo=(Fi_min, Fp_min, Fd_min), hi=(Fi_max, Fp_max, Fd_max),
        simtype=simtype, minu=minu, maxu=maxu, noisy=(dist_b != 0.0),
    ) if variant != 'basic' else None

    for i in range(1, n_iter):
        Fp_cur = Fp_hist[i - 1]
        Fi_cur = Fi_hist[i - 1]
        Fd_cur = Fd_hist[i - 1]

        feats, k1, k2, sigs = loop_response_features(
            description,
            tau, K, L, Ts,
            Fp_cur * Kp, Fi_cur * Ki, Fd_cur * Kd,
            dtype=dtype, Tsim=Tsim,
            simtype=simtype, minu=minu, maxu=maxu,
            dist_a=dist_a, dist_b=dist_b, delta=delta, eps=eps,
            metric=metric,
            rng=(None if seed_seq is None
                 else np.random.default_rng(seed_seq.spawn(1)[0])),
        )

        _, unstable = stability_screen(k1, k2, sigs['e'])

        Fp_new, Fi_new, Fd_new, row = rule(
            feats, Nbar, unstable, Fp_cur, Fi_cur, Fd_cur, gamma,
            Fp_min, Fp_max, Fi_min, Fi_max, Fd_min, Fd_max,
        )

        if variant != 'basic':
            Fp_new, Fi_new, Fd_new, row = cert_state.step(
                sigs, unstable, row, (Fi_cur, Fp_cur, Fd_cur), (Fi_new, Fp_new, Fd_new))

        if on_iteration is not None:
            on_iteration(i, n_iter, float(Fp_cur), float(Fi_cur), float(Fd_cur),
                         row, feats, sigs)

        Fp_hist[i] = Fp_new
        Fi_hist[i] = Fi_new
        Fd_hist[i] = Fd_new

    # Certified variants return the last accepted tuning, the one their
    # guarantee is about, rather than wherever the last move happened to land.
    if cert_state is not None and cert_state.last_accepted is not None:
        Fi_a, Fp_a, Fd_a = cert_state.last_accepted
        Fp_hist[-1], Fi_hist[-1], Fd_hist[-1] = Fp_a, Fi_a, Fd_a

    if on_iteration is not None:
        on_iteration(n_iter, n_iter, float(Fp_hist[-1]), float(Fi_hist[-1]),
                     float(Fd_hist[-1]), row, feats, sigs)

    return Fp_hist, Fi_hist, Fd_hist
