"""
AMIGO reference tuning (Astrom & Hagglund, Advanced PID Control, 2006).

The Step Response plot overlays the loop AMIGO produces on the same plant, as a
fixed baseline the slider or tuned gains can be read against. AMIGO is defined
on a FOPTD model K*exp(-L*s)/(T*s + 1), so the plant is reduced to one first by
the tangent method (foptd_fit), then the rule is applied (amigo_gains).

Gains come out in the parallel form the rest of core uses: Ki = Kp/Ti in 1/s,
Kd = Kp*Td in s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .params import HORIZON_SPANS

# Resolution of the fit's own grid. Independent of the display grid, so the
# reduced model -- and with it the reference gains -- does not move when the
# user edits the horizon.
FIT_SAMPLES_PER_SPAN = 2000


def foptd_fit(tau, L: float) -> tuple[float, float]:
    """
    (T, L_app) of the FOPTD model the tangent method reads off the plant's
    normalized open-loop step response.

    Only the lag chain is simulated. Dead time shifts the response rigidly, so
    it adds to the apparent dead time unchanged, and leaving it out keeps the
    fine fit grid away from the long delay polynomial core.params warns lfilter
    cannot handle. Each lag is its own first-order section, discretized as
    plant_tf does (the first carries the one-sample delay, the rest a zero at
    z=0), so the cascade stays well conditioned however many lags there are.

    The tangent at the steepest point crosses zero at L_lags; T runs from there
    to 63% of the final value.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    Ts = float(np.sum(tau)) / FIT_SAMPLES_PER_SPAN
    n = int(round(HORIZON_SPANS * FIT_SAMPLES_PER_SPAN)) + 1
    t = np.arange(n) * Ts

    y = np.ones(n)
    for i, tau_i in enumerate(tau):
        p = float(np.exp(-Ts / tau_i))
        num = [0.0, 1.0 - p] if i == 0 else [1.0 - p]
        y = lfilter(num, [1.0, -p], y)

    dy = np.gradient(y, Ts)
    k = int(np.argmax(dy))
    L_lags = max(float(t[k] - y[k] / dy[k]), 0.0)
    t63 = float(np.interp(1.0 - np.exp(-1.0), y, t))
    T = max(t63 - L_lags, Ts)
    return T, L + L_lags


def amigo_gains(K: float, T: float, L: float, ctype: str) -> tuple[float, float, float]:
    """
    (Kp, Ki, Kd) from the AMIGO rule for the FOPTD model (K, T, L).

    'PID' gets the AMIGO PID rule; every other structure gets AMIGO PI, which is
    also the reference shown for an I-only controller (AMIGO has no I rule).

    L must be positive: both rules scale Kp with T/L. A lag-only plant fits to
    L = 0, so the caller floors it -- at the sampling period, which is roughly
    the delay the sampled loop carries anyway.
    """
    if ctype == 'PID':
        Kp = (0.2 + 0.45 * T / L) / K
        Ti = L * (0.4 * L + 0.8 * T) / (L + 0.1 * T)
        Td = 0.5 * L * T / (0.3 * L + T)
        return Kp, Kp / Ti, Kp * Td

    Kp = (0.15 + (0.35 - L * T / (L + T) ** 2) * T / L) / K
    Ti = 0.35 * L + 13.0 * L * T ** 2 / (T ** 2 + 12.0 * L * T + 7.0 * L ** 2)
    return Kp, Kp / Ti, 0.0


@dataclass(frozen=True)
class AmigoDesign:
    """An AMIGO tuning together with the FOPTD model it was computed on."""
    structure: str      # 'PI' or 'PID'
    K: float
    T: float
    L: float            # apparent dead time the rule was applied with
    Kp: float
    Ki: float
    Kd: float


def amigo_design(tau, K: float, L: float, ctype: str,
                 L_min: float = 0.0) -> AmigoDesign | None:
    """
    The AMIGO tuning of this plant: foptd_fit, then amigo_gains.

    The one place the app turns a plant into AMIGO gains, so the admissibility
    target and the plotted reference cannot drift apart. ctype selects the rule
    as amigo_gains does, with unfiltered PID treated as PID.

    L_min floors the apparent dead time (see amigo_gains). Returns None where
    the rule is undefined: K not a positive finite number, or no positive dead
    time left after the floor.
    """
    if not (np.isfinite(K) and K > 0):
        return None
    structure = 'PID' if ctype in ('PID', 'PID_UNFILTERED') else 'PI'
    T, L_app = foptd_fit(tau, L)
    L_eff = max(L_app, L_min)
    if not L_eff > 0:
        return None
    Kp, Ki, Kd = amigo_gains(K, T, L_eff, structure)
    if not np.all(np.isfinite((Kp, Ki, Kd))):
        return None
    return AmigoDesign(structure=structure, K=float(K), T=float(T), L=float(L_eff),
                       Kp=float(Kp), Ki=float(Ki), Kd=float(Kd))
