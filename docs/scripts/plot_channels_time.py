"""
Time-domain view of the three Pachner portraits.

The identity behind it (verified against core/pid.py::action_components):

    u  =  K_i * E   +   K_p * e   +   K_d * de        (E = running sum of e)
          -------       -------       --------
           u_I           u_P           u_D

and the three portraits of the companion paper are exactly the phase planes
(xdot, x) of those same three signals:

    Gamma_0 : x = E   = u_I / K_i     -> turn index N_0 -> gain K^(0) = K_i
    Gamma_1 : x = e   = u_P / K_p     -> turn index N_1 -> gain K^(1) = K_p
    Gamma_2 : x = de  = u_D / K_d     -> turn index N_2 -> gain K^(2) = K_d

So N_k is the turn index of the phase portrait of precisely the signal that
the gain K^(k) multiplies. The plot below shows those three signals against
time, which is the time-domain form of the same diagnosis: a channel whose
contribution rings is the channel whose gain is at fault.

Only ONE step response exists; these are three linear views of it, not three
independent records.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.features import standard_pid_features, loop_response_features

# P2 of the battery, deliberately mis-tuned so that bands 1 and 2 object
TAU, KGAIN, LDEL, TS = np.array([5., 5., 5., 5.]), 1.25, 8.0, 1.0
GAINS = dict(Kp=0.6, Ki=0.10, Kd=1.2)
NBAR = (0.5, 0.75, 1.0)

CH = [("$u_I = K_i E$", "#1b6ca8", "K_i"),
      ("$u_P = K_p e$", "#c0392b", "K_p"),
      ("$u_D = K_d \\dot e$", "#2e8b57", "K_d")]


def main():
    desc = standard_pid_features()
    feats, k1, k2, s = loop_response_features(
        desc, TAU, KGAIN, LDEL, TS,
        GAINS["Kp"], GAINS["Ki"], GAINS["Kd"], dtype="y")
    N = [float(f["N"]) for f in feats]
    kd_ = int(s["k_delta"])

    t = s["t"]
    n = min(len(t), len(s["uP"]))
    t = t[:n]
    eps = -s["e"][:n]                       # control error r - y
    chans = [s["uI"][:n], s["uP"][:n], s["uD"][:n]]

    fig, axes = plt.subplots(4, 1, figsize=(6.4, 6.6), sharex=True)

    ax = axes[0]
    ax.plot(t, eps, color="0.15", lw=1.6)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.axvline(t[min(kd_, n - 1)], color="0.6", lw=0.8, ls=":")
    ax.set_ylabel("error $e$", fontsize=9)
    ax.set_title("One step response, three channel views  (P2, mis-tuned)",
                 fontsize=10)

    for j, (lab, col, gname) in enumerate(CH):
        ax = axes[j + 1]
        ax.plot(t, chans[j], color=col, lw=1.6)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axvline(t[min(kd_, n - 1)], color="0.6", lw=0.8, ls=":")
        ax.set_ylabel(lab, fontsize=9)
        viol = N[j] > NBAR[j]
        ax.text(0.985, 0.86,
                f"$N_{j}$ = {N[j]:.2f}   (limit {NBAR[j]})"
                + ("   violated" if viol else "   ok"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color=("#b03060" if viol else "0.35"))

    axes[-1].set_xlabel("time", fontsize=9)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig_channels_time.png", dpi=240)
    fig.savefig("/mnt/user-data/outputs/fig_channels_time.pdf")
    print("N =", [round(x, 3) for x in N], " k_delta =", kd_)


if __name__ == "__main__":
    main()
