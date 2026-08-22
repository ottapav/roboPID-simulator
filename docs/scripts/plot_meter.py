"""
What the drift meter reads.

Runs the published rule (core.tuning, unmodified) and, from the recorded row
that fires at each iteration, computes the meter of Proposition 3:

    reading = || M^T p ||        p = move shares over a sliding window of 8

Top panel  : the three gains, log scale.
Bottom     : the meter. High while the gains travel, zero once the moves
             cancel and the state circles the triple point.

The point of the figure is that the lower curve is computed from the move
sequence alone -- no plant, no extra experiment, no magnitudes.
"""
from __future__ import annotations
import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import regindex
regindex.patch()
from core.features import standard_pid_features
from core.tuning import pid_tuning

TS = 1.0
W = 8                                  # window = one period (Corollary 2)
MOVES = {"none": np.array([1., 1., 1.]),      # e      (band order Ki,Kp,Kd)
         "N0":   np.array([-1., 0., 0.]),     # d0
         "N1":   np.array([1., -1., 0.]),     # d1
         "N2":   np.array([1., 1., -1.])}     # d2
ORDER = ["none", "N0", "N1", "N2"]
M = np.array([MOVES[k] for k in ORDER])

PLANT = ([10., 1., 1., 1.], 1.0, 1.0)          # P1
START = dict(Kp=1.0, Ki=1.0, Kd=1.0)
N_ITER = 120


def run():
    tau, K, L = PLANT
    rows, Fp, Fi, Fd = [], [], [], []

    def cb(i, n, fp, fi, fd, row):
        rows.append(row)

    fp, fi, fd = pid_tuning(standard_pid_features(), np.array(tau), K, L, TS,
                            START["Kp"], START["Ki"], START["Kd"], dtype="y",
                            n_iter=N_ITER, beta=0.1, on_iteration=cb)
    return rows, np.array(fi), np.array(fp), np.array(fd)


def meter(rows, W=W):
    """|| M^T p || over a sliding window; None where the window is not full."""
    out = []
    for i in range(len(rows)):
        if i + 1 < W:
            out.append(np.nan); continue
        win = rows[i + 1 - W:i + 1]
        p = np.array([win.count(k) for k in ORDER], float)
        if p.sum() == 0:
            out.append(np.nan); continue
        p /= p.sum()
        out.append(float(np.linalg.norm(M.T @ p)))
    return np.array(out)


def main():
    rows, Fi, Fp, Fd = run()
    m = meter(rows)
    n = min(len(m), len(Fi), len(Fp), len(Fd))
    it = np.arange(n)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 3.1), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1]))

    a1.semilogy(it, Fi[:n], color="#6b2e9e", lw=1.8, label="$K_i$")
    a1.semilogy(it, Fp[:n], color="#c77a06", lw=1.8, label="$K_p$")
    a1.semilogy(it, Fd[:n], color="#1f5fa9", lw=1.8, label="$K_d$")
    a1.set_ylabel("gains", fontsize=9)
    a1.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    a1.grid(alpha=.25, lw=.5)

    a2.plot(it, m[:n], color="#12212f", lw=1.8)
    a2.axhline(0, color="#9aa7b4", lw=.8)
    a2.set_ylabel(r"meter  $\|M^{\top}p\|$", fontsize=9)
    a2.set_xlabel("iteration", fontsize=9)
    a2.grid(alpha=.25, lw=.5)
    a2.set_ylim(-0.08, 1.85)

    # annotate the two regimes from the meter itself
    finite = np.where(np.isfinite(m[:n]))[0]
    if len(finite):
        lo = np.where(m[:n] < 0.15)[0]
        if len(lo):
            k = int(lo[0])
            for ax in (a1, a2):
                ax.axvline(k, color="#b8112b", lw=1.0, ls="--")
            a2.annotate("moves cancel:\non the cycle", (k, 1.15),
                        xytext=(k + 6, 1.35), fontsize=8, color="#b8112b",
                        arrowprops=dict(arrowstyle="->", color="#b8112b", lw=.9))
            a2.annotate("gains travelling", (k * 0.35, 1.5), fontsize=8,
                        color="#12212f", ha="center")
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig_meter.pdf")
    fig.savefig("/mnt/user-data/outputs/fig_meter.png", dpi=250)
    print("rows:", len(rows), " meter tail:", np.round(m[-12:], 3))


if __name__ == "__main__":
    main()
