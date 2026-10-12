"""
Figure 2: Lemma 2 and Theorem 1.
  (a) true |S_d| of the digital loop (thick, light) and |S_d| read from one setpoint step (dashed),
      for P4 with SIMC PI and for the slow-integral loop that no turn count detects
  (b) certified lower and upper bounds (Theorem 1, tau = 1e-3) relative to the true M_s, for every
      tuning accepted in the SPIN runs (thm1_points.json, made by confirm_thm1.py)
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin_cert_run import sim
from cert import L_d, increments

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def simc_pi(K, L, lags):
    l = sorted(lags, reverse=True); tau = l[0] + l[1] / 2; th = L + l[1] / 2 + sum(l[2:])
    Kc = tau / (K * 2 * th); Ti = min(tau, 8 * th)
    return Kc / Ti, Kc


def slow_integral_Kp():
    """first-order plant T = 0.3 L_d, PI with Ti = 8T, Kc for gain margin 6 - 4.7*27/29 (continuous design)"""
    T, Ti = 0.3, 2.4; w = np.logspace(-4, 3, 200000)
    Lw = (1 + 1 / (Ti * 1j * w)) * np.exp(-1j * w) / (T * 1j * w + 1)
    ph = np.unwrap(np.angle(Lw))
    return 1 / abs(Lw[np.argmin(abs(ph + np.pi))]) / (6 - 4.7 * 27 / 29)


def panel_curve(ax, K, L, lags, Ki, Kp, dt, n, w, col):
    e = sim(K, L, lags, Ki, Kp, 0.0, dt, n); a = increments(e)
    th = w * dt; idx = np.arange(len(a))
    Sr = np.array([np.sum(a * np.exp(-1j * t * idx)) for t in th])
    St = 1 / (1 + L_d(K, L, lags, Ki, Kp, 0.0, dt, th))
    ax.loglog(w, abs(St), color=col, lw=4, alpha=0.3, solid_capstyle="butt")
    ax.loglog(w, abs(Sr), color=col, lw=1.0, ls=(0, (4, 2)))
    return float(abs(St).max())


fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.28))
K, L, lags = 1.0, 4.0, [8] * 6; Ki, Kp = simc_pi(K, L, lags); Ta = L + sum(lags); dt = Ta / 1000
m1 = panel_curve(a, K, L, lags, Ki, Kp, dt, int(15 * Ta / dt), np.logspace(-3, math.log10(0.6), 300), BLUE)
a.text(2.2e-3, 0.3, "$P_4$, SIMC PI\n$M_s=%.2f$" % m1, color=INK, fontsize=6.8)
Kp = slow_integral_Kp()
m2 = panel_curve(a, 1.0, 1.0, [0.3], Kp / 2.4, Kp, 0.01, int(150 / 0.01), np.logspace(-2, math.log10(3.2), 300), ORANGE)
a.text(0.35, 0.05, "slow integral:\nno crossing,\n$M_s=%.2f$" % m2, color=INK, fontsize=6.8)
a.plot([], [], color=INK2, lw=4, alpha=0.3, label="true $|S_d|$")
a.plot([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label="from one step, (5)")
a.legend(fontsize=6.8, frameon=False, loc="upper left", handlelength=2.2)
a.set_xlim(1e-3, 4); a.set_ylim(2e-2, 5)
a.set_xlabel("frequency $\\omega$ (rad/s)"); a.set_ylabel("$|S_d|$", labelpad=1)
a.set_title("(a) the step gives the whole sensitivity function", fontsize=8, color=INK)

P = json.load(open("thm1_points.json"))
for grp, col in [("published SPIN", ORANGE), ("certified SPIN", BLUE)]:
    x = np.array([p["Ms"] for p in P if p["group"] == grp])
    lo = np.array([p["lower"] for p in P if p["group"] == grp]) - x
    up = np.array([p["upper"] for p in P if p["group"] == grp]) - x
    b.plot(x, 1e3 * up, "o", ms=3.6, color=col, mec="white", mew=0.5, ls="none", label=f"{grp}, upper bound")
    b.plot(x, 1e3 * lo, "o", ms=3.6, mfc="white", mec=col, mew=0.8, ls="none", label=f"{grp}, lower bound")
b.axhline(0, color=INK2, lw=0.7)
b.set_xlabel("true $M_s$ of the digital loop"); b.set_ylabel("bound $-$ true $M_s$  ($10^{-3}$)")
b.set_title("(b) the certified bounds bracket every $M_s$", fontsize=8, color=INK)
b.legend(fontsize=5.8, frameon=False, loc="upper right", handletextpad=0.2, labelspacing=0.2)
yl = b.get_ylim(); b.set_ylim(yl[0], yl[1] + 0.6 * (yl[1] - yl[0]))
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.89)
fig.savefig("fig_record.pdf"); fig.savefig("fig_record.png", dpi=200)
print("P4 SIMC PI Ms_d", round(m1, 4), " slow integral Ms_d", round(m2, 4))
