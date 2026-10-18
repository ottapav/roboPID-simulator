"""
Figure 2: what a turn count is, on the lag-cancelled loop L = k e^{-s}/s (L_d = 1), with SPIN's counts
(Definition 1 of SPIN: each axis divided by its peak).
  (a) the three counts at delta = 0.02 in turns against M_s, each with its published limit: a staircase
      whose steps sit where the (n+1)-th extremum equals delta (1.534, 1.842, 2.210)
  (b) one error record (k = 0.7) with its setpoint crossings marked
  (c) the same record as the portrait Gamma_0: between two crossings it turns by exactly half a turn
"""
import math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spin_run as S
from compare import counts_old          # SPIN Definition 1 (per-axis peak normalization)
from lagcancel import records, Ms

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
DELTA, DT = 0.02, 0.01
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- data for (a): the counts at delta = 0.02
ks = np.linspace(0.2, 1.0, 240)
ms = np.array([Ms(k) for k in ks]); N2 = np.zeros((len(ks), 3))
for c0 in range(0, len(ks), 40):
    kk = ks[c0:c0 + 40]; t, E = records(kk, tend=60.0); step = int(round(DT / (t[1] - t[0])))
    for j in range(len(kk)):
        N2[c0 + j] = counts_old(E[::step, j], DT, DELTA)

fig = plt.figure(figsize=(3.45, 2.25))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=1.0, wspace=0.42)
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])

# (a) robustness from the limits: the counts at delta = 0.02 in turns, each band with its published limit
# On this ideal loop de/dt jumps at t = L_d, so Gamma_2 starts with a quarter turn clockwise
# (N_2 = n/2 - 1/4). N_2 and its limit are drawn a quarter turn up, so the three staircases align.
SHIFT = [0.0, 0.0, 0.25]
LAB = ["$\\Gamma_0$, $\\bar N_0=0.5$", "$\\Gamma_1$, $\\bar N_1=0.75$", "$\\Gamma_2+\\frac{1}{4}$, $\\bar N_2+\\frac{1}{4}=1.25$"]
for kk, (col, lim) in enumerate([(INK, 0.5), (BLUE, 0.75), (ORANGE, 1.0)]):
    a.plot(ms, N2[:, kk] + SHIFT[kk], color=col, lw=0.9, label=LAB[kk])
    a.axhline(lim + SHIFT[kk], color=col, lw=0.6, ls=(0, (1, 2)), zorder=0)
for x in (1.534, 1.842, 2.210):
    a.axvline(x, color=RULE, lw=0.6, ls=":", zorder=0)
    a.text(x, 1.88, f"{x:.2f}", fontsize=5.6, color=INK2, ha="center", va="bottom", clip_on=False)
a.set_xlim(1.15, 2.55); a.set_ylim(-0.15, 1.85); a.set_yticks([0, 0.5, 1, 1.5])
a.set_yticklabels(["0", "½", "1", "1½"])
a.set_xlabel("$M_s$ of the loop"); a.set_ylabel("$N_k$ (turns)", labelpad=1)
a.set_title(f"(a) the counts at $\\delta={DELTA:g}$ and their limits", fontsize=8, color=INK, pad=9)
a.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.2, labelspacing=0.15, borderaxespad=0.1)

# ---- (b), (c): one record
k0 = 0.7
t1, E1 = records([k0], tend=40.0); e = E1[::step, 0]; tt = t1[::step]
kd = min(len(e), np.where(abs(e) > DELTA * abs(e).max())[0][-1] + 2)
zc = [i for i in range(1, kd) if np.sign(e[i]) != np.sign(e[i - 1]) and e[i] != 0]

b.axhline(0, color=RULE, lw=0.7)
b.fill_between([0, tt[kd - 1]], -DELTA, DELTA, color=RULE, alpha=0.35, lw=0)
b.plot(tt[:kd], e[:kd], color=BLUE, lw=1.2)
b.plot(tt[kd:], e[kd:], color=RULE, lw=1.0)
for n, i in enumerate(zc, 1):
    b.plot(tt[i], 0, "o", ms=4.5, color=INK, zorder=4)
    b.text(tt[i], 0.14 if n % 2 else -0.24, f"{n}", fontsize=7, color=INK, ha="center")
b.set_xlim(0, 12); b.set_ylim(-0.45, 1.1)
b.set_xlabel("time $t/L_d$"); b.set_ylabel("error $e$")
b.set_title(f"(b) error, $M_s={Ms(k0):.2f}$", fontsize=8, color=INK)

# Gamma_0 = (e, E - E_end), each axis divided by its peak (SPIN Definition 1). The radius is drawn
# compressed (r -> r^0.3) so the small late turns stay visible; the angle is unchanged.
I = np.cumsum(e) * DT
P0, Q0 = e[:kd], (I - I[-1])[:kd]
Y = np.vstack([P0 / abs(P0).max(), Q0 / abs(Q0).max()]); R = np.hypot(*Y); Z = Y * R ** (0.3 - 1)
k1 = np.where(R > 0.1)[0][0]                     # the count starts where the curve leaves the disc
p = np.array([0.0, 1.0])                         # the setpoint line e = 0 is the vertical axis
c.plot([0, 0], [-1.25, 1.25], color=INK2, lw=0.7, ls=(0, (3, 2)))
c.text(0.06, 1.18, "$e=0$", fontsize=6.3, color=INK2, ha="left", va="center")
c.plot(Z[0, :k1 + 1], Z[1, :k1 + 1], color=RULE, lw=1.2)
c.plot(Z[0, k1:], Z[1, k1:], color=BLUE, lw=1.2)
c.plot(*Z[:, k1], "o", ms=4, mfc="white", mec=INK, mew=0.9, zorder=4)
c.text(Z[0, k1] + 0.1, Z[1, k1], "start", fontsize=6.3, color=INK2, va="center")
c.plot(0, 0, "+", color=INK, ms=6, mew=1.0)
for n, i in enumerate(zc, 1):
    c.plot(*Z[:, i], "o", ms=4.5, color=INK, zorder=4)
    c.text(Z[0, i] + 0.2, Z[1, i] + (0.12 if n == 1 else -0.12), f"{n}", fontsize=7, color=INK, ha="center", va="center")
m = (zc[0] + zc[1]) // 2; u = Z[:, m] / np.hypot(*Z[:, m])
c.text(*(Z[:, m] + 0.25 * u), "$\\frac{1}{2}$ turn", fontsize=6.5, color=INK, ha="center", va="center")
Nk = counts_old(e, DT, DELTA)
c.set_aspect("equal"); c.set_xlim(-1.15, 1.15); c.set_ylim(-1.15, 1.15); c.axis("off")
c.set_title(f"(c) $\\Gamma_0$, $N_0={Nk[0]:.2f}$", fontsize=8, color=INK)

fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.88)
fig.savefig("fig_stair.pdf"); fig.savefig("fig_stair.png", dpi=200)
print("crossings in (b):", len(zc), " N0 =", round(Nk[0], 3))
ang = np.unwrap(np.arctan2(Y[1], Y[0]))
print("turn between crossings:", [round((ang[zc[i + 1]] - ang[zc[i]]) / (2 * math.pi), 4) for i in range(len(zc) - 1)])
