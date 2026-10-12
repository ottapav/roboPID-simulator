"""
Figure 1: what a turn count is, on the lag-cancelled loop L = k e^{-s}/s (L_d = 1).
  (a) the three whitened counts divided by ln(1/delta) against the true M_s: at delta = 0.02 a
      staircase set by the trimming, at delta = 1e-8 close to the limit omega/(2 pi |sigma|) of
      Proposition 2, which does not depend on delta (data: counts_delta.json, confirm_counts_delta.py)
  (b) one error record (k = 0.7) with its setpoint crossings marked
  (c) the same record as the portrait Gamma_1 = (de, e), whitened: between two
      crossings the curve turns by exactly half a turn around the end point
"""
import math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spin_run as S
from lagcancel import records, Ms

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
DELTA, DT = 0.02, 0.01
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- data for (a): counts at delta = 0.02 and 1e-8, divided by ln(1/delta), and the limit
SMALL = 1e-8
ks = np.linspace(0.2, 1.0, 240)
ms = np.array([Ms(k) for k in ks]); N2, N8 = np.zeros((len(ks), 3)), np.zeros((len(ks), 3))
for c0 in range(0, len(ks), 40):
    kk = ks[c0:c0 + 40]; t, E = records(kk, tend=120.0); step = int(round(DT / (t[1] - t[0])))
    for j in range(len(kk)):
        N2[c0 + j] = S.counts(E[::step, j], DT, DELTA); N8[c0 + j] = S.counts(E[::step, j], DT, SMALL)
D = json.load(open("counts_delta.json"))
mf, rf = np.array(D["Ms"]), np.array(D["rate"])

fig = plt.figure(figsize=(3.45, 3.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.42)
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])

M0 = Ms(1 / math.e)
a.axvline(M0, color=RULE, lw=0.7, ls=":", zorder=0)
a.text(M0 + 0.025, 0.3, f"$M_s={M0:.3f}$", fontsize=6.3, color=INK2, ha="left")
a.plot(mf, rf, color=ORANGE, lw=3.4, alpha=0.35, solid_capstyle="butt", zorder=1,
       label="$\\delta\\to0$: $\\omega/(2\\pi|\\sigma|)$, Prop. 2")
a.plot(ms, N2[:, 0] / math.log(1 / DELTA), color=INK2, lw=0.8, ls=(0, (2, 1.5)), zorder=2,
       label=f"$\\delta={DELTA:g}$ ($\\Gamma_0$)")
for kk, col in enumerate([INK, BLUE, ORANGE]):
    a.plot(ms, N8[:, kk] / math.log(1 / SMALL), color=col, lw=0.8, zorder=3,
           label="$\\delta=10^{-8}$ ($\\Gamma_0,\\Gamma_1,\\Gamma_2$)" if kk == 0 else None)
a.set_xlim(1.15, 3.15); a.set_ylim(0, 0.75)
a.set_xlabel("$M_s$ of the loop"); a.set_ylabel("$N_k/\\ln(1/\\delta)$")
a.set_title("(a) the counts per $\\ln(1/\\delta)$ against $M_s$", fontsize=8, color=INK)
a.legend(fontsize=6.3, frameon=False, loc="upper left", handlelength=1.8, labelspacing=0.25)

# ---- data for (b), (c): one record
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

# Gamma_0 = (e, E - E_end), whitened. The radius is drawn compressed (r -> r^0.3) so the
# small late turns stay visible; the angle, which is all a count measures, is unchanged.
I = np.cumsum(e) * DT
Y = S.whiten(e[:kd], (I - I[-1])[:kd]); R = np.hypot(*Y); Z = Y * R ** (0.3 - 1)
k1 = np.where(R > 0.1)[0][0]                     # the count starts where the curve leaves the disc
p = Z[:, zc[0]] / np.hypot(*Z[:, zc[0]])         # image of the setpoint line e = 0
c.plot([-1.25 * p[0], 1.25 * p[0]], [-1.25 * p[1], 1.25 * p[1]], color=INK2, lw=0.7, ls=(0, (3, 2)))
c.text(1.2 * p[0] + 0.05, 1.2 * p[1], "$e=0$", fontsize=6.3, color=INK2, ha="left", va="center")
c.plot(Z[0, :k1 + 1], Z[1, :k1 + 1], color=RULE, lw=1.2)
c.plot(Z[0, k1:], Z[1, k1:], color=BLUE, lw=1.2)
c.plot(*Z[:, k1], "o", ms=4, mfc="white", mec=INK, mew=0.9, zorder=4)
c.text(Z[0, k1] + 0.1, Z[1, k1], "start", fontsize=6.3, color=INK2, va="center")
c.plot(0, 0, "+", color=INK, ms=6, mew=1.0)
for n, i in enumerate(zc, 1):
    c.plot(*Z[:, i], "o", ms=4.5, color=INK, zorder=4)
    nrm = np.array([p[1], -p[0]]) * (0.15 if n == 1 else -0.15)   # beside the line, not on it
    c.text(Z[0, i] + nrm[0], Z[1, i] + nrm[1], f"{n}", fontsize=7, color=INK, ha="center", va="center")
m = (zc[0] + zc[1]) // 2; u = Z[:, m] / np.hypot(*Z[:, m])
c.text(*(Z[:, m] + 0.2 * u), "$\\frac{1}{2}$ turn", fontsize=6.5, color=INK, ha="center", va="center")
Nk = S.counts(e, DT, DELTA)
c.text(0.0, 0.93, f"count\n$N_0={Nk[0]:.2f}$", transform=c.transAxes, fontsize=6.5, color=INK2, va="top")
c.set_aspect("equal"); c.set_xlim(-1.05, 1.05); c.set_ylim(-1.05, 1.05); c.axis("off")
c.set_title("(c) portrait $\\Gamma_0$", fontsize=8, color=INK)

fig.subplots_adjust(left=0.13, right=0.97, bottom=0.115, top=0.94)
fig.savefig("fig_stair.pdf"); fig.savefig("fig_stair.png", dpi=200)
print("crossings in (b):", len(zc))
ang = np.unwrap(np.arctan2(Y[1], Y[0]))
print("turn between crossings:", [round((ang[zc[i + 1]] - ang[zc[i]]) / (2 * math.pi), 4) for i in range(len(zc) - 1)])
