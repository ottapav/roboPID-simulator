"""
Figure 4: tuning history on P3 with PID (cautious start, gain box, fixed T_s), as the roboPID app
shows it: gains K_i, K_p, K_d (top) and M_s of the applied tuning (bottom) in every iteration, for
basic, certified and safe SPIN. Reads traj_P3PID.json and traj_P3PID_gains.json (traj_P3PID.py).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2 = "#0b0b0b", "#52514e"
KI, KP, KD = "#6b2e9e", "#c77a06", "#1f5fa9"          # band colors of the poster and the app
OVER = "#d62728"
TARGET, XMAX = 1.7, 120
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

Ms = json.load(open("traj_P3PID.json"))
G = json.load(open("traj_P3PID_gains.json"))
RULES = [("original", "basic"), ("cert-shape", "certified"), ("cert-ahead", "safe")]

fig, ax = plt.subplots(2, 3, figsize=(3.45, 2.1), sharex=True, sharey="row",
                       gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.12, wspace=0.12))
for j, (rule, name) in enumerate(RULES):
    g = np.array(G[rule]); m = np.array(Ms[rule]); it = np.arange(len(m))
    top, bot = ax[0, j], ax[1, j]
    for k, (c, lab) in enumerate([(KI, "$K_i$"), (KP, "$K_p$"), (KD, "$K_d$")]):
        top.step(it, g[:, k], where="post", color=c, lw=0.9, label=lab)
    top.set_yscale("log")
    top.set_title(name, fontsize=8, color=INK, pad=2)
    mc = np.minimum(m, 2.3)
    bot.step(it, mc, where="post", color=INK2, lw=0.7)
    above = m > TARGET
    bot.plot(it[above], mc[above], "o", ms=1.6, color=OVER, mew=0)
    bot.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.7)
    bot.text(XMAX - 2, 1.27, f"{int(above.sum())} above", ha="right", va="bottom", fontsize=6.3,
             color=OVER if above.any() else INK2)
    bot.set_xlim(0, XMAX); bot.set_xticks([0, 50, 100])
    bot.set_ylim(1.25, 2.3)
    if j == 0:
        top.set_ylabel("gain"); bot.set_ylabel("$M_s$")
        bot.text(3, TARGET + 0.02, "$M^*$", fontsize=6.3, color=INK2, va="bottom")
    bot.set_xlabel("iteration" if j == 1 else "")
ax[0, 2].legend(fontsize=6.0, frameon=False, loc="lower right", handlelength=1.0,
                labelspacing=0.1, handletextpad=0.3, borderaxespad=0.0, ncol=3, columnspacing=0.6)
fig.subplots_adjust(left=0.15, right=0.99, bottom=0.19, top=0.92)
fig.savefig("../fig_history.pdf"); fig.savefig("../fig_history.png", dpi=200)
for rule, name in RULES:
    m = np.array(Ms[rule]); g = np.array(G[rule])
    print(name, "above M*:", int((m > TARGET).sum()), "of", len(m), " max", round(float(m.max()), 3),
          " final gains", np.round(g[-1], 4))
