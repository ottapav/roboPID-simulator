"""
Figure 3: original SPIN against certified SPIN.
  (a) all loops of Table III, grouped by the controller term that owns the
      sensitivity peak of the original tuning (largest elasticity of |C| at the peak)
  (b) the acceptance test at work on P1 with PID: true |S| and |S| read from the
      step for both tunings; at the peak of the original reading, K_p owns |C|
  (c) a plant family with growing dead-time share
Reads designed_*.jsonl (designed.py) and owner.json.
"""
import json, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from designed import loop_fr, S_rec, C_fr
from spin_cert_run import sim, elasticities

ORIG, CERT = "#eb6834", "#2a78d6"          # categorical slots 2 and 1 (validated pair)
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
TARGET = 1.7
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

# original = SPIN as published (gain box [0.001, 10]); certified = the same plus the acceptance test
def load(prefix, rule):
    return {json.loads(l)["plant"]: json.loads(l) for f in glob.glob(f"{prefix}_*.jsonl") for l in open(f)
            if json.loads(l)["rule"] == rule}
D = {}
for p, r in load("box", "original").items(): D[(p, "original")] = r
for p, r in load("shapehi", "original+shape-hi").items(): D[(p, "certified")] = r
for p, r in load("box2", "original").items(): D[(p, "original2")] = r          # start twice as bold
own = json.load(open("owner_box.json"))

GROUPS = [("$\\Gamma_0$\nintegral", ["delay-dominant PI", "P3 PI"]),
          ("$\\Gamma_1$\nproportional", ["P1 PI", "P2 PI", "P4 PI", "lag-dominant 2-lag PI", "lag-dominant PI",
                                         "P1 PID", "P2 PID", "P3 PID", "P4 PID"]),
          ("$\\Gamma_2$\nderiv.", ["fast PID"])]
KEY = {"$\\Gamma_0$": "G0", "$\\Gamma_1$": "G1", "$\\Gamma_2$": "G2"}
LABEL = {"delay-dominant PI": "delay\nPI", "lag-dominant PI": "lag\nPI", "lag-dominant 2-lag PI": "2-lag\nPI",
         "fast PID": "fast\nPID"}
LABEL.update({f"P{i} {c}": f"$P_{i}$\n{c}" for i in range(1, 5) for c in ("PI", "PID")})
for title, plants in GROUPS:                  # the grouping must match the computed owner
    want = [v for k, v in KEY.items() if k in title][0]
    for p in plants: assert own[p][0] == want, (p, own[p])

fig = plt.figure(figsize=(7.16, 2.4))
gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.25, 0.95], wspace=0.36)
a, b, c = (fig.add_subplot(gs[i]) for i in range(3))

# (a) --------------------------------------------------------------------------
x = 0; ticks = []; labels = []; spans = []
for title, plants in GROUPS:
    x0 = x
    for p in plants:
        mo = D[(p, "original")]["res"]["worst"]["Ms"]; mc = D[(p, "certified")]["res"]["worst"]["Ms"]
        a.plot([x, x], [mo, mc], color=RULE, lw=1.4, zorder=1, solid_capstyle="round")
        a.plot(x, mo, "o", ms=8, color=ORIG, mec="white", mew=1.0, zorder=3)
        a.plot(x, mc, "o", ms=4.8, color=CERT, mec="white", mew=0.8, zorder=4)
        m2 = D[(p, "original2")]["res"]["worst"]["Ms"]
        a.plot(x + 0.28, m2, "o", ms=4.6, mfc="white", mec=ORIG, mew=1.1, zorder=3)
        ticks.append(x); labels.append(LABEL[p]); x += 1
    spans.append((x0 - 0.5, x - 0.5, title)); x += 0.6
a.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.8, zorder=0)
a.text(6.4, TARGET + 0.012, "target $M^*$", fontsize=6.5, color=INK2, ha="right")
for i, (lo, hi, title) in enumerate(spans):
    a.text((lo + hi) / 2, 2.17, title, ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.1)
    if i: a.axvline(lo - 0.3, color=RULE, lw=0.6, ls=":")
a.annotate("counts alone\nkeep the target:\nsame tuning", xy=(0.5, 1.655), xytext=(-0.4, 1.43),
           ha="left", fontsize=6.3, color=INK2,
           arrowprops=dict(arrowstyle="-", color=RULE, lw=0.7))
a.set_xticks(ticks); a.set_xticklabels(labels, fontsize=6.3)
a.set_xlim(-0.6, x - 0.9); a.set_ylim(1.4, 2.18)
a.set_ylabel("$M_s$ of the tuning SPIN accepts")
a.set_title("(a) the twelve loops of the table", fontsize=8, color=INK)
a.plot([], [], "o", ms=7, color=ORIG, label="original SPIN")
a.plot([], [], "o", ms=4.6, mfc="white", mec=ORIG, mew=1.1, label="original, bolder start")
a.plot([], [], "o", ms=4.6, color=CERT, label="certified SPIN")
a.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.75, 2.0), bbox_transform=a.transData,
         handletextpad=0.2, borderaxespad=0.0, labelspacing=0.25)

# (b) --------------------------------------------------------------------------
def record(K, L, lags, Ki, Kp, Kd):
    """the setpoint-step record, sampled as in designed.py"""
    Ta = L + sum(lags); dt = min(15 * Ta / 1500, (Kd / Kp / 10) / 20)
    return sim(K, L, lags, Ki, Kp, Kd, dt, int(15 * Ta / dt)), dt


p = "P3 PID"; r = D[(p, "original")]; K, L, lags = r["K"], r["L"], r["lags"]
w = np.logspace(-2, 0.7, 700)
g0 = D[(p, "original")]["res"]["worst"]["gains"]
e, dt = record(K, L, lags, *g0)
wr = np.logspace(math.log10(0.01 / (L + sum(lags))), math.log10(0.3 * math.pi / dt), 600)
Sr = S_rec(e, dt, wr); ip = int(np.argmax(abs(Sr))); wp = wr[ip]; Mp = abs(Sr[ip])
Lr = 1 / S_rec(e, dt, w) - 1                                     # loop read from the step
Cw = C_fr(*g0, w); Tp = 1 - Sr[ip]; Cp = C_fr(*g0, np.array([wp]))[0]
grad = []
for k in range(3):
    f = np.ones(3); f[k] = math.exp(1e-6); ck = (C_fr(*(np.array(g0) * f), np.array([wp]))[0] - Cp) / 1e-6
    grad.append(float((-(Tp * ck / Cp)).real))
gg = 1 / 0.9
Fn = np.array([1 / gg, 1.0, 1.0]); g1 = list(np.array(g0) * Fn)       # move of band 0: lower K_i
pred = abs(1 / (1 + Lr * C_fr(*g1, w) / Cw))
true1 = abs(1 / (1 + loop_fr(K, L, lags, *g1, w)))
b.semilogx(w, abs(1 / (1 + loop_fr(K, L, lags, *g0, w))), color=ORIG, lw=4, alpha=0.3, solid_capstyle="butt")
b.semilogx(w, abs(1 / S_rec(e, dt, w) ** -1) if False else abs(S_rec(e, dt, w)), color=ORIG, lw=1.0, ls=(0, (4, 2)))
b.semilogx(w, true1, color=CERT, lw=4, alpha=0.3, solid_capstyle="butt")
b.semilogx(w, pred, color=CERT, lw=1.0, ls=(0, (4, 2)))
b.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.8)
b.plot(wp, Mp, "o", ms=5, mfc="white", mec=INK, mew=1.0, zorder=5)
b.annotate(f"read {Mp:.2f} > $M^*$ at $\\omega_p$;\n$g=({grad[0]:.2f},{grad[1]:.2f},{grad[2]:.0f})$"
           .replace("{grad[2]:.0f}", "") if False else
           f"read {Mp:.2f} > $M^*$ at $\\omega_p$\n$g_i,g_p,g_d={grad[0]:.2f},{grad[1]:.2f},{grad[2]:.2f}$\n$\\Rightarrow$ lower $K_i$ only",
           xy=(wp, Mp), xytext=(0.0115, 2.0), fontsize=6.3, color=INK, va="bottom",
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
b.text(0.0125, TARGET - 0.03, "target $M^*$", fontsize=6.3, color=INK2, va="top")
b.plot([], [], color=ORIG, lw=4, alpha=0.5, label="before: true")
b.plot([], [], color=ORIG, lw=1.0, ls=(0, (4, 2)), label="before: read from step")
b.plot([], [], color=CERT, lw=4, alpha=0.5, label="after the move: true")
b.plot([], [], color=CERT, lw=1.0, ls=(0, (4, 2)), label="after: predicted from step")
b.set_xlim(w[0], w[-1]); b.set_ylim(0, 2.6); b.set_xlabel("frequency $\\omega$ (rad/s)"); b.set_ylabel("$|S(j\\omega)|$")
b.set_title("(b) one band move on $P_3$ with PID", fontsize=8, color=INK)
b.legend(fontsize=5.6, frameon=False, loc="lower right", handlelength=1.8, labelspacing=0.2)
print(p, ": omega_p", round(wp, 4), "read", round(Mp, 3), "grad", np.round(grad, 3), "pred max", round(pred.max(), 3), "true new", round(true1.max(), 3),
      "max |pred-true|/true %", round(100 * np.max(abs(pred - true1) / true1), 2))

# (c) --------------------------------------------------------------------------
fam = sorted([k[0] for k in D if k[0].startswith("F L=") and k[1] == "original"
              and D[k]["res"] is not None and D.get((k[0], "certified"), {}).get("res")],
             key=lambda s: float(s[4:]))
lags = D[(fam[0], "original")]["lags"]
share = [float(s[4:]) / (float(s[4:]) + sum(lags)) for s in fam]
mo = [D[(s, "original")]["res"]["worst"]["Ms"] for s in fam]
mc = [D[(s, "certified")]["res"]["worst"]["Ms"] for s in fam]
for xs, o, q in zip(share, mo, mc): c.plot([xs, xs], [o, q], color=RULE, lw=1.4, zorder=1)
c.plot(share, mo, "o-", color=ORIG, ms=5.5, lw=0.9, mec="white", mew=0.9, label="original SPIN", zorder=3)
c.plot(share, mc, "o-", color=CERT, ms=4.2, lw=0.9, mec="white", mew=0.8, label="certified SPIN", zorder=4)
m2 = [D[(s, "original2")]["res"]["worst"]["Ms"] for s in fam]              # bold start
c.plot(share, m2, "o", ms=4.2, mfc="white", mec=ORIG, mew=1.0, zorder=3)
c.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.8)
c.set_xlim(0, 1); c.set_ylim(1.4, 2.18); c.set_xticks([0, 0.25, 0.5, 0.75, 1]); c.set_xticklabels(["0", ".25", ".5", ".75", "1"])
c.set_xlabel("dead-time share"); c.set_ylabel("$M_s$ of the tuning SPIN accepts")
c.set_title("(c) a plant family, PID", fontsize=8, color=INK)
c.legend(fontsize=6.3, frameon=False, loc="lower left", handletextpad=0.3)

fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.9)
fig.savefig("fig_certified.pdf"); fig.savefig("fig_certified.png", dpi=150)
print("family shares", [round(s, 2) for s in share])
