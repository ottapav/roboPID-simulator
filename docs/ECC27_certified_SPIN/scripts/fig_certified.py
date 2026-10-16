"""
Figure 3: published SPIN against certified SPIN (Algorithm 2), exact M_s of the digital loop.
  (a) the twelve loops of Table II, grouped by the band that owns the peak of the published tuning
  (b) one band move on P3 with PID: true |S_d| before, the prediction of Prop. 3(d) from the same
      record, and the true |S_d| after the move
  (c) P3 with PID: M_s of the tuning applied in each iteration, published, certified and safe SPIN
      (traj_P3PID.json from traj_P3PID.py)
Reads summary_cert.json (analyze_cert.py), owner_box.json, box_*.jsonl.
"""
import json, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin_cert_run import sim
from cert import L_d, C_d, increments, certificate, S_grid

ORIG, CERT = "#eb6834", "#2a78d6"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
TARGET = 1.7
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open("summary_cert.json"))
ORDER = ["P1 PI", "P1 PID", "P2 PI", "P2 PID", "P3 PI", "P3 PID", "P4 PI", "P4 PID",
         "delay-dominant PI", "lag-dominant PI", "lag-dominant 2-lag PI", "fast PID"]
LABEL = {"delay-dominant PI": "delay\nPI", "lag-dominant PI": "lag\nPI", "lag-dominant 2-lag PI": "2-lag\nPI",
         "fast PID": "fast\nPID"}
LABEL.update({f"P{i} {c}": f"$P_{i}$\n{c}" for i in range(1, 5) for c in ("PI", "PID")})

fig = plt.figure(figsize=(7.16, 1.85))
gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1.2, 1.1], wspace=0.36)
a, b, c = (fig.add_subplot(gs[i]) for i in range(3))

# (a) ----------------------------------------------------------------------------------------
ticks = list(range(len(ORDER))); labels = [LABEL[p] for p in ORDER]
for x, p in enumerate(ORDER):
    mo, mc, m2 = S[p]["box"], S[p]["cbox"], S[p]["box2"]
    a.plot([x, x], [mo, mc], color=RULE, lw=1.4, zorder=1, solid_capstyle="round")
    a.plot(x, mo, "o", ms=8, color=ORIG, mec="white", mew=1.0, zorder=3)
    a.plot(x, mc, "o", ms=4.8, color=CERT, mec="white", mew=0.8, zorder=4)
    a.plot(x + 0.3, m2, "o", ms=4.6, mfc="white", mec=ORIG, mew=1.1, zorder=3)
a.axvline(7.6, color=RULE, lw=0.6, ls=":")
a.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.8, zorder=0)
a.text(11.6, TARGET + 0.012, "target $M^*$", fontsize=6.5, color=INK2, ha="right")
a.set_xticks(ticks); a.set_xticklabels(labels, fontsize=5.9)
a.set_xlim(-0.6, len(ORDER) - 0.3); a.set_ylim(1.4, 2.18)
a.set_ylabel("$M_s$ of the accepted tuning")
a.set_title("(a) the twelve loops of the table", fontsize=8, color=INK)
a.plot([], [], "o", ms=7, color=ORIG, label="published SPIN")
a.plot([], [], "o", ms=4.6, mfc="white", mec=ORIG, mew=1.1, label="published, bold start")
a.plot([], [], "o", ms=4.6, color=CERT, label="certified SPIN")
a.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(-0.4, 2.17), bbox_transform=a.transData,
         handletextpad=0.2, borderaxespad=0.0, labelspacing=0.25)

# (b) ----------------------------------------------------------------------------------------
r = [json.loads(l) for f in glob.glob("box_*.jsonl") for l in open(f)
     if json.loads(l)["plant"] == "P3 PID" and json.loads(l)["rule"] == "original"][0]
K, L, lags = r["K"], r["L"], r["lags"]; Ta = L + sum(lags)
G0 = np.array(r["res"]["worst"]["gains"]); dt = min(15 * Ta / 1500, (G0[2] / G0[1] / 10) / 20)
e = sim(K, L, lags, *G0, dt, int(15 * Ta / dt)); ce = certificate(e, 1e-3)
w = np.logspace(-2, 0.7, 700); th = w * dt
a_n = increments(e); idx = np.arange(len(a_n))
Sr = np.array([np.sum(a_n * np.exp(-1j * t * idx)) for t in th])
z = np.exp(1j * th); C0 = C_d(*G0, dt, z); Lr = 1 / Sr - 1
ip = int(np.argmax(abs(Sr))); wp = w[ip]
g = 1 / 0.9; G1 = G0 * np.array([1 / g, 1.0, 1.0])                     # move of the integral band
pred = abs(1 / (1 + Lr * C_d(*G1, dt, z) / C0))
S0 = abs(1 / (1 + L_d(K, L, lags, *G0, dt, th))); S1 = abs(1 / (1 + L_d(K, L, lags, *G1, dt, th)))
rc = [json.loads(l) for l in open("cbox_0.jsonl") if json.loads(l)["plant"] == "P3 PID"][0]
Gc = np.array(rc["res"]["worst"]["gains"]); dtc = min(15 * Ta / 1500, (Gc[2] / Gc[1] / 10) / 20)
Sc = abs(1 / (1 + L_d(K, L, lags, *Gc, dtc, w * dtc)))
b.semilogx(w, S0, color=ORIG, lw=4, alpha=0.35, solid_capstyle="butt")
b.semilogx(w, S1, color=INK2, lw=0.9)
b.semilogx(w, pred, color=INK, lw=1.0, ls=(0, (2, 2)))
b.semilogx(w, Sc, color=CERT, lw=1.8)
b.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.8)
b.plot(wp, S0[ip], "o", ms=5, mfc="white", mec=INK, mew=1.0, zorder=5)
b.text(0.073, TARGET - 0.03, "target $M^*$", fontsize=6.3, color=INK2, va="top")
b.plot([], [], color=ORIG, lw=4, alpha=0.5, label="published")
b.plot([], [], color=INK2, lw=0.9, label="one move ($K_i\\downarrow$)")
b.plot([], [], color=INK, lw=1.0, ls=(0, (2, 2)), label="its prediction")
b.plot([], [], color=CERT, lw=1.8, label="certified")
b.set_xlim(0.07, 0.45); b.set_ylim(0.75, 2.3)
b.set_xticks([0.1, 0.2, 0.4]); b.set_xticklabels(['0.1', '0.2', '0.4']); b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); b.set_xlabel("frequency $\\omega$ (rad/s)"); b.set_ylabel("$|S|$")
b.set_title("(b) $P_3$ with PID, near the peak", fontsize=8, color=INK)
b.legend(fontsize=5.8, frameon=False, loc="upper right", handlelength=1.6, labelspacing=0.15, borderaxespad=0.1)
print("P3 PID: upper", round(ce["upper"], 4), "true before", round(S0.max(), 4), "pred", round(pred.max(), 5),
      "true after", round(S1.max(), 5))

# (c) ----------------------------------------------------------------------------------------
T = json.load(open("traj_P3PID.json")); SAFE = "#1f9e6a"
it = np.arange(len(T["original"]))
c.step(it, T["original"], where="post", color=ORIG, lw=0.8, label="published")
c.step(it, T["cert-shape"], where="post", color=CERT, lw=0.8, label="certified")
c.step(it, T["cert-ahead"], where="post", color=SAFE, lw=1.1, label="safe")
c.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.8)
c.set_xlim(0, 80); c.set_ylim(1.3, 2.1); c.set_xticks([0, 40, 80])
c.set_xlabel("iteration"); c.set_ylabel("$M_s$ of the applied tuning")
c.set_title("(c) $P_3$, PID, while tuning", fontsize=8, color=INK)
c.legend(fontsize=6.0, frameon=False, loc="lower right", handlelength=1.2, labelspacing=0.15,
         handletextpad=0.3, borderaxespad=0.1)
fig.subplots_adjust(left=0.065, right=0.99, bottom=0.22, top=0.89)
fig.savefig("fig_certified.pdf"); fig.savefig("fig_certified.png", dpi=150)
