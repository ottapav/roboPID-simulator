"""
Figure 3: P3 with PID, exact M_s of the digital loop.
  (a) one band move on P3 with PID: true |S_d| before, the prediction of Prop. 3(d) from the same
      record, and the true |S_d| after the move
  (b) M_s of the tuning applied in each iteration, published, certified and safe SPIN
      (traj_P3PID.json from traj_P3PID.py)
Reads safe1_*.jsonl (run_safe.sh) and traj_P3PID.json (traj_P3PID.py).
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
ORDER = ["P1 PI", "P1 PID", "P2 PI", "P2 PID", "P3 PI", "P3 PID", "P4 PI", "P4 PID",
         "delay-dominant PI", "lag-dominant PI", "lag-dominant 2-lag PI", "fast PID"]
LABEL = {"delay-dominant PI": "delay\nPI", "lag-dominant PI": "lag\nPI", "lag-dominant 2-lag PI": "2-lag\nPI",
         "fast PID": "fast\nPID"}
LABEL.update({f"P{i} {c}": f"$P_{i}$\n{c}" for i in range(1, 5) for c in ("PI", "PID")})

fig = plt.figure(figsize=(3.45, 1.45))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.55)
b, c = (fig.add_subplot(gs[i]) for i in range(2))

# (b) ----------------------------------------------------------------------------------------
RUNS = [json.loads(l) for f in glob.glob("safe1_*.jsonl") for l in open(f)]       # fixed T_s, cautious start
r = [x for x in RUNS if x["plant"] == "P3 PID" and x["rule"] == "original"][0]
K, L, lags = r["K"], r["L"], r["lags"]; Ta = L + sum(lags)
G0 = np.array(r["res"]["worst"]["gains"]); dt = r["res"]["worst"]["dt"]
e = sim(K, L, lags, *G0, dt, int(15 * Ta / dt)); ce = certificate(e, 1e-3)
w = np.logspace(-2, 0.7, 700); th = w * dt
a_n = increments(e); idx = np.arange(len(a_n))
Sr = np.array([np.sum(a_n * np.exp(-1j * t * idx)) for t in th])
z = np.exp(1j * th); C0 = C_d(*G0, dt, z); Lr = 1 / Sr - 1
ip = int(np.argmax(abs(Sr))); wp = w[ip]
g = 1 / 0.9; G1 = G0 * np.array([1 / g, 1.0, 1.0])                     # move of the integral band
pred = abs(1 / (1 + Lr * C_d(*G1, dt, z) / C0))
S0 = abs(1 / (1 + L_d(K, L, lags, *G0, dt, th))); S1 = abs(1 / (1 + L_d(K, L, lags, *G1, dt, th)))
rc = [x for x in RUNS if x["plant"] == "P3 PID" and x["rule"] == "cert-shape"][0]
Gc = np.array(rc["res"]["worst"]["gains"]); dtc = rc["res"]["worst"]["dt"]
Sc = abs(1 / (1 + L_d(K, L, lags, *Gc, dtc, w * dtc)))
b.semilogx(w, S0, color=ORIG, lw=4, alpha=0.35, solid_capstyle="butt")
b.semilogx(w, S1, color=INK2, lw=0.9)
b.semilogx(w, pred, color=INK, lw=1.0, ls=(0, (2, 2)))
b.semilogx(w, Sc, color=CERT, lw=1.8)
b.axhline(TARGET, ls=(0, (4, 3)), color=INK2, lw=0.8)
b.plot(wp, S0[ip], "o", ms=5, mfc="white", mec=INK, mew=1.0, zorder=5)
b.text(0.073, TARGET - 0.03, "$M^*$", fontsize=6.3, color=INK2, va="top")
b.plot([], [], color=ORIG, lw=4, alpha=0.5, label="published")
b.plot([], [], color=INK2, lw=0.9, label="one move ($K_i\\downarrow$)")
b.plot([], [], color=INK, lw=1.0, ls=(0, (2, 2)), label="its prediction")
b.plot([], [], color=CERT, lw=1.8, label="certified")
b.set_xlim(0.07, 0.45); b.set_ylim(0.75, 2.9)
b.set_xticks([0.1, 0.2, 0.4]); b.set_xticklabels(['0.1', '0.2', '0.4']); b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); b.set_xlabel("frequency $\\omega$ (rad/s)"); b.set_ylabel("$|S|$")
b.set_title("(a) near the peak", fontsize=8, color=INK)
b.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.3, labelspacing=0.1, handletextpad=0.3, borderaxespad=0.0)
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
c.set_xlabel("iteration"); c.set_ylabel("$M_s$ of applied tuning")
c.set_title("(b) while tuning", fontsize=8, color=INK)
c.legend(fontsize=6.0, frameon=False, loc="lower right", handlelength=1.2, labelspacing=0.15,
         handletextpad=0.3, borderaxespad=0.1)
fig.subplots_adjust(left=0.13, right=0.99, bottom=0.24, top=0.88)
fig.savefig("fig_certified.pdf"); fig.savefig("fig_certified.png", dpi=150)
