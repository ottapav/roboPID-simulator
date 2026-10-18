"""
Figure 4: the tuned loops, P3 with PID, cautious start: the final tuning of basic, certified and
safe SPIN (the most aggressive tuning accepted in the last 60 iterations, as in Table III).
  (a) |S| of the digital loop, with M*
  (b) output after a unit setpoint step (the record the tuner reads, y = 1 - e)
  (c) output after a unit load step at the plant input (the load-step IAE of Table III)
Also prints the prediction (4) of one K_i band move from the record of the basic tuning, which the
text cites (predicted and true M_s after the move).
The simulations are those of the experiment: spin_cert_run.sim for the setpoint step and the loop
of compare.load_iae for the load step (same discretization, derivative filter T_f = T_d/10).
Reads safe1_*.jsonl (run_safe.sh).
"""
import glob, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin_cert_run import sim
from compare import load_iae
from cert import L_d, Ms_d, increments

INK, INK2 = "#0b0b0b", "#52514e"
COL = {"original": "#eb6834", "cert-shape": "#2a78d6", "cert-ahead": "#1f9e6a"}
NAME = {"original": "basic", "cert-shape": "certified", "cert-ahead": "safe"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_response(K, L, lags, Ki, Kp, Kd):
    """Output after a unit load step at the plant input: the loop of compare.load_iae."""
    Ta = L + sum(lags); Tf = Kd / Kp / 10 if Kd > 0 else 1e9; dt = min(Ta / 300, Tf / 5)
    n = int(40 * Ta / dt); nd = int(round(L / dt)); a = [math.exp(-dt / t) for t in lags]
    x = [0.0] * len(lags); af = math.exp(-dt / Tf) if Kd > 0 else 0.0
    ub = np.zeros(n + nd + 1); I = xf = y = 0.0; out = np.zeros(n)
    for i in range(n):
        ek = -y; D = (Kd / Tf) * (ek - xf) if Kd > 0 else 0.0
        if Kd > 0: xf = af * xf + (1 - af) * ek
        u = Kp * ek + Ki * I + D; I += ek * dt
        ub[i + nd] = K * (u + 1.0); v = ub[i]
        for j, aj in enumerate(a): x[j] = aj * x[j] + (1 - aj) * v; v = x[j]
        y = v; out[i] = y
    return np.arange(n) * dt, out


RUNS = {}
for f in sorted(glob.glob("safe1_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r["plant"] == "P3 PID" and r["rule"] in COL:
            RUNS[r["rule"]] = r
fig = plt.figure(figsize=(3.45, 2.75))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.32)
s_ax = fig.add_subplot(gs[0, :]); a = fig.add_subplot(gs[1, 0]); b = fig.add_subplot(gs[1, 1])
wS = np.logspace(-2, 0, 800)
for rule in ("original", "cert-shape", "cert-ahead"):
    r = RUNS[rule]; K, L, lags = r["K"], r["L"], r["lags"]; w = r["res"]["worst"]
    lw = 1.3 if rule == "cert-ahead" else 0.9
    S = abs(1 / (1 + L_d(K, L, lags, *w["gains"], w["dt"], wS * w["dt"])))
    s_ax.semilogx(wS, S, color=COL[rule], lw=lw, label=f"{NAME[rule]}, $M_s$={w['Ms_d']:.2f}")
s_ax.axhline(1.7, color=INK2, lw=0.7, ls=(0, (4, 3)))
s_ax.text(0.92, 1.72, "$M^*$", fontsize=6.5, color=INK2, va="bottom", ha="right")
s_ax.set_xlim(0.01, 1); s_ax.set_ylim(0, 2.0); s_ax.set_yticks([0, 1, 2])
s_ax.set_xticks([0.01, 0.1, 1]); s_ax.set_xticklabels(["0.01", "0.1", "1"])
s_ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
s_ax.set_xlabel("frequency $\\omega$ (rad/s)", labelpad=1); s_ax.set_ylabel("$|S|$")
s_ax.set_title("(a) sensitivity", fontsize=8, color=INK, pad=2)
s_ax.legend(fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.8), handlelength=1.2, labelspacing=0.1,
            handletextpad=0.3, borderaxespad=0.1)

for rule in ("original", "cert-shape", "cert-ahead"):
    r = RUNS[rule]; K, L, lags = r["K"], r["L"], r["lags"]; w = r["res"]["worst"]
    G = w["gains"]; dt = w["dt"]; Ta = L + sum(lags)
    n = int(6 * Ta / dt); e = sim(K, L, lags, *G, dt, n); t = np.arange(n) * dt
    lw = 1.3 if rule == "cert-ahead" else 0.9
    a.plot(t, 1 - e, color=COL[rule], lw=lw)
    tl, yl = load_response(K, L, lags, *G)
    b.plot(tl, yl, color=COL[rule], lw=lw, label=f"{NAME[rule]}, IAE={w['IAE']:.1f}")
    print(NAME[rule], "gains", np.round(G, 4), "Ms_d", round(w["Ms_d"], 3), "IAE", round(w["IAE"], 2),
          "IAE check", round(load_iae(K, L, lags, *G), 2), "overshoot", round(float((1 - e).max() - 1), 3))
a.axhline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
a.set_xlim(0, 90); a.set_ylim(0, 1.45); a.set_xlabel("time (s)"); a.set_ylabel("output $y$")
a.set_title("(b) setpoint step", fontsize=8, color=INK, pad=2)
b.axhline(0, color=INK2, lw=0.6)
b.set_xlim(0, 140); b.set_xticks([0, 50, 100]); b.set_xlabel("time (s)")
b.set_title("(c) load step", fontsize=8, color=INK, pad=2)
b.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=0.9, labelspacing=0.1,
         handletextpad=0.3, borderaxespad=0.0)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.93)
fig.savefig("../fig_response.pdf")

# the prediction (4) cited in the text: one K_i band move from the record of the basic tuning
r = RUNS["original"]; K, L, lags = r["K"], r["L"], r["lags"]; w = r["res"]["worst"]
G0 = np.array(w["gains"]); dt = w["dt"]; Ta = L + sum(lags)
e = sim(K, L, lags, *G0, dt, int(15 * Ta / dt)); an = increments(e); n_ = np.arange(len(an))
from cert import C_d
G1 = G0 * np.array([0.9, 1.0, 1.0])
th = np.linspace(1e-4, 0.6, 4000) * dt
Sr = np.array([np.sum(an * np.exp(-1j * x * n_)) for x in th])
pred = np.max(abs(Sr / (1 + (1 - Sr) * (C_d(*G1, dt, np.exp(1j * th)) / C_d(*G0, dt, np.exp(1j * th)) - 1))))
print(f"K_i move: predicted M_s {pred:.3f}, true {Ms_d(K, L, lags, *G1, dt)[0]:.3f}")
