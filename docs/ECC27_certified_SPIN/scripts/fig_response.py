"""
Figure 5: the tuned loops, P3 with PID, cautious start: the final tuning of basic, certified and
safe SPIN (the most aggressive tuning accepted in the last 60 iterations, as in Table II).
  (a) output after a unit setpoint step (the record the tuner reads, y = 1 - e)
  (b) output after a unit load step at the plant input (the load-step IAE of Table II)
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
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.4), gridspec_kw=dict(wspace=0.32))
for rule in ("original", "cert-shape", "cert-ahead"):
    r = RUNS[rule]; K, L, lags = r["K"], r["L"], r["lags"]; w = r["res"]["worst"]
    G = w["gains"]; dt = w["dt"]; Ta = L + sum(lags)
    n = int(6 * Ta / dt); e = sim(K, L, lags, *G, dt, n); t = np.arange(n) * dt
    lw = 1.3 if rule == "cert-ahead" else 0.9
    lab = f"{NAME[rule]}, $M_s$={w['Ms_d']:.2f}"
    a.plot(t, 1 - e, color=COL[rule], lw=lw, label=lab)
    tl, yl = load_response(K, L, lags, *G)
    b.plot(tl, yl, color=COL[rule], lw=lw, label=f"{NAME[rule]}, IAE={w['IAE']:.1f}")
    print(NAME[rule], "gains", np.round(G, 4), "Ms_d", round(w["Ms_d"], 3), "IAE", round(w["IAE"], 2),
          "IAE check", round(load_iae(K, L, lags, *G), 2), "overshoot", round(float((1 - e).max() - 1), 3))
a.axhline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
a.set_xlim(0, 90); a.set_ylim(0, 1.45); a.set_xlabel("time (s)"); a.set_ylabel("output $y$")
a.set_title("(a) setpoint step", fontsize=8, color=INK, pad=2)
b.axhline(0, color=INK2, lw=0.6)
b.set_xlim(0, 140); b.set_xticks([0, 50, 100]); b.set_xlabel("time (s)")
b.set_title("(b) load step", fontsize=8, color=INK, pad=2)
a.legend(fontsize=5.6, frameon=False, loc="lower right", handlelength=1.1, labelspacing=0.1,
         handletextpad=0.3, borderaxespad=0.0)
b.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.1, labelspacing=0.1,
         handletextpad=0.3, borderaxespad=0.0)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.24, top=0.9)
fig.savefig("../fig_response.pdf"); fig.savefig("../fig_response.png", dpi=200)
