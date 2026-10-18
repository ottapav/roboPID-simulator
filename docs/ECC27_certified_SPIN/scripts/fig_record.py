"""
Figure 1: one setpoint step certifies M_s (Lemma 1, Theorem 1), on the running example: P3 with PID,
the final tuning of basic SPIN from the cautious start (as in Table III and Figs. 3-5), on the
digital loop of the experiment (fixed T_s, record of 15 T_a, tau = 1e-3, eta = 2e-3).
  (a) the error after a unit setpoint step, with the +-2% band
  (b) the true |S| of the digital loop, |S| from the record (1), the certified bounds (2), and M*
Reads safe1_*.jsonl (run_safe.sh). Also prints the gain margin and the largest crossing of the
slow-integral loop that Section IV-C cites (first-order plant T_1 = 0.3 L_d, PI with T_i = 8 T_1,
T_s = 0.01 L_d): M_s = 2.62 without a crossing above 2 %.
"""
import glob, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin_cert_run import sim
from cert import increments, certificate, L_d

BLUE = "#2a78d6"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
TAU, ETA, MSTAR = 1e-3, 2e-3, 1.7
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

r = next(json.loads(l) for f in sorted(glob.glob("safe1_*.jsonl")) for l in open(f)
         if json.loads(l)["plant"] == "P3 PID" and json.loads(l)["rule"] == "original")
K, L, lags = r["K"], r["L"], r["lags"]; w0 = r["res"]["worst"]
G = w0["gains"]; dt = w0["dt"]; Ta = L + sum(lags)
N = int(15 * Ta / dt); e = sim(K, L, lags, *G, dt, N); a = increments(e); t = np.arange(N) * dt
c = certificate(e, TAU, ETA)
long = sim(K, L, lags, *G, dt, 4 * N); tail = float(np.sum(abs(increments(long)[N:])))

w = np.logspace(-2.5, math.log10(math.pi / dt), 1200); th = w * dt
from scipy.signal import czt
S_rec = np.array([abs(np.sum(a * np.exp(-1j * x * np.arange(len(a))))) for x in th])
S_true = abs(1 / (1 + L_d(K, L, lags, *G, dt, th)))

fig, (A, B) = plt.subplots(1, 2, figsize=(3.45, 1.42), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.42))
A.fill_between([0, 100], -0.02, 0.02, color=INK2, alpha=0.18, lw=0)
A.plot(t, e, color=BLUE, lw=1.1)
A.set_xlim(0, 100); A.set_ylim(-0.1, 1.05); A.set_yticks([0, 0.5, 1]); A.set_xticks([0, 50, 100])
A.set_xlabel("time (s)", labelpad=1); A.set_ylabel("error $e$", labelpad=1)
A.set_title("(a) one setpoint step", fontsize=8)

wp = w[np.argmax(S_true)]
B.semilogx(w, S_true, color=BLUE, lw=3.5, alpha=0.3, solid_capstyle="butt", label="true")
B.semilogx(w, S_rec, color=BLUE, lw=0.9, ls=(0, (4, 2)), label="from (a)")
B.plot([wp / 3, wp * 3], [c["upper"], c["upper"]], color=INK, lw=0.8)
B.text(wp * 3.4, c["upper"] + 0.05, f"$\\overline{{M}}={c['upper']:.2f}$", fontsize=6.5, va="bottom")
B.axhline(MSTAR, color=INK2, lw=0.7, ls=(0, (4, 3))); B.text(0.004, MSTAR - 0.08, "$M^*$", fontsize=6.5,
                                                             color=INK2, va="top")
B.set_xlim(10 ** -2.5, math.pi / dt); B.set_ylim(0, 2.3); B.set_xticks([0.01, 1, 100])
B.set_xlabel("frequency $\\omega$ (rad/s)", labelpad=1); B.set_ylabel("$|S|$", labelpad=1)
B.set_title(f"(b) $|S|$, $M_s={S_true.max():.2f}$", fontsize=8)
B.legend(fontsize=6, frameon=False, loc="lower left", handlelength=1.8, labelspacing=0.2, borderaxespad=0.2)
fig.subplots_adjust(left=0.125, right=0.99, bottom=0.27, top=0.85)
fig.savefig("../fig_record.pdf")
print(f"P3 PID basic: gains {np.round(G, 4)} dt {dt}  M_s {S_true.max():.4f}  bounds {c['lower']:.4f} "
      f"{c['upper']:.4f}  tail after record {tail:.2e}  min e {e.min():.4f}")

# the slow-integral loop of Section IV-C (formerly Fig. 1): M_s = 2.62 without a crossing above 2 %
T1, Ti = 0.3, 2.4; ww = np.logspace(-4, 3, 200000)
Lw = (1 + 1 / (Ti * 1j * ww)) * np.exp(-1j * ww) / (T1 * 1j * ww + 1)
ph = np.unwrap(np.angle(Lw)); Kp = 1 / abs(Lw[np.argmin(abs(ph + np.pi))]) / (6 - 4.7 * 27 / 29)
e2 = sim(1.0, 1.0, [T1], Kp / Ti, Kp, 0.0, 0.01, 6000)
S2 = abs(1 / (1 + L_d(1.0, 1.0, [T1], Kp / Ti, Kp, 0.0, 0.01, np.linspace(1e-4, math.pi, 200000))))
print(f"slow-integral loop: M_s {S2.max():.3f}  largest crossing {-e2.min():.4f}")
