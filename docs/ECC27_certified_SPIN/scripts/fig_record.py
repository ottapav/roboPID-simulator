"""
Figure 1: one setpoint step certifies M_s (Lemma 1, Theorem 1), on the slow-integral loop
(first-order plant T = 0.3 L_d, PI with T_i = 8T, gain margin 1.624; digital, T_s = 0.01 L_d).
  (a) the error after a unit setpoint step; the record ends at N = 8 L_d
  (b) the true |S| of the digital loop, |S| from the record (1), and the certified bounds (2)
      with tau the true remaining variation of the record
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin_cert_run import sim
from cert import increments, certificate, L_d

BLUE = "#2a78d6"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def slow_integral_Kp():
    """first-order plant T = 0.3 L_d, PI with Ti = 8T, Kc for gain margin 6 - 4.7*27/29 (continuous design)"""
    T, Ti = 0.3, 2.4; w = np.logspace(-4, 3, 200000)
    Lw = (1 + 1 / (Ti * 1j * w)) * np.exp(-1j * w) / (T * 1j * w + 1)
    ph = np.unwrap(np.angle(Lw))
    return 1 / abs(Lw[np.argmin(abs(ph + np.pi))]) / (6 - 4.7 * 27 / 29)


K, L, lags, Ti = 1.0, 1.0, [0.3], 2.4
Kp = slow_integral_Kp(); Ki = Kp / Ti
dt = 0.01; e = sim(K, L, lags, Ki, Kp, 0.0, dt, int(60 / dt)); a = increments(e); t = np.arange(len(e)) * dt
TN = 8.0; N = int(TN / dt); tau = float(np.sum(abs(a[N + 1:])))
c = certificate(e[:N + 1], tau, 2e-3)

w = np.logspace(-2, math.log10(math.pi / dt), 1500); th = w * dt; idx = np.arange(N + 1)
S_rec = np.array([abs(np.sum(a[:N + 1] * np.exp(-1j * x * idx))) for x in th])
S_true = abs(1 / (1 + L_d(K, L, lags, Ki, Kp, 0.0, dt, th)))

fig, (A, B) = plt.subplots(1, 2, figsize=(3.45, 1.55), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.42))
A.fill_between([0, 20], -0.02, 0.02, color=RULE, alpha=0.45, lw=0)
A.plot(t[:N + 1], e[:N + 1], color=BLUE, lw=1.1)
A.plot(t[N:int(20 / dt)], e[N:int(20 / dt)], color=RULE, lw=0.9)
A.axvline(TN, color=INK2, lw=0.6, ls=":")
A.text(TN + 0.5, 0.62, "end of\nrecord", fontsize=6.3, color=INK2)
A.set_xlim(0, 20); A.set_ylim(-0.1, 1.05); A.set_yticks([0, 0.5, 1])
A.set_xlabel("time $t/L_d$", labelpad=1); A.set_ylabel("error $e$", labelpad=1)
A.set_title("(a) one setpoint step", fontsize=8)

wp = w[np.argmax(S_true)]
B.semilogx(w, S_true, color=BLUE, lw=3.5, alpha=0.3, solid_capstyle="butt", label="true")
B.semilogx(w, S_rec, color=BLUE, lw=0.9, ls=(0, (4, 2)), label="from (a)")
for v in (c["lower"], c["upper"]):
    B.plot([wp / 2.2, wp * 2.2], [v, v], color=INK, lw=0.8)
B.text(wp * 2.5, (c["lower"] + c["upper"]) / 2, f"bounds\n{c['lower']:.2f}, {c['upper']:.2f}", fontsize=6.3, va="center")
B.axhline(1.7, color=INK2, lw=0.7, ls=(0, (4, 3))); B.text(0.012, 1.78, "$M^*$", fontsize=6.3, color=INK2)
B.set_xlim(0.01, math.pi / dt); B.set_ylim(0, 3.0); B.set_xticks([0.01, 1, 100])
B.set_xlabel("frequency $\\omega$ (rad/s)", labelpad=1); B.set_ylabel("$|S|$", labelpad=1)
B.set_title(f"(b) $|S|$, $M_s={S_true.max():.2f}$", fontsize=8)
B.legend(fontsize=6, frameon=False, loc="lower right", handlelength=1.8, labelspacing=0.2, borderaxespad=0.2)
fig.subplots_adjust(left=0.125, right=0.99, bottom=0.25, top=0.86)
fig.savefig("fig_record.pdf"); fig.savefig("fig_record.png", dpi=200)
print(f"M_s {S_true.max():.4f}  record {TN} L_d  tau {tau:.4f}  bounds {c['lower']:.4f} {c['upper']:.4f}")
