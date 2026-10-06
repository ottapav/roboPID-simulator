"""
Figure 2: Theorem 1, S(jw) = 1 + int de/dt e^{-jwt} dt, read from one setpoint step.
  (a) true |S| (thick, light) and |S| read from the step (dashed) for a SIMC PI loop
      and for the slow-integral loop that no turn count detects
  (b) reading error of M_s against the true M_s for
        - the tunings accepted by certified SPIN (read inside the rule, designed_*.jsonl)
        - the tunings accepted by original SPIN (read here, after the run)
        - SIMC PI on P1..P4, the slow-integral loop, and the lag-cancelled loop
Data are cached in record_points.json; delete it to recompute.
"""
import json, glob, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin_cert_run import sim
from designed import S_rec, loop_fr
from lagcancel import records, Ms as Ms_lc

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
PLANTS = {"P1": (1.0, 1.0, [10, 1, 1, 1]), "P2": (1.25, 8.0, [5, 5, 5, 5]),
          "P3": (1.0, 10.0, [2, 1, 1, 1]), "P4": (1.0, 4.0, [8] * 6)}


def true_Ms(K, L, lags, Ki, Kp, Kd):
    w = np.logspace(-4, 2.5, 80000)
    return float(np.max(1 / abs(1 + loop_fr(K, L, lags, Ki, Kp, Kd, w))))


def read_Ms(K, L, lags, Ki, Kp, Kd, pid, tend=None):
    """same settings as the acceptance test in designed.py"""
    Ta = L + sum(lags); dts = 15 * Ta / 1500
    dt = min(dts, (Kd / Kp / 10) / 20) if pid else Ta / 1000
    n = int((tend or 15 * Ta) / dt); e = sim(K, L, lags, Ki, Kp, Kd, dt, n)
    w = np.logspace(math.log10(0.01 / Ta), math.log10(min(3 / L, 0.3 * math.pi / dt)), 500)
    return float(np.max(abs(S_rec(e, dt, w)))), e, dt


def simc_pi(K, L, lags):
    l = sorted(lags, reverse=True); tau = l[0] + l[1] / 2; th = L + l[1] / 2 + sum(l[2:])
    Kc = tau / (K * 2 * th); Ti = min(tau, 8 * th)
    return Kc / Ti, Kc


def slow_integral():
    """T = 0.3 L_d, Ti = 8T, gain margin 6 - 4.7*27/29 = 1.624: the record never crosses the setpoint by more than 2%
    (the no-crossing loop with the largest M_s in the 630-loop grid of reproduce.py --grid)"""
    T, Ti = 0.3, 2.4; w = np.logspace(-4, 3, 200000)
    Lw = (1 + 1 / (Ti * 1j * w)) * np.exp(-1j * w) / (T * 1j * w + 1)
    ph = np.unwrap(np.angle(Lw)); Kp = 1 / abs(Lw[np.argmin(abs(ph + np.pi))]) / (6 - 4.7 * 27 / 29)
    e = sim(1.0, 1.0, [T], Kp / Ti, Kp, 0.0, 0.01, int((60 + 30 * Ti) / 0.01))
    kc = np.where(abs(e) > 0.02)[0][-1] + 1; sg = np.sign(e[:kc]); sg = sg[sg != 0]
    assert np.sum(sg[1:] != sg[:-1]) == 0
    return Kp, true_Ms(1.0, 1.0, [T], Kp / Ti, Kp, 0.0)


if not os.path.exists("record_points.json"):
    P = {"certified": [], "original": [], "other": []}
    R = [json.loads(l) for f in glob.glob("designed_*.jsonl") for l in open(f)]
    for r in R:
        if r["res"] is None: continue
        for c in r["res"]["accepted"]:
            if r["rule"] == "certified":
                P["certified"].append([r["plant"], c["Ms"], c["read"]])
            else:
                rd = read_Ms(r["K"], r["L"], r["lags"], *c["gains"], r["pid"])[0]
                P["original"].append([r["plant"], c["Ms"], rd])
            print(r["plant"], r["rule"], P[r["rule"]][-1][1:], flush=True)
    for name, (K, L, lags) in PLANTS.items():
        Ki, Kp = simc_pi(K, L, lags)
        P["other"].append([name + " SIMC PI", true_Ms(K, L, lags, Ki, Kp, 0), read_Ms(K, L, lags, Ki, Kp, 0, False)[0]])
    Kp, Ms = slow_integral()
    P["slow_Kp"] = Kp
    P["other"].append(["slow integral", Ms, read_Ms(1.0, 1.0, [0.3], Kp / 2.4, Kp, 0, False, tend=150)[0]])
    for k in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        t, E = records([k], dt=1e-3, tend=120.0)
        wg = np.linspace(0.05, 3.0, 3000)
        P["other"].append([f"lag-cancelled k={k}", Ms_lc(k), float(np.max(abs(S_rec(E[:, 0], 1e-3, wg))))])
    for row in P["other"]: print(row, flush=True)
    json.dump(P, open("record_points.json", "w"), indent=1)
P = json.load(open("record_points.json"))
# add the runs of SPIN with its gain box (both starts), once; tunings already read are not repeated
if "boxed" not in P:
    seen = {(p[0], round(p[1], 6)) for k in ("original", "certified") for p in P[k]}
    for pre, rule, key in [("box", "original", "original"), ("box2", "original", "original"),
                           ("boxtest", "original+test", "certified"), ("box2test", "original+test", "certified")]:
        for f in glob.glob(f"{pre}_*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r["rule"] != rule or r["res"] is None: continue
                for c in r["res"]["accepted"]:
                    if (r["plant"], round(c["Ms"], 6)) in seen: continue
                    seen.add((r["plant"], round(c["Ms"], 6)))
                    rd = c["read"] if key == "certified" else read_Ms(r["K"], r["L"], r["lags"], *c["gains"], r["pid"])[0]
                    P[key].append([r["plant"], c["Ms"], rd]); print(pre, r["plant"], c["Ms"], rd, flush=True)
    P["boxed"] = True
    json.dump(P, open("record_points.json", "w"), indent=1)

if "shaped" not in P:
    # tunings accepted by certified SPIN (read inside the rule)
    P["certified"] = []
    for pre in ("shapehi", "shapehi2"):
        for l in open(f"{pre}_0.jsonl"):
            r = json.loads(l)
            if r["res"] is None: continue
            for c in r["res"]["accepted"]:
                P["certified"].append([r["plant"], c["Ms"], c["read"]])
    P["shaped"] = True
    json.dump(P, open("record_points.json", "w"), indent=1)

fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.28))

# (a) --------------------------------------------------------------------------
K, L, lags = PLANTS["P4"]; Ki, Kp = simc_pi(K, L, lags)
Ta = L + sum(lags); _, e, dt = read_Ms(K, L, lags, Ki, Kp, 0, False)
w1 = np.logspace(-3, math.log10(0.6), 400)
a.loglog(w1, abs(1 / (1 + loop_fr(K, L, lags, Ki, Kp, 0, w1))), color=BLUE, lw=4, alpha=0.3, solid_capstyle="butt")
a.loglog(w1, abs(S_rec(e, dt, w1)), color=BLUE, lw=1.0, ls=(0, (4, 2)))
a.text(2.2e-3, 0.3, "$P_4$, SIMC PI\n$M_s=%.2f$" % true_Ms(K, L, lags, Ki, Kp, 0), color=INK, fontsize=6.8)

Kp = P["slow_Kp"]; Ki = Kp / 2.4
_, e, dt = read_Ms(1.0, 1.0, [0.3], Ki, Kp, 0, False, tend=150)
w2 = np.logspace(-2, math.log10(3.2), 400)
a.loglog(w2, abs(1 / (1 + loop_fr(1.0, 1.0, [0.3], Ki, Kp, 0, w2))), color=ORANGE, lw=4, alpha=0.3, solid_capstyle="butt")
a.loglog(w2, abs(S_rec(e, dt, w2)), color=ORANGE, lw=1.0, ls=(0, (4, 2)))
a.text(0.35, 0.05, "slow integral:\nno crossing,\n$M_s=%.2f$" % true_Ms(1.0, 1.0, [0.3], Ki, Kp, 0),
       color=INK, fontsize=6.8)
a.plot([], [], color=INK2, lw=4, alpha=0.3, label="true")
a.plot([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label="from one step")
a.legend(fontsize=6.8, frameon=False, loc="upper left", handlelength=2.2)
a.set_xlim(1e-3, 4); a.set_ylim(2e-2, 5)
a.set_xlabel("frequency $\\omega$ (rad/s)"); a.set_ylabel("$|S(j\\omega)|$", labelpad=1)
a.set_title("(a) the step gives the whole sensitivity function", fontsize=8, color=INK)

# (b) --------------------------------------------------------------------------
b.axhspan(-0.9, 0.9, color=RULE, alpha=0.25, lw=0)
b.axhline(0, color=INK2, lw=0.6)
groups = [("other", INK, "s", 3.6, "SIMC PI, slow integral,\nlag-cancelled"),
          ("original", ORANGE, "o", 4.6, "original SPIN, accepted"),
          ("certified", BLUE, "o", 3.6, "certified SPIN, accepted")]
for key, col, mk, ms, lab in groups:
    x = np.array([p[1] for p in P[key]]); y = 100 * (np.array([p[2] for p in P[key]]) / x - 1)
    b.plot(x, y, mk, ms=ms, color=col, mec="white", mew=0.6, label=lab, ls="none")
    print(key, "n", len(x), "error range %", round(y.min(), 2), round(y.max(), 2))
b.text(2.68, 0.95, "$\\pm0.9\\%$", fontsize=6.5, color=INK2, va="bottom", ha="right")
b.set_xlabel("true $M_s$"); b.set_ylabel("reading error (%)")
b.set_ylim(-1.6, 1.6)
b.set_title("(b) reading error of $M_s$, every loop", fontsize=8, color=INK)
b.legend(fontsize=6.3, frameon=False, loc="lower right", handletextpad=0.2, labelspacing=0.3)

fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.89)
fig.savefig("fig_record.pdf"); fig.savefig("fig_record.png", dpi=200)
