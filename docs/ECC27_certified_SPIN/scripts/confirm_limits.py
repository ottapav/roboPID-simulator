"""
Robust turn limits at delta = 0.02 (remark after Proposition 2).
  1. lag-cancelled loop L = k e^{-s}/s: with n setpoint crossings above delta, every count lies in
     (n/2, (n+1)/2) for Gamma_0, Gamma_1, Gamma_2; the steps sit where the (n+1)-th extremum equals delta;
     where each candidate limit trips
  2. general loops (the test plants of Section V, random PI/PID gains): how often a count with m sign
     changes of its first coordinate leaves (m/2, (m+1)/2)
Writes limits.json.
"""
import json, math
import numpy as np
import spin_run as S
from lagcancel import records, Ms

DT, D = 0.01, 0.02

# 1. lag-cancelled loop ---------------------------------------------------------------------
ks = np.linspace(0.2, 1.0, 801); ms = np.array([Ms(k) for k in ks])
N = np.zeros((len(ks), 3)); n = np.zeros(len(ks), int); E = np.zeros((len(ks), 4))
for c0 in range(0, len(ks), 50):
    kk = ks[c0:c0 + 50]; t, R = records(kk, tend=60.0); st = int(round(DT / (t[1] - t[0])))
    for j in range(len(kk)):
        e = R[::st, j]; N[c0 + j] = S.counts(e, DT, D)
        kd = np.where(abs(e) > D)[0][-1] + 1; s = np.sign(e[:kd]); s = s[s != 0]
        n[c0 + j] = int(np.sum(s[1:] != s[:-1]))
        sg = np.sign(e); cr = np.where(sg[1:] != sg[:-1])[0]
        E[c0 + j] = [abs(e[cr[i]:cr[i + 1]]).max() if i + 1 < len(cr) else 0.0 for i in range(4)]
plateaus = []
print("1. lag-cancelled loop, delta = 0.02")
for m in range(5):
    sel = n == m
    if not sel.any(): continue
    lo, hi = N[sel].min(axis=0), N[sel].max(axis=0)
    inside = bool(np.all(lo > m / 2) and np.all(hi < (m + 1) / 2))
    plateaus.append({"n": m, "Ms": [float(ms[sel].min()), float(ms[sel].max())], "lo": lo.tolist(), "hi": hi.tolist(), "inside": inside})
    print(f"  n={m}: M_s {ms[sel].min():.3f}..{ms[sel].max():.3f}  " +
          "  ".join(f"N{b} {lo[b]:.2f}..{hi[b]:.2f}" for b in range(3)) + f"  in (n/2,(n+1)/2): {inside}")
steps = [float(ms[np.argmax(E[:, i] > D)]) for i in range(4)]
print("  steps (|E_{n+1}| = delta) at M_s =", [round(x, 3) for x in steps])
trips = {str(L): [float(ms[np.argmax(N[:, b] > L)]) for b in range(3)] for L in (0.5, 0.75, 1.0, 1.5)}
for L, v in trips.items(): print(f"  limit {L}: trips at M_s =", [round(x, 3) for x in v])

# 2. general loops --------------------------------------------------------------------------
rng = np.random.default_rng(1)
PL = dict(S.PLANTS)
PL.update({"delay": (1.0, 5.0, [1]), "lag": (1.0, 1.0, [10]), "2lag": (1.0, 0.5, [10, 1]), "fast": (1.0, 0.1, [1, .5]),
           "fam1": (1.0, 1.0, [1, .5, .25]), "fam4": (1.0, 4.0, [1, .5, .25]), "fam16": (1.0, 16.0, [1, .5, .25])})


def counts_m(e, dt, delta=D, eps=0.1):
    big = np.where(abs(e) > delta * abs(e).max())[0]; kd = min(len(e), big[-1] + 2)
    I = np.cumsum(e) * dt; de = np.diff(e, prepend=e[0]); d2 = np.diff(de, prepend=de[0])
    out = []
    for p, q in [(e[:kd], (I - I[-1])[:kd]), (de[:kd], e[:kd]), (d2[:kd], de[:kd])]:
        Y = S.whiten(p, q)
        if Y is None: out.append(None); continue
        k0 = np.where(np.hypot(*Y) > eps)[0][0]
        s = np.sign(p[k0:]); s = s[s != 0]
        out.append((S.wind(Y[0][k0:], Y[1][k0:]), int(np.sum(s[1:] != s[:-1]))))
    return out


tot, out = np.zeros(3, int), np.zeros(3, int)
for name, (K, L, lags) in PL.items():
    l = sorted(lags, reverse=True) + [0.0]; tau = l[0] + l[1] / 2; th = L + l[1] / 2 + sum(l[2:]); Ta = L + sum(lags)
    for pid in (False, True):
        if pid and len(lags) < 2: continue
        Kp0 = tau / (K * 2 * th); Ki0 = Kp0 / min(tau, 8 * th); Kd0 = Kp0 * l[1] * 0.5 if pid else 0.0
        for trial in range(60):
            f = np.exp(rng.uniform(-1.5, 1.2, 3)) if trial >= 10 else np.full(3, np.exp(np.linspace(-1.5, 1.0, 10)[trial]))
            Ki, Kp, Kd = Ki0 * f[0], Kp0 * f[1], Kd0 * f[2]
            dt = min(Ta / 300, (Kd / max(Kp, 1e-9) / 10) / 20) if pid else Ta / 300
            nn = int(25 * Ta / dt); e = S.sim(K, L, lags, Ki, Kp, Kd, dt, nn)
            if abs(e[int(0.8 * nn):]).max() > 0.01 * abs(e).max(): continue
            for b, r in enumerate(counts_m(e, dt)):
                if r is None or (b == 2 and not pid): continue
                tot[b] += 1; out[b] += not (r[1] / 2 < r[0] < (r[1] + 1) / 2)
frac = (out / tot).tolist()
print(f"2. general loops: records {tot.tolist()}, outside (m/2,(m+1)/2): " + ", ".join(f"{100 * x:.0f}%" for x in frac))
json.dump({"plateaus": plateaus, "steps": steps, "trips": trips, "general": {"records": tot.tolist(), "outside": frac}},
          open("limits.json", "w"), indent=1)
