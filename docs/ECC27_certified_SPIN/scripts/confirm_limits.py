"""
Robust turn limits at delta = 0.02, with SPIN's own counts (Definition 1: per-axis peak normalization).
  1. lag-cancelled loop L = k e^{-s}/s: with n setpoint crossings above delta, the range of each
     count; the steps sit where the (n+1)-th extremum equals delta; where each limit trips
  2. general loops: replay certified SPIN (published limits, gain box, cautious start) on the 19 test
     loops of Section IV and record every count at every decision; for each band, how often the count
     falls in the open gap (n/2 + 1/4, (n+1)/2) above a half-integer
Writes limits.json.
"""
import json, math, os
import numpy as np
from compare import counts_old
from lagcancel import records, Ms

DT, D = 0.01, 0.02

# 1. lag-cancelled loop ---------------------------------------------------------------------
ks = np.linspace(0.2, 1.0, 801); ms = np.array([Ms(k) for k in ks])
N = np.zeros((len(ks), 3)); n = np.zeros(len(ks), int); E = np.zeros((len(ks), 4))
for c0 in range(0, len(ks), 50):
    kk = ks[c0:c0 + 50]; t, R = records(kk, tend=60.0); st = int(round(DT / (t[1] - t[0])))
    for j in range(len(kk)):
        e = R[::st, j]; N[c0 + j] = counts_old(e, DT, D)
        kd = np.where(abs(e) > D)[0][-1] + 1; s = np.sign(e[:kd]); s = s[s != 0]
        n[c0 + j] = int(np.sum(s[1:] != s[:-1]))
        sg = np.sign(e); cr = np.where(sg[1:] != sg[:-1])[0]
        E[c0 + j] = [abs(e[cr[i]:cr[i + 1]]).max() if i + 1 < len(cr) else 0.0 for i in range(4)]
plateaus = []
print("1. lag-cancelled loop, delta = 0.02, SPIN counts")
for m in range(5):
    sel = n == m
    if not sel.any(): continue
    lo, hi = N[sel].min(axis=0), N[sel].max(axis=0)
    plateaus.append({"n": m, "Ms": [float(ms[sel].min()), float(ms[sel].max())], "lo": lo.tolist(), "hi": hi.tolist()})
    print(f"  n={m}: M_s {ms[sel].min():.3f}..{ms[sel].max():.3f}  " +
          "  ".join(f"N{b} {lo[b]:+.2f}..{hi[b]:+.2f}" for b in range(3)))
steps = [float(ms[np.argmax(E[:, i] > D)]) for i in range(4)]
print("  steps (|E_{n+1}| = delta) at M_s =", [round(x, 3) for x in steps])
trips = {str(L): [float(ms[np.argmax(N[:, b] > L)]) for b in range(3)] for L in (0.5, 0.75, 1.0, 1.25, 1.5)}
for L, v in trips.items(): print(f"  limit {L}: trips at M_s =", [round(x, 3) for x in v])

# 2. general loops: every count decision of certified SPIN on the 19 test loops ---------------
os.environ["SPIN_BOX"] = "1"
import designed as d
seen = []
counter, lim, Mstar = d.RULES["cert-shape"]
d.RULES["cert-shape"] = (lambda e, dt: seen.append(counts_old(e, dt)) or seen[-1], lim, Mstar)
for name, K, L, lags, pid in d.PLANTS:
    d.run(K, L, lags, pid, "cert-shape")
V = np.array(seen); frac = V - np.floor(2 * V) / 2                 # position above the half-integer below
gap = [float(np.mean((frac[:, b] > 0.25 + 1e-9) & (frac[:, b] < 0.5 - 1e-9))) for b in range(3)]
in_075_1 = [int(np.sum((V[:, b] > 0.75) & (V[:, b] <= 1.0))) for b in range(3)]
print(f"2. {len(V)} count decisions on the 19 test loops: share in (n/2+1/4, (n+1)/2): " +
      ", ".join(f"N{b} {100 * g:.1f}%" for b, g in enumerate(gap)) + f";  values in (0.75, 1]: {in_075_1}")
json.dump({"plateaus": plateaus, "steps": steps, "trips": trips,
           "general": {"decisions": len(V), "gap_share": gap, "in_075_1": in_075_1}},
          open("limits.json", "w"), indent=1)
