"""
Confirms Proposition 2 (counts at small delta) on the lag-cancelled loop L = k e^{-s}/s (L_d = 1).
  1. the rightmost root of s e^{s} = -k is W_0(-k): its real part exceeds that of every other branch
  2. k > 1/e: N_j(delta) = rate * ln(1/delta) + O(1), rate = omega / (2 pi |sigma|), for j = 0, 1, 2
     (least-squares slope over delta in [1e-12, 1e-3], and the range of the remainder)
  3. k <= 1/e: N_j(delta) does not change with delta
  4. the rate against M_s on a fine grid (zero up to M_s = 1.394, then increasing)
Writes counts_delta.json (used by fig_stair.py).
"""
import json, math
import numpy as np
from scipy.special import lambertw
import spin_run as S
from lagcancel import records, Ms

DT = 0.01
LOGD = np.linspace(math.log(1e3), math.log(1e12), 37)          # ln(1/delta)


def rate(k):
    s = lambertw(-k, 0)
    return float(s.imag / (2 * math.pi * abs(s.real))) if abs(s.imag) > 1e-12 else 0.0


# 1. rightmost root ------------------------------------------------------------------------
worst = 0.0
for k in np.linspace(0.05, 1.55, 61):
    re = [lambertw(-k, j).real for j in range(-20, 21)]
    worst = max(worst, max(r for j, r in zip(range(-20, 21), re) if j != 0) - re[20])
print(f"1. max over k of (Re W_j - Re W_0), j != 0: {worst:.3e}  (negative: W_0 is rightmost)")

# 2.-3. counts against delta ---------------------------------------------------------------
ks = [0.2, 0.3, 1 / math.e, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
t, E = records(ks, tend=300.0)
st = int(round(DT / (t[1] - t[0])))
rows = []
for j, k in enumerate(ks):
    e = E[::st, j]
    N = np.array([S.counts(e, DT, math.exp(-x)) for x in LOGD])        # shape (len(LOGD), 3)
    r = rate(k)
    out = {"k": k, "Ms": Ms(k), "rate": r}
    if k <= 1 / math.e + 1e-12:
        out["spread"] = float(np.ptp(N, axis=0).max())
        print(f"3. k={k:.3f} Ms={Ms(k):.3f}: N_j over delta in [1e-12,1e-3] varies by {out['spread']:.1e}")
    else:
        sl = [float(np.polyfit(LOGD, N[:, i], 1)[0]) for i in range(3)]
        rem = [float(np.ptp(N[:, i] - r * LOGD)) for i in range(3)]
        out.update(slope=sl, rem=rem)
        print(f"2. k={k:.2f} Ms={Ms(k):.3f} rate={r:.4f} slopes=" + ", ".join(f"{x:.4f}" for x in sl)
              + "  rel.err=" + ", ".join(f"{(x - r) / r:+.3f}" for x in sl)
              + "  remainder range=" + ", ".join(f"{x:.2f}" for x in rem))
    rows.append(out)

# 4. rate against M_s ------------------------------------------------------------------------
kf = np.linspace(0.2, 1.3, 600)
rf = np.array([rate(k) for k in kf]); mf = np.array([Ms(k) for k in kf])
inc = bool(np.all(np.diff(rf[kf > 1 / math.e]) > 0)) and bool(np.all(np.diff(mf) > 0))
print(f"4. rate zero for M_s <= {Ms(1 / math.e):.3f}; increasing in M_s beyond: {inc}")
json.dump({"rows": rows, "kf": kf.tolist(), "Ms": mf.tolist(), "rate": rf.tolist()},
          open("counts_delta.json", "w"))
