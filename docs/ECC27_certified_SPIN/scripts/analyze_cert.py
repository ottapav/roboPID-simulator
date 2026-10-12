"""Table II and the numbers of Section V from thm1_points.json (all accepted tunings, exact digital M_s)."""
import json, math, sys
import numpy as np

def iae_d(K, L, lags, Ki, Kp, Kd, dt, Ta):
    """unit load step at the plant input, same digital loop as the record (spin_cert_run.sim)"""
    n = int(40 * Ta / dt); nd = int(round(L / dt)); a = [math.exp(-dt / t) for t in lags]; x = [0.0] * len(lags)
    Tf = Kd / Kp / 10 if Kd > 0 else 1.0; af = math.exp(-dt / Tf) if Kd > 0 else 0.0
    ub = np.zeros(n + nd + 1); I = 0.0; xf = 0.0; y = 0.0; iae = 0.0
    for i in range(n):
        ek = -y; D = (Kd / Tf) * (ek - xf) if Kd > 0 else 0.0
        if Kd > 0: xf = af * xf + (1 - af) * ek
        u = Kp * ek + Ki * I + D; I += ek * dt
        ub[i + nd] = K * (u + 1.0); v = ub[i]
        for j, aj in enumerate(a): x[j] = aj * x[j] + (1 - aj) * v; v = x[j]
        y = v; iae += abs(y) * dt
    return iae

P = json.load(open("thm1_points.json"))
def worst(run, plant):
    c = [p for p in P if p["run"] == run and p["plant"] == plant]
    return max(c, key=lambda p: p["Ms"]) if c else None
plants = sorted({p["plant"] for p in P})
own = json.load(open("owner_box.json"))
S = {}
for pl in plants:
    row = {}
    for run in ("box", "box2", "cbox", "cbox2", "designed", "cnob"):
        w = worst(run, pl); row[run] = None if w is None else w["Ms"]
        if w is not None and run in ("box", "cbox", "box2", "cbox2"):
            Ta = w["L"] + sum(w["lags"]); row[run + "_iae"] = iae_d(w["K"], w["L"], w["lags"], *w["gains"], w["dt"], Ta)
    row["owner"] = own.get(pl, ["?"])[0]
    S[pl] = row
json.dump(S, open("summary_cert.json", "w"), indent=1)
M = 1.7
for run in ("box", "box2", "cbox", "cbox2", "designed", "cnob"):
    v = [S[p][run] for p in plants if S[p][run] is not None]
    above = [p for p in plants if S[p][run] is not None and S[p][run] > M]
    if not v: continue
    print(f"{run:8s} loops {len(v):2d} max {max(v):.3f} above {len(above)}: {above}")
for s, c in (("box", "cbox"), ("box2", "cbox2")):
    d = [100 * (S[p][c + "_iae"] / S[p][s + "_iae"] - 1) for p in plants if S[p].get(c + "_iae") and S[p].get(s + "_iae")]
    print(f"IAE {c} vs {s}: mean {np.mean(d):+.1f}%  min {min(d):+.1f}%  max {max(d):+.1f}%")
for p in plants:
    r = S[p]; f = lambda x: "--" if x is None else f"{x:.3f}"
    d = 100 * (r["cbox_iae"] / r["box_iae"] - 1) if r.get("cbox_iae") else None
    print(f"{p:22s} pub {f(r['box'])} {f(r['box2'])} cert {f(r['cbox'])} {f(r['cbox2'])} | nobox pub {f(r['designed'])} cert {f(r['cnob'])} | {r['owner']} dIAE {d if d is None else round(d,1)}")
