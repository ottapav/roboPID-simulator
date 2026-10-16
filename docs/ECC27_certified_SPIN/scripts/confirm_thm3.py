"""
Theorem 3 (certify before applying) on random moves. From each tuning that safe SPIN accepted
(safe1_*.jsonl, cautious start), apply 12 random gain multipliers in [1/2, 2]^3 and 12 in [1, 8]^3 (PI: K_d = 0) and check:
whenever (4) holds, the new loop is stable and its exact M_s lies below the bound (5).
Writes thm3_points.json.
"""
import glob, json, math
import numpy as np
from spin_cert_run import sim
from cert import Ms_d
from cert_ahead import ahead

TAU, MSTAR = 1e-3, 1.7
rng = np.random.default_rng(1)
pts = []
for f in sorted(glob.glob("safe1_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r["rule"] != "cert-ahead" or not r["res"].get("accepted"): continue
        K, L, lags, pid = r["K"], r["L"], r["lags"], r["pid"]; Ta = L + sum(lags)
        c = r["res"]["accepted"][0]; G = np.array(c["gains"]); dt = c["dt"]
        n = int(15 * Ta / dt); e = sim(K, L, lags, *G, dt, n)
        for k_ in range(24):
            lo, hi = (0.5, 2) if k_ < 12 else (1, 8)
            m = np.exp(rng.uniform(math.log(lo), math.log(hi), 3)); m[2] = m[2] if pid else 1.0
            G2 = G * m
            A = ahead(e, dt, tuple(G), tuple(G2), TAU, MSTAR)
            e2 = sim(K, L, lags, *G2, dt, 12 * n)                  # stable iff the error keeps decaying
            A1, A2 = abs(e2[4 * n:6 * n]).max(), abs(e2[10 * n:]).max()
            stable = bool(A2 < 1e-10 or A2 < 0.5 * A1)
            Ms = Ms_d(K, L, lags, *G2, dt)[0] if stable else math.inf
            pts.append(dict(plant=r["plant"], mult=m.tolist(), cert_stable=A["stable"], U=A["U"],
                            stable=stable, Ms=Ms))
ok = [p for p in pts if p["cert_stable"]]
viol = [p for p in ok if not p["stable"] or p["Ms"] > p["U"] + 1e-9]
gap = [p["U"] - p["Ms"] for p in ok if math.isfinite(p["U"])]
print(f"{len(pts)} moves, {len(ok)} certified stable, violations {len(viol)}, "
      f"truly unstable among uncertified {sum(not p['stable'] for p in pts if not p['cert_stable'])}, "
      f"bound - true: median {np.median(gap):.4f}, max {max(gap):.4f}, "
      f"certified <= M*: {sum(p['U'] <= MSTAR for p in ok)}, true <= M*: {sum(p['Ms'] <= MSTAR for p in pts)}")
json.dump(pts, open("thm3_points.json", "w"), default=lambda x: None)
