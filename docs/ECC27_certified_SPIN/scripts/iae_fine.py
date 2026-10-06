"""Load-step IAE (unit step at the plant input) of the tunings in designed_*.jsonl,
with the controller sampled as in designed.py (PID: Tf/20, PI: Ta/1000)."""
import json, glob, math
import numpy as np

def load_iae(K, L, lags, Ki, Kp, Kd):
    Ta = L + sum(lags); Tf = Kd / Kp / 10 if Kd > 0 else 1e9
    dt = min(Ta / 1000, Tf / 20); n = int(40 * Ta / dt)
    nd = int(round(L / dt)); a = [math.exp(-dt / t) for t in lags]; x = [0.0] * len(lags)
    af = math.exp(-dt / Tf) if Kd > 0 else 0.0; ub = np.zeros(n + nd + 1); I = xf = y = iae = 0.0
    for i in range(n):
        ek = -y; D = (Kd / Tf) * (ek - xf) if Kd > 0 else 0.0
        if Kd > 0: xf = af * xf + (1 - af) * ek
        u = Kp * ek + Ki * I + D; I += ek * dt
        ub[i + nd] = K * (u + 1.0); v = ub[i]
        for j, aj in enumerate(a): x[j] = aj * x[j] + (1 - aj) * v; v = x[j]
        y = v; iae += abs(y) * dt
    return iae

import os
# default: the unboxed runs of designed.py; IAE_PAIR="box:boxtest" compares SPIN with its gain box
# against SPIN with its box plus the acceptance test
po, pc = os.environ.get("IAE_PAIR", "designed:designed").split(":")
O = {json.loads(l)["plant"]: json.loads(l) for f in glob.glob(f"{po}_*.jsonl") for l in open(f)
     if json.loads(l)["rule"] == "original"}
C = {json.loads(l)["plant"]: json.loads(l) for f in glob.glob(f"{pc}_*.jsonl") for l in open(f)
     if json.loads(l)["rule"] in ("certified", "original+test")}
out = {}
for p in sorted(O):
    o, c = O[p], C.get(p)
    if c is None or c["res"] is None or o["res"] is None: continue
    if o["res"] is None: continue
    io = load_iae(o["K"], o["L"], o["lags"], *o["res"]["worst"]["gains"])
    ic = load_iae(c["K"], c["L"], c["lags"], *c["res"]["worst"]["gains"])
    out[p] = [io, ic, 100 * (ic / io - 1)]
    print(f"{p:24} IAE original {io:9.4f} certified {ic:9.4f}  {out[p][2]:+6.1f}%", flush=True)
json.dump(out, open("iae_fine.json" if po == "designed" else f"iae_{po}.json", "w"), indent=1)
