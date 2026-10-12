"""Theorem 1 as the record grows: lower and upper bound against record length, tau = true tail."""
import json, glob, math, numpy as np
from spin_cert_run import sim
from cert import certificate, increments, Ms_d
cases = [("P3 PID", "box"), ("P4 PID", "box"), ("lag-dominant PI", "box")]
out = {}
for plant, run in cases:
    r = [json.loads(l) for f in glob.glob(f"{run}_*.jsonl") for l in open(f) if json.loads(l)["plant"] == plant][0]
    K, L, lags, pid = r["K"], r["L"], r["lags"], r["pid"]; Ta = L + sum(lags)
    G = r["res"]["worst"]["gains"]; dt = min(15 * Ta / 1500, (G[2] / G[1] / 10) / 20) if pid else Ta / 1000
    nT = int(Ta / dt); eL = sim(K, L, lags, *G, dt, 30 * nT); a = increments(eL)
    Ms = Ms_d(K, L, lags, *G, dt)[0]; rows = []
    for T in np.geomspace(0.3, 15, 24):
        n = int(T * nT); tail = float(np.sum(abs(a[n:])))
        c = certificate(eL[:n], tail)
        rows.append((float(T), c["lower"], c["upper"], tail))
        print(plant, T, round(c["lower"], 4), round(Ms, 4), round(c["upper"], 4), "tail %.1e" % tail, flush=True)
    out[plant] = dict(Ms=Ms, rows=rows)
json.dump(out, open("bounds_vs_length.json", "w"), indent=1)
