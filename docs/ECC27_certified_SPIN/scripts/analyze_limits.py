"""
Certified SPIN with three sets of count limits (gain box on, cautious and bold start):
    published (0.5, 0.75, 1.0): cbox_0.jsonl, cbox2_0.jsonl
    (0.5, 1.0, 1.0):            lim101_0.jsonl, lim101b_0.jsonl
    (0.5, 0.5, 0.5):            lim050_0.jsonl, lim050b_0.jsonl + lim050b_rest.jsonl (run_rest.py)
For each loop: the most aggressive accepted tuning (exact digital M_s), its load-step IAE (same
digital loop as analyze_cert.py), and which test rejected in the last 60 iterations
(G<k>: the count of band k; shape<k>/certU: the certificate). Writes limits_runs.json.
"""
import json, math
import numpy as np


def iae_d(K, L, lags, Ki, Kp, Kd, dt, Ta):
    """unit load step at the plant input, same digital loop as the record (as in analyze_cert.py)"""
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


SETS = {"0.5,0.75,1.0": ("cbox_0.jsonl", "cbox2_0.jsonl"),
        "0.5,1.0,1.0": ("lim101_0.jsonl", "lim101b_0.jsonl"),
        "0.5,0.5,0.5": ("lim050_0.jsonl", ("lim050b_0.jsonl", "lim050b_rest.jsonl"))}
M = 1.7


def load(fns):
    out = {}
    lines = [l for fn in ((fns,) if isinstance(fns, str) else fns) for l in open(fn)]
    for line in lines:
        r = json.loads(line)
        if r["rule"] != "cert-shape" or r["res"] is None: continue
        w = max(r["res"]["accepted"], key=lambda c: c["Ms_d"])
        Ta = r["L"] + sum(r["lags"])
        f = r["res"]["fired"]
        out[r["plant"]] = dict(Ms=w["Ms_d"], iae=iae_d(r["K"], r["L"], r["lags"], *w["gains"], w["dt"], Ta),
                               count=[f.get(f"G{k}", 0) for k in range(3)],
                               cert=sum(v for k, v in f.items() if k.startswith("shape") or k == "certU"),
                               acc=f.get("none", 0))
    return out


R = {s: [load(a), load(b)] for s, (a, b) in SETS.items()}
base = "0.5,0.75,1.0"
summary = {}
for s in SETS:
    row = {}
    for i, start in enumerate(("cautious", "bold")):
        d = R[s][i]; b = R[base][i]; pl = sorted(set(d) & set(b))
        ms = [d[p]["Ms"] for p in pl]
        diae = [100 * (d[p]["iae"] / b[p]["iae"] - 1) for p in pl]
        cnt = np.sum([d[p]["count"] for p in pl], axis=0); cert = sum(d[p]["cert"] for p in pl)
        acc = sum(d[p]["acc"] for p in pl)
        none = len(set(b) - set(d))                       # loops with no tuning accepted in the last 60
        row[start] = dict(loops=len(pl), no_tuning=none, Ms_max=max(ms), Ms_min=min(ms), above=sum(m > M for m in ms),
                          iae_mean=float(np.mean(diae)), iae_median=float(np.median(diae)),
                          iae_min=float(min(diae)), iae_max=float(max(diae)),
                          rej_count=cnt.tolist(), rej_cert=int(cert), acc=int(acc))
        print(f"{s:13s} {start:8s} loops {len(pl):2d} (no tuning {none})  M_s {min(ms):.3f}..{max(ms):.3f}  above M* {row[start]['above']}"
              f"  IAE vs published: median {np.median(diae):+5.1f}%, mean {np.mean(diae):+5.1f}% ({min(diae):+.0f}..{max(diae):+.0f}%)"
              f"  last-60 rejections: counts G0/G1/G2 {cnt.tolist()}, certificate {cert}, accepts {acc}")
    summary[s] = row
json.dump(summary, open("limits_runs.json", "w"), indent=1)
