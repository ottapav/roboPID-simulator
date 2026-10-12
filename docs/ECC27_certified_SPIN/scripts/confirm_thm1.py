"""
Numerical confirmation of Lemma 2 and Theorem 1 on every tuning accepted in the SPIN runs.
For each accepted tuning: the record used by SPIN (15 apparent time constants), a record four
times longer to measure the true tail variation tau_true, the certificate with the design value
tau = 1e-3, and the exact discrete M_s of the loop that ran (cert.Ms_d).
Checks:  (L2) |S_d - S_N| <= tau_true on the certificate grid,
         (A1) tau_true <= tau,   (T1) lower <= M_s <= upper.
Output: thm1_points.json
"""
import json, glob, math, sys
import numpy as np
from spin_cert_run import sim
from cert import certificate, increments, S_grid, L_d, Ms_d

TAU = 1e-3
GROUPS = [("box", "original", "published SPIN"), ("box2", "original", "published SPIN"),
          ("designed", "original", "published SPIN"),
          ("cbox", "cert-shape", "certified SPIN"), ("cbox2", "cert-shape", "certified SPIN"),
          ("cnob", "cert-shape", "certified SPIN")]
import os
ONLY = os.environ.get("RUNS"); OUT = os.environ.get("OUT", "thm1_points.json")
out = []; seen = set()
for pre, rule, lab in GROUPS:
    if ONLY and pre not in ONLY.split(","): continue
    for f in sorted(glob.glob(f"{pre}_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["rule"] != rule or r["res"] is None: continue
            K, L, lags, pid = r["K"], r["L"], r["lags"], r["pid"]; Ta = L + sum(lags)
            for c in r["res"]["accepted"]:
                Ki, Kp, Kd = c["gains"]; key = (pre, r["plant"], round(Ki, 10), round(Kp, 10), round(Kd, 10))
                if key in seen: continue
                seen.add(key)
                dt = min(15 * Ta / 1500, (Kd / Kp / 10) / 20) if pid else Ta / 1000
                n = int(15 * Ta / dt); eL = sim(K, L, lags, Ki, Kp, Kd, dt, 4 * n); e = eL[:n]
                a = increments(eL); tail = float(np.sum(abs(a[n:])))
                ce = certificate(e, TAU)
                S, h = S_grid(a[:n], ce["th1"], ce["M"]); th = np.arange(ce["M"]) * h; th[0] = 1e-12
                Sd = 1 / (1 + L_d(K, L, lags, Ki, Kp, Kd, dt, th))
                idm = float(np.max(abs(Sd[1:] - S[1:])))
                Ms, thp = Ms_d(K, L, lags, Ki, Kp, Kd, dt)
                row = dict(group=lab, run=pre, plant=r["plant"], gains=[Ki, Kp, Kd], dt=dt, K=K, L=L, lags=lags, Ms=Ms, Ms_cont=c["Ms"], lower=ce["lower"],
                           upper=ce["upper"], read=ce["read"], tail=tail, id_err=idm, grid=ce["grid_term"],
                           hf=ce["up_hf"], N=ce["N"], M=ce["M"])
                out.append(row)
                ok = row["lower"] <= Ms <= row["upper"] and tail <= TAU and idm <= tail + 1e-12
                print(pre, r["plant"], f"Ms {Ms:.4f} in [{row['lower']:.4f}, {row['upper']:.4f}] tail {tail:.1e} "
                      f"id {idm:.1e}", "OK" if ok else "FAIL", flush=True)
json.dump(out, open(OUT, "w"), indent=0)
ok = [p["lower"] <= p["Ms"] <= p["upper"] for p in out]
print("records", len(out), "bracketed", sum(ok), "tail max", max(p["tail"] for p in out),
      "identity err max", max(p["id_err"] for p in out),
      "width max", max(p["upper"] - p["lower"] for p in out),
      "upper - Ms in", min(p["upper"] - p["Ms"] for p in out), max(p["upper"] - p["Ms"] for p in out))
