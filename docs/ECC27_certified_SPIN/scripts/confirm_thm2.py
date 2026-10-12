"""
Numerical confirmation of Theorem 2(ii) and Proposition 3(a),(d).
(ii)  For gains lambda*K: the loop is stable and M_s <= 1/(1 - lambda |m|), m = min(0, min Re L), when lambda|m| < 1.
(a)   sum_k g_k = Re S(th_p) - 1 at the peak.
(d)   the M_s predicted from the record after a SPIN band move equals the true one up to the stated
      error formula |C'/C| |S' S'_hat| / |S S_hat| |eps|.
Starts from the most aggressive tuning that published SPIN accepted (box_*.jsonl).
"""
import json, glob, math
import numpy as np
from spin_cert_run import sim
from cert import L_d, Ms_d, minReL, C_d, certificate, increments, S_grid

beta = 0.1; g = 1 / (1 - beta); res = []
R = {json.loads(l)["plant"]: json.loads(l) for f in glob.glob("box_*.jsonl") for l in open(f)
     if json.loads(l)["rule"] == "original"}
for name, r in sorted(R.items()):
    if r["res"] is None: continue
    K, L, lags, pid = r["K"], r["L"], r["lags"], r["pid"]; Ta = L + sum(lags)
    G0 = np.array(r["res"]["worst"]["gains"])
    dt = min(15 * Ta / 1500, (G0[2] / G0[1] / 10) / 20) if pid else Ta / 1000
    m = min(0.0, minReL(K, L, lags, *G0, dt)); ok2 = True; js = []
    for j in range(0, 80):
        lam = (1 - beta) ** j
        if lam * abs(m) >= 1: continue
        Ms, _ = Ms_d(K, L, lags, *(lam * G0), dt, n=60000); bound = 1 / (1 - lam * abs(m))
        e = sim(K, L, lags, *(lam * G0), dt, int(30 * Ta / dt))
        stable = abs(e[-len(e) // 10:]).max() < 0.5 * abs(e[: len(e) // 2]).max()
        ok2 &= (Ms <= bound + 1e-9) and stable; js.append((j, Ms, bound))
        if lam < 0.3: break
    # Prop. 3(a) and (d) at the original tuning
    n = int(15 * Ta / dt); e = sim(K, L, lags, *G0, dt, n); a = increments(e)
    c = certificate(e, 1e-3); S, h = S_grid(a, c["th1"], c["M"])
    th = np.arange(c["M"]) * h; th[0] = h * 1e-3; S[0] = S_grid(a, h * 1e-3, 2)[0][1]
    z = np.exp(1j * th); C = C_d(*G0, dt, z); Lr = 1 / S - 1; ip = int(np.argmax(abs(S)))
    gk = []
    for k in range(3 if pid else 2):
        Gp = G0.copy(); Gp[k] *= math.exp(1e-7)
        ck = (C_d(*Gp, dt, z[ip]) - C[ip]) / 1e-7; gk.append(float((-(1 - S[ip]) * ck / C[ip]).real))
    sum_err = abs(sum(gk) - (S[ip].real - 1))
    pred_err = []
    for k in range(3 if pid else 2):
        F = np.ones(3); F[:k] *= g; F[k] /= g; G1 = G0 * F
        pred = float(np.max(abs(1 / (1 + Lr * C_d(*G1, dt, z) / C))))
        true = float(np.max(abs(1 / (1 + L_d(K, L, lags, *G1, dt, th)))))
        pred_err.append(pred - true)
    res.append(dict(plant=name, m=m, steps=len(js), Ms0=js[0][1] if js else None, ok=bool(ok2), g=gk,
                    sum_err=float(sum_err), pred_err=pred_err))
    print(name, f"m={m:.3f} back-off steps {len(js)} bound ok {ok2}  g={np.round(gk, 3)} "
          f"|sum g - (ReS-1)|={sum_err:.1e}  pred-true={np.round(pred_err, 5)}", flush=True)
json.dump(res, open("thm2_points.json", "w"), indent=1)
print("all back-off bounds hold:", all(r["ok"] for r in res),
      " max |pred-true|:", max(max(abs(x) for x in r["pred_err"]) for r in res))
