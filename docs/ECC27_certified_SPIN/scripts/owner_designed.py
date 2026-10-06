"""Band that owns the sensitivity peak of each original tuning (largest elasticity of |C|).
Reads designed_*.jsonl, writes owner.json (used by fig_certified.py and Table 3)."""
import json, glob, os, numpy as np
from spin_cert_run import elasticities
from designed import loop_fr
PRE = os.environ.get("OWN_IN", "designed")          # e.g. OWN_IN=box for SPIN with its gain box
R = [json.loads(l) for f in glob.glob(f"{PRE}_*.jsonl") for l in open(f)]
w = np.logspace(-4, 2.5, 80000); names = ["G0", "G1", "G2"]; own = {}
for r in R:
    if r["rule"] != "original" or r["res"] is None: continue
    Ki, Kp, Kd = r["res"]["worst"]["gains"]
    S = abs(1 / (1 + loop_fr(r["K"], r["L"], r["lags"], Ki, Kp, Kd, w))); i = int(np.argmax(S))
    sg = elasticities(Ki, Kp, Kd, w[i]); own[r["plant"]] = [names[int(np.argmax(sg))], [float(x) for x in sg]]
    print(f"{r['plant']:24} peak {S[i]:.3f} at w={w[i]:.3f}  owner {own[r['plant']][0]}  sigma {np.round(sg, 2)}")
json.dump(own, open("owner.json" if PRE == "designed" else f"owner_{PRE}.json", "w"), indent=1)
