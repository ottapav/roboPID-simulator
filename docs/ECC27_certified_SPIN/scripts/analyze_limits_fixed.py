"""
Limit values under the certificate (Sec. 5), fixed T_s: certified SPIN with every limit 1/2
(limf1_*.jsonl cautious, limf2_*.jsonl bold; run_limits_fixed.sh) against certified SPIN with the
published limits (safe1/safe2 runs, rule cert-shape). Reports loops without an accepted tuning, the
largest final M_s, and the change of the load-step IAE of the reported tuning. Writes limits_fixed.json.
"""
import glob, json
import numpy as np
out = {}
for sc in ("1", "2"):
    ref = {json.loads(l)["plant"]: json.loads(l)["res"] for f in glob.glob(f"safe{sc}_*.jsonl") for l in open(f)
           if json.loads(l)["rule"] == "cert-shape"}
    new = {json.loads(l)["plant"]: json.loads(l)["res"] for f in glob.glob(f"limf{sc}_*.jsonl") for l in open(f)}
    none = [p for p, r in new.items() if not r.get("worst")]
    fin = [r["worst"]["Ms_d"] for r in new.values() if r.get("worst")]
    ch = [new[p]["worst"]["IAE"] / ref[p]["worst"]["IAE"] - 1 for p in new if new[p].get("worst") and ref[p].get("worst")]
    rej_cert = sum(r["fired"].get(k, 0) for r in new.values() if r.get("fired") for k in r["fired"] if k.startswith("shape") or k == "certU")
    rej_counts = sum(r["fired"].get(k, 0) for r in new.values() if r.get("fired") for k in r["fired"] if k.startswith("G"))
    above = sum(r["traj"]["above"] for r in new.values())
    out[sc] = dict(loops=len(new), no_accept=none, max_final=max(fin), IAE_median=float(np.median(ch)),
                   IAE_max=float(np.max(ch)), rejections_certificate=rej_cert, rejections_counts=rej_counts,
                   iters_applied_above=above)
    print(sc, out[sc])
json.dump(out, open("limits_fixed.json", "w"), indent=1)
