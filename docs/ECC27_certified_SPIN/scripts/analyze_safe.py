"""
Safety during tuning (Theorem 3, safe SPIN). Reads safe1_*.jsonl (cautious start) and safe2_*.jsonl
(bold start), written by run_safe.sh, and reports for published, certified and safe SPIN:
  - applied: iterations (of 200) whose applied tuning had M_s > M* = 1.7, and the largest applied M_s
  - final: M_s of the most aggressive tuning accepted in the last 60 iterations, and its load-step IAE
Writes summary_safe.json.
"""
import glob, json, math
import numpy as np

MSTAR = 1.7
RULES = ["original", "cert-shape", "cert-ahead"]
out = {}
for sc in ("1", "2"):
    rows = {}
    for f in sorted(glob.glob(f"safe{sc}_*.jsonl")):
        for line in open(f):
            r = json.loads(line); rows.setdefault(r["plant"], {})[r["rule"]] = r["res"]
    tab = {}
    for plant, d in rows.items():
        tab[plant] = {}
        for rule in RULES:
            if rule not in d: continue
            res = d[rule]; tr = res["traj"]; w = res.get("worst")
            tab[plant][rule] = dict(above=tr["above"], above_safe=tr["above_safe"], max_applied=tr["max_Ms"],
                                    unstable=tr["unstable"], first_safe=tr["first_safe"],
                                    final=None if w is None else w["Ms_d"],
                                    IAE=None if w is None else w["IAE"], fired=res.get("fired"))
    S = {}
    for rule in RULES:
        v = [t[rule] for t in tab.values() if rule in t]
        if not v: continue
        fin = [x["final"] for x in v if x["final"] is not None]
        S[rule] = dict(loops=len(v), loops_applied_above=sum(x["above"] > 0 for x in v),
                       iters_above=sum(x["above"] for x in v), max_applied=max(x["max_applied"] for x in v),
                       after_first_safe_above=sum(x["above_safe"] for x in v),
                       loops_final_above=sum(f > MSTAR for f in fin), max_final=max(fin) if fin else None,
                       no_accept=sum(x["final"] is None for x in v))
    for a, b in (("cert-ahead", "original"), ("cert-ahead", "cert-shape"), ("cert-shape", "original")):
        ch = [t[a]["IAE"] / t[b]["IAE"] - 1 for t in tab.values()
              if a in t and b in t and t[a]["IAE"] and t[b]["IAE"]]
        if ch: S[f"IAE {a} vs {b}"] = dict(mean=float(np.mean(ch)), median=float(np.median(ch)),
                                           max=float(np.max(ch)), min=float(np.min(ch)), n=len(ch))
    out[f"start{sc}"] = dict(summary=S, loops=tab)
    print(f"--- start {sc} ({len(tab)} loops)")
    for k, v in S.items(): print(" ", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    print("  per loop: above / max applied / final  (original | cert-shape | cert-ahead)")
    for plant, t in tab.items():
        print(f"  {plant:24s}", " | ".join(
            f"{t[r]['above']:3d} {t[r]['max_applied']:.3f} {t[r]['final'] if t[r]['final'] is None else round(t[r]['final'], 3)}"
            for r in RULES if r in t))
json.dump(out, open("summary_safe.json", "w"), indent=1, default=lambda x: None if x == math.inf else x)
