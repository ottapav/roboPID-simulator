"""Compare the shaping rule with the elasticity rule and uniform back-off; validate the theory logs."""
import json, glob, sys, io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import iae_fine
from iae_fine import load_iae
def load(pre, rule):
    return {json.loads(l)['plant']: json.loads(l) for f in glob.glob(f'{pre}_*.jsonl') for l in open(f)
            if json.loads(l)['rule'] == rule}
for tag, (sh, el, un, orig) in {"cautious": ("shape", "boxtest", "unif", "box"), "bold": ("shape2", "box2test", "unif2", "box2")}.items():
    S, E, U, O = load(sh, "original+shape"), load(el, "original+test"), load(un, "original+uniform"), load(orig, "original")
    print(f"\n== {tag} start ==")
    logs = []
    for p in sorted(S):
        s = S[p]; r = s["res"]
        if r is None: print(p, "no convergence"); continue
        logs += r["shape_log"]
        m = s; ge = E[p]["res"]["worst"]["gains"]; gs = r["worst"]["gains"]; go = O[p]["res"]["worst"]["gains"]
        iS = load_iae(m["K"], m["L"], m["lags"], *gs); iE = load_iae(m["K"], m["L"], m["lags"], *ge); iO = load_iae(m["K"], m["L"], m["lags"], *go)
        u = U.get(p); iU = load_iae(m["K"], m["L"], m["lags"], *u["res"]["worst"]["gains"]) if u and u["res"] else None
        print(f"{p:22} Ms orig {O[p]['res']['worst']['Ms']:.3f} shape {r['worst']['Ms']:.3f} (read {r['worst']['read']:.3f}) elast {E[p]['res']['worst']['Ms']:.3f}"
              f" | IAE vs orig: shape {100*(iS/iO-1):+6.1f}% elast {100*(iE/iO-1):+6.1f}%" + (f" unif {100*(iU/iO-1):+6.1f}%" if iU else ""))
    if logs:
        pred = np.array([x["pred"] for x in logs]); tn = np.array([x["true_new"] for x in logs]); rd = np.array([x["read"] for x in logs])
        band = [x for x in logs if x["move"] != "all"]
        print(f"moves: {len(logs)}  band moves {len(band)}  uniform {len(logs)-len(band)}")
        print(f"exact prediction error |pred/true-1|: median {np.median(abs(pred/tn-1))*100:.2f}%  max {np.max(abs(pred/tn-1))*100:.2f}%")
        print(f"moves that lowered true M_s below the read value: {np.mean(tn < rd)*100:.0f}%")
        print(f"Re S > 1 at the peak: {np.mean([x['ReS'] > 1 for x in logs])*100:.0f}%   gradient says some SPIN move descends: {np.mean([x['grad_ok'] for x in logs])*100:.0f}%")
        print(f"gradient pick == chosen band move: {np.mean([x['grad_pick'] == x['move'] for x in band])*100:.0f}% of band moves")
