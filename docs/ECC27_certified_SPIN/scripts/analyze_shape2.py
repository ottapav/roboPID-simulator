"""Load IAE and M_s of five rules against published SPIN (box), per start."""
import json, glob, io, contextlib, sys, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import iae_fine
from iae_fine import load_iae
def load(pre, rule):
    return {json.loads(l)['plant']: json.loads(l) for f in glob.glob(f'{pre}_*.jsonl') for l in open(f)
            if json.loads(l)['rule'] == rule}
starts = {"cautious": dict(orig=("box", "original"), elast=("boxtest", "original+test"), unif=("unif", "original+uniform"),
                           shape=("shape", "original+shape"), shapehi=("shapehi", "original+shape-hi")),
          "bold": dict(orig=("box2", "original"), elast=("box2test", "original+test"), unif=("unif2", "original+uniform"),
                       shape=("shape2", "original+shape"), shapehi=("shapehi2", "original+shape-hi"))}
which = sys.argv[1:] or list(starts)
cache = {}
def iae(m, gains):
    k = (m["plant"], tuple(round(x, 12) for x in gains))
    if k not in cache: cache[k] = load_iae(m["K"], m["L"], m["lags"], *gains)
    return cache[k]
for st in which:
    D = {k: load(*v) for k, v in starts[st].items()}
    print(f"\n== {st} start ==   (M_s; load IAE change vs published SPIN)")
    summ = {k: [] for k in D if k != "orig"}
    for p in sorted(D["orig"]):
        o = D["orig"][p]
        if o["res"] is None: continue
        io_ = iae(o, o["res"]["worst"]["gains"]); line = f"{p:22} orig {o['res']['worst']['Ms']:.3f} |"
        for k in ("elast", "unif", "shape", "shapehi"):
            r = D[k].get(p)
            if r is None or r["res"] is None: line += f" {k} --      |"; continue
            v = 100 * (iae(o, r["res"]["worst"]["gains"]) / io_ - 1); summ[k].append((p, v, r["res"]["worst"]["Ms"]))
            line += f" {k} {r['res']['worst']['Ms']:.3f} {v:+6.1f}% |"
        print(line)
    for k, v in summ.items():
        if v: print(f"  {k:8} loops {len(v):2}  max M_s {max(x[2] for x in v):.3f}  mean IAE change {np.mean([x[1] for x in v]):+.1f}%  worst {max(x[1] for x in v):+.1f}%")
