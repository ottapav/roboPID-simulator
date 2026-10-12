import json, sys, designed as D
names = sys.argv[1].split(","); out = sys.argv[2]
with open(out, "a") as f:
    for p in D.PLANTS:
        if p[0] not in names: continue
        res = D.run(p[1], p[2], p[3], p[4], "cert-shape")
        f.write(json.dumps(dict(plant=p[0], K=p[1], L=p[2], lags=p[3], pid=p[4], rule="cert-shape", res=res)) + "\n"); f.flush()
        print(p[0], None if res is None else res["Ms_all"][-1], flush=True)
print("DONE")
