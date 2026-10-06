import numpy as np, math
from battery import half_rule, simc, C_tf, freq, simulate, PLANTS
def S_rec(e,dt,w):
    d=np.diff(np.concatenate([[0.0],e])); t=np.arange(len(d))*dt; out=[]
    for wc in np.array_split(w,10): out.append(np.exp(-1j*np.outer(wc,t))@d)
    return np.concatenate(out)
print(f"{'loop':12}{'Ms':>7}" + "".join(f"{'noise '+str(s)+'%':>16}" for s in [0.5,1,2,5]))
for name in ['P1','P2','P3','P4']:
    K,L0,lags=PLANTS[name]; taus,th=half_rule(lags,L0,1); Kc,Ti,_=simc(K,taus,th,1)
    num,den=C_tf(Kc,Ti,0); GM,Ms=freq(K,L0,lags,num,den)
    dt=max(L0,1)/50; e=simulate(K,L0,lags,num,den,dt,80*max(L0,1)+15*max(lags))
    w=np.linspace(1e-3,2/max(L0,1),600); M0=np.max(abs(S_rec(e,dt,w)))
    row=f"{name+' SIMC PI':12}{M0:7.3f}"
    for s in [0.005,0.01,0.02,0.05]:
        vals=[np.max(abs(S_rec(e+s*np.random.default_rng(q).standard_normal(len(e)),dt,w))) for q in range(20)]
        row+=f"   {np.mean(vals):6.3f} ±{np.std(vals):5.3f}"
    print(row)
