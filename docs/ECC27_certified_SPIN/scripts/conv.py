import numpy as np, math
from battery import half_rule, simc, C_tf, freq, simulate, PLANTS
from spin_cert import S_rec
from ex2 import erec
print("A) exact loop k e^-s/s, record from the exact residue series (no simulation):")
k=0.7854
t=np.arange(1.0,120,0.001); e=np.concatenate([np.ones(1000),erec(k,t)])
w=np.linspace(1e-3,3,3000)
Ms_true=1/np.min(np.sqrt(w*w-2*k*w*np.sin(w)+k*k)/w)
for dt_skip in [8,4,2,1]:
    ee=e[::dt_skip]; dt=0.001*dt_skip
    print(f"   dt={dt:.3f}:  Ms from record {np.max(abs(S_rec(ee,dt,w))):.5f}   true {Ms_true:.5f}")
print("\nB) battery loop P4 under SIMC PI, zero-order-hold simulation, step shrinking:")
K,L0,lags=PLANTS['P4']; taus,th=half_rule(lags,L0,1); Kc,Ti,_=simc(K,taus,th,1); num,den=C_tf(Kc,Ti,0)
GM,Ms=freq(K,L0,lags,num,den); w=np.linspace(1e-3,0.75,2000)
for n in [25,50,100,200,400]:
    dt=L0/n; e=simulate(K,L0,lags,num,den,dt,420)
    Mr=np.max(abs(S_rec(e,dt,w))); print(f"   {n:3d} steps per dead time: Ms {Mr:.4f}  (true {Ms:.4f}, error {100*(Mr-Ms)/Ms:+.3f}%)")
