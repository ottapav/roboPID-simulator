import numpy as np, math, designed as D
from compare import counts_old
from spin_cert_run import sim
K,L,lags=1.0,16.0,D.FAMILY_LAGS; Ta=L+sum(lags); g=1/0.9
l=sorted(lags,reverse=True)+[0.0]; tau=l[0]+l[1]/2; th=L+l[1]/2+sum(l[2:])
Kp0=tau/(K*4*th); Ki0=Kp0/tau; Kd0=Kp0*l[1]*0.5; F=np.ones(3); dts=15*Ta/1500
for it in range(200):
    Ki,Kp,Kd=Ki0*F[0],Kp0*F[1],Kd0*F[2]; dt=min(dts,(Kd/Kp/10)/20); n=int(15*Ta/dt)
    e=sim(K,L,lags,Ki,Kp,Kd,dt,n)
    if abs(e[int(0.8*n):]).max()>0.5*abs(e).max():
        F=np.clip(F/np.array([2,4,8]),1e-4,1e4); print(it,'unstable',np.round(F,2),flush=True); continue
    st=max(1,int(round(dts/dt))); N=counts_old(e[::st],dt*st); V=[k for k in range(3) if N[k]>(0.5,0.75,1.0)[k]]
    kmin=min(V) if V else 3
    for k in range(3):
        if k<kmin: F[k]*=g
        elif k==kmin: F[k]/=g
    print(it,'none' if kmin==3 else f'G{kmin}',np.round(N,2),np.round(F,2),f'dt={dt:.4f} n={n}',flush=True)
