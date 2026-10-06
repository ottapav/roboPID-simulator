import numpy as np, math, sys
import spin_run as S
from spin_cert_run import sim, S_rec, Ms_true, elasticities
PLANTS=S.PLANTS
def turn(p,q,eps=0.1):                          # SPIN Definition 1 (per-axis peak normalization)
    pm,qm=np.max(abs(p)),np.max(abs(q))
    if pm==0 or qm==0: return 0.0
    p=p/pm; q=q/qm; r=np.hypot(p,q); out=np.where(r>eps)[0]
    if len(out)==0: return 0.0
    k0=out[0]; ins=np.where(r[k0:]<=eps)[0]; end=k0+ins[-1]+1 if len(ins) else len(p)
    a=np.unwrap(np.arctan2(q[:end],p[:end])); return (a[-1]-a[0])/(2*math.pi)
def counts_old(e,dt,delta=0.02):                # SPIN Definitions 1 and 2
    big=np.where(abs(e)>delta*abs(e).max())[0]; kd=min(len(e),big[-1]+2)
    E=np.cumsum(e)*dt; de=np.diff(e,prepend=e[0]); d2=np.diff(de,prepend=de[0])
    return [turn(e[:kd],(E-E[-1])[:kd]), turn(de[:kd],e[:kd]), turn(d2[:kd],de[:kd])]
def load_iae(K,L,lags,Ki,Kp,Kd):
    Ta=L+sum(lags); Tf=Kd/Kp/10 if Kd>0 else 1e9; dt=min(Ta/300,Tf/5); n=int(40*Ta/dt)
    nd=int(round(L/dt)); a=[math.exp(-dt/t) for t in lags]; x=[0.0]*len(lags)
    af=math.exp(-dt/Tf) if Kd>0 else 0.0; ub=np.zeros(n+nd+1); I=0.0; xf=0.0; y=0.0; iae=0.0
    for i in range(n):
        ek=-y; D=(Kd/Tf)*(ek-xf) if Kd>0 else 0.0
        if Kd>0: xf=af*xf+(1-af)*ek
        u=Kp*ek+Ki*I+D; I+=ek*dt
        ub[i+nd]=K*(u+1.0); v=ub[i]                       # unit load step at the plant input
        for j,aj in enumerate(a): x[j]=aj*x[j]+(1-aj)*v; v=x[j]
        y=v; iae+=abs(y)*dt
    return iae
def spin(name,pid,counter,limits,Mstar=None,iters=200,beta=0.1):
    K,L,lags=PLANTS[name]; Ta=L+sum(lags); g=1/(1-beta); nb=3 if pid else 2
    l=sorted(lags,reverse=True); tau=l[0]+l[1]/2; th=L+l[1]/2+sum(l[2:])
    Kp0=tau/(K*4*th); Ki0=Kp0/tau; Kd0=(Kp0*l[1]*0.5 if pid else 0.0)
    F=np.ones(3); acc=[]; w=np.linspace(1e-3,3/max(L,1),400)
    for it in range(iters):
        Ki,Kp,Kd=Ki0*F[0],Kp0*F[1],Kd0*F[2]
        Tf=Kd/Kp/10 if Kd>0 else 1e9; dt=min(15*Ta/1500,Tf/5); n=int(15*Ta/dt)
        e=sim(K,L,lags,Ki,Kp,Kd,dt,n)
        if abs(e[int(0.8*n):]).max()>0.5*abs(e).max():
            F=np.clip(F/np.array([2,4,8]),1e-3,10); continue
        N=counter(e,dt); V=[k for k in range(nb) if N[k]>limits[k]]
        if not V and Mstar is not None:
            Sw=S_rec(e,dt,w); i=np.argmax(abs(Sw))
            if abs(Sw[i])>Mstar: V=[int(np.argmax(elasticities(Ki,Kp,Kd,w[i])[:nb]))]
        if not V and it>=iters-60: acc.append((Ki,Kp,Kd))
        kmin=min(V) if V else 3
        for k in range(nb):
            if k<kmin: F[k]*=g
            elif k==kmin: F[k]/=g
        F=np.clip(F,1e-3,10)
    return acc[-1] if acc else None
if __name__=="__main__":
    methods={'old':(counts_old,(0.5,0.75,1.0),None),'new':(S.counts,(0.5,1.0,1.5),1.7)}
    rows=[]
    for name in PLANTS:
        K,L,lags=PLANTS[name]
        for pid in [False,True]:
            r=[name,'PID' if pid else 'PI']
            for m,(cnt,lim,Ms_) in methods.items():
                gains=spin(name,pid,cnt,lim,Ms_)
                if gains is None: r+= [np.nan,np.nan,np.nan]; continue
                w=np.logspace(-4,1.5,60000); s=1j*w; Ki,Kp,Kd=gains; Tf=Kd/Kp/10 if Kd>0 else 1
                C=Kp+Ki/s+(Kd*s/(Tf*s+1) if Kd>0 else 0); G=K*np.exp(-s*L)
                for t in lags: G=G/(t*s+1)
                Lw=C*G; ph=np.unwrap(np.angle(Lw)); GM=1/abs(Lw[np.argmin(abs(ph+np.pi))])
                r+=[np.max(1/abs(1+Lw)),GM,load_iae(K,L,lags,*gains)]
                r+=[gains]
            rows.append(r); print(r[:5],r[6:9],flush=True)
    import pickle; pickle.dump(rows,open('compare.pkl','wb'))
