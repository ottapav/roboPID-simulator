import numpy as np, math, sys
import spin_run as S
PLANTS=S.PLANTS
def sim(K,L,lags,Ki,Kp,Kd,dt,n):
    nd=int(round(L/dt)); a=[math.exp(-dt/t) for t in lags]; x=[0.0]*len(lags)
    Tf=Kd/max(Kp,1e-12)/10; af=math.exp(-dt/Tf) if Kd>0 else 0.0
    ub=np.zeros(n+nd+1); e=np.zeros(n); I=0.0; xf=0.0; y=0.0
    for i in range(n):
        ek=1.0-y; e[i]=ek
        D=(Kd/Tf)*(ek-xf) if Kd>0 else 0.0
        if Kd>0: xf=af*xf+(1-af)*ek
        u=Kp*ek+Ki*I+D; I+=ek*dt
        ub[i+nd]=K*u; v=ub[i]
        for j,aj in enumerate(a): x[j]=aj*x[j]+(1-aj)*v; v=x[j]
        y=v
    return e
def S_rec(e,dt,w):
    d=np.diff(np.concatenate([[0.0],e])); t=np.arange(len(d))*dt
    return np.exp(-1j*np.outer(w,t))@d
def Ms_true(K,L,lags,Ki,Kp,Kd):
    w=np.logspace(-4,1.5,60000); s=1j*w; Tf=Kd/max(Kp,1e-12)/10
    C=Kp+Ki/s+(Kd*s/(Tf*s+1) if Kd>0 else 0); G=K*np.exp(-s*L)
    for t in lags: G=G/(t*s+1)
    return np.max(1/abs(1+C*G))
def elasticities(Ki,Kp,Kd,w):
    X=Kd*w-Ki/w; C2=Kp**2+X**2
    return np.array([-X*Ki/(w*C2), Kp**2/C2, X*Kd*w/C2])
def spin(name,pid,limits,Mstar=None,iters=200,beta=0.1):
    K,L,lags=PLANTS[name]; Ta=L+sum(lags); g=1/(1-beta); nb=3 if pid else 2
    l=sorted(lags,reverse=True); tau=l[0]+l[1]/2; th=L+l[1]/2+sum(l[2:])
    Kp0=tau/(K*4*th); Ki0=Kp0/tau; Kd0=(Kp0*l[1]*0.5 if pid else 0.0)
    F=np.ones(3); hist=[]; w=np.linspace(1e-3,3/max(L,1),400)
    for it in range(iters):
        Ki,Kp,Kd=Ki0*F[0],Kp0*F[1],Kd0*F[2]
        Tf=Kd/Kp/10 if Kd>0 else 1e9; dt=min(15*Ta/1500,Tf/5); n=int(15*Ta/dt)
        e=sim(K,L,lags,Ki,Kp,Kd,dt,n)
        if abs(e[int(0.8*n):]).max()>0.5*abs(e).max():
            F=np.clip(F/np.array([2,4,8]),1e-3,10); hist.append((it,'unstable')); continue
        N=S.counts(e,dt); V=[k for k in range(nb) if N[k]>limits[k]]
        cert=None
        if not V and Mstar is not None:
            Sw=S_rec(e,dt,w); i=np.argmax(abs(Sw)); cert=abs(Sw[i])
            if cert>Mstar:                                   # rejected: blame the band owning the peak
                V=[int(np.argmax(elasticities(Ki,Kp,Kd,w[i])[:nb]))]
        kmin=min(V) if V else 3
        for k in range(nb):
            if k<kmin: F[k]*=g
            elif k==kmin: F[k]/=g
        F=np.clip(F,1e-3,10); hist.append((it,N,V,(Ki,Kp,Kd),cert))
    acc=[h for h in hist[-60:] if h[1]!='unstable' and not h[2]]
    return [(Ms_true(K,L,lags,*h[3]),h[4]) for h in acc]
if __name__=="__main__":
    for lims,Mstar,lab in [((0.5,1.0,1.5),None,"rising half turns, no certificate"),
                           ((0.5,1.0,1.5),1.7,"rising half turns + certificate M*=1.7"),
                           ((0.5,0.5,1.0),1.7,"(1/2,1/2,1) + certificate M*=1.7")]:
        print(f"\n{lab}")
        for name in PLANTS:
            for pid in [False,True]:
                r=spin(name,pid,lims,Mstar)
                if not r: print(f"  {name} {'PID' if pid else 'PI '}: no accepted iterate in the final 60"); continue
                M=np.array([x[0] for x in r]); C=[x[1] for x in r if x[1] is not None]
                cs=f"  record {min(C):.2f}-{max(C):.2f}" if C else ""
                print(f"  {name} {'PID' if pid else 'PI '}: true Ms {M.min():.2f}-{M.max():.2f}{cs}   ({len(r)} accepted)")
        sys.stdout.flush()
