import numpy as np, math, sys
PLANTS={'P1':(1.0,1.0,[10,1,1,1]),'P2':(1.25,8.0,[5,5,5,5]),
        'P3':(1.0,10.0,[2,1,1,1]),'P4':(1.0,4.0,[8]*6)}
def wind(p,q):
    a=np.unwrap(np.arctan2(q,p)); return (a[-1]-a[0])/(2*math.pi)
def whiten(p,q):
    Z=np.vstack([p,q]); w,V=np.linalg.eigh(Z@Z.T/Z.shape[1])
    if w.min()<=1e-14*w.max(): return None
    Y=V@np.diag(1/np.sqrt(w))@V.T@Z; r=np.hypot(*Y); return Y/r.max()
def counts(e,dt,delta=0.02,eps=0.1):
    big=np.where(abs(e)>delta*abs(e).max())[0]; kd=min(len(e),big[-1]+2)
    E=np.cumsum(e)*dt; de=np.diff(e,prepend=e[0]); d2=np.diff(de,prepend=de[0])
    N=[]
    for p,q in [(e[:kd],(E-E[-1])[:kd]),(de[:kd],e[:kd]),(d2[:kd],de[:kd])]:
        Y=whiten(p,q)
        if Y is None: N.append(0.0); continue
        k0=np.where(np.hypot(*Y)>eps)[0]; N.append(wind(Y[0][k0[0]:],Y[1][k0[0]:]) if len(k0) else 0.0)
    return N
def sim(K,L,lags,Ki,Kp,Kd,dt,n):
    nd=int(round(L/dt)); a=[math.exp(-dt/t) for t in lags]; x=[0.0]*len(lags)
    Tf=max(Kd/max(Kp,1e-9)/10,dt); af=math.exp(-dt/Tf)
    ub=np.zeros(n+nd+1); e=np.zeros(n); I=0.0; ef=1.0; y=0.0
    for i in range(n):
        ek=1.0-y; e[i]=ek
        efn=af*ef+(1-af)*ek; D=Kd*(efn-ef)/dt if Kd>0 else 0.0; ef=efn
        u=Kp*ek+Ki*I+D; I+=ek*dt
        u=max(-10,min(10,u)); ub[i+nd]=K*u; v=ub[i]
        for j,aj in enumerate(a): x[j]=aj*x[j]+(1-aj)*v; v=x[j]
        y=v
    return e
def Ms_true(K,L,lags,Ki,Kp,Kd):
    w=np.logspace(-4,1.5,60000); s=1j*w; Tf=max(Kd/max(Kp,1e-9)/10,1e-9)
    C=Kp+Ki/s+Kd*s/(Tf*s+1); G=K*np.exp(-s*L)
    for t in lags: G=G/(t*s+1)
    Lw=C*G; ph=np.unwrap(np.angle(Lw)); i=np.argmin(abs(ph+np.pi))
    return np.max(1/abs(1+Lw)), 1/abs(Lw[i])
def spin(name,pid,limits,iters=160,beta=0.1):
    K,L,lags=PLANTS[name]; Ta=L+sum(lags); dt=12*Ta/1500; n=1500; g=1/(1-beta)
    # conservative start: SIMC-like with tau_c = 3 theta on a half-rule FOPTD
    l=sorted(lags,reverse=True); tau=l[0]+l[1]/2; th=L+l[1]/2+sum(l[2:])
    Kp0=tau/(K*4*th); Ki0=Kp0/tau; Kd0=(Kp0*l[1]*0.5 if pid else 0.0)
    F=np.ones(3); hist=[]
    for it in range(iters):
        Ki,Kp,Kd=Ki0*F[0],Kp0*F[1],Kd0*F[2]
        e=sim(K,L,lags,Ki,Kp,Kd,dt,n)
        tail=e[int(0.8*n):]
        if abs(tail).max()>0.5*abs(e).max():         # divergence screen
            F=np.clip(F/np.array([2,4,8]),1e-3,10); hist.append((it,None)); continue
        N=counts(e,dt); V=[k for k in range(3 if pid else 2) if N[k]>limits[k]]
        kmin=min(V) if V else 3
        for k in range(3 if pid else 2):
            if k<kmin: F[k]*=g
            elif k==kmin: F[k]/=g
        F=np.clip(F,1e-3,10); hist.append((it,N,V,(Ki,Kp,Kd)))
    feas=[h for h in hist[-60:] if h[1] is not None and not h[2]]
    ms=[Ms_true(K,L,lags,*h[3]) for h in feas]
    return ms, feas
if __name__=="__main__":
    for lims,lab in [((0.5,1.0,1.5),"new limits (0.5, 1.0, 1.5)"),((0.5,0.75,1.0),"old SPIN limits (0.5, 0.75, 1.0)")]:
      print(f"\n{lab}, unified index, delta = 0.02; feasible iterates in the final limit cycle")
      for name in PLANTS:
          for pid in [False,True]:
              ms,feas=spin(name,pid,lims)
              if not ms: print(f"  {name} {'PID' if pid else 'PI '}: no feasible iterate in final cycle"); continue
              M=np.array([m[0] for m in ms]); G=np.array([m[1] for m in ms])
              print(f"  {name} {'PID' if pid else 'PI '}: Ms {M.min():.2f}-{M.max():.2f}   GM {G.min():.2f}-{G.max():.2f}   ({len(ms)} iterates)")
      sys.stdout.flush()
