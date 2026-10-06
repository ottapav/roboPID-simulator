import numpy as np, math
PLANTS={'P1':(1.0,1.0,[10,1,1,1]),'P2':(1.25,8.0,[5,5,5,5]),'P3':(1.0,10.0,[2,1,1,1]),'P4':(1.0,4.0,[8]*6)}
def half_rule2(lags,L):
    l=sorted(lags,reverse=True)+[0,0]; t1,t2=l[0],l[1]+l[2]/2; return sorted([t1,t2],reverse=True), L+l[2]/2+sum(l[3:])
def simc_pid_parallel(K,lags,L):
    (t1,t2),th=half_rule2(lags,L); Kc=t1/(K*2*th); Ti=min(t1,8*th); Td=t2
    return Kc*(1+Td/Ti), Kc/Ti, Kc*Td          # series -> parallel Kp, Ki, Kd
def sim(K,L,lags,Kp,Ki,Kd,b,c,dt,n):
    """ISA PID: u = Kp(b r - y) + Ki int(r-y) + Kd D[c r - y], D = s/(Tf s+1), Tf = Td/10"""
    Tf=(Kd/Kp)/10; af=math.exp(-dt/Tf); nd=int(round(L/dt))
    a=[math.exp(-dt/t) for t in lags]; x=[0.0]*len(lags); ub=np.zeros(n+nd+1)
    y_=np.zeros(n); u_=np.zeros(n); I=0.0; xf=0.0; y=0.0; r=1.0
    for i in range(n):
        dsig=c*r-y; D=(Kd/Tf)*(dsig-xf); xf=af*xf+(1-af)*dsig
        u=Kp*(b*r-y)+Ki*I+D; I+=(r-y)*dt
        y_[i]=y; u_[i]=u; ub[i+nd]=K*u; v=ub[i]
        for j,aj in enumerate(a): x[j]=aj*x[j]+(1-aj)*v; v=x[j]
        y=v
    return y_,u_
def F(sig,dt,w,start):
    d=np.diff(np.concatenate([[start],sig])); t=np.arange(len(d))*dt; out=[]
    for wc in np.array_split(w,12): out.append(np.exp(-1j*np.outer(wc,t))@d)
    return np.concatenate(out)
print(f"{'loop':6}{'b':>5}{'c':>3}{'Ms true':>9}{'naive jwE':>11}{'from u,y':>10}")
for name,(K,L,lags) in PLANTS.items():
    Kp,Ki,Kd=simc_pid_parallel(K,lags,L); Tf=(Kd/Kp)/10
    Ta=L+sum(lags); dt=min(Ta/2000,Tf/20); n=int(40*Ta/dt)   # controller resolved as in designed.py
    w=np.linspace(1e-3,3/(L+sorted(lags)[-1]*0+1e-9) if False else 3/max(L,1),1200)
    s=1j*w; Cfb=Kp+Ki/s+Kd*s/(Tf*s+1); G=K*np.exp(-s*L)
    for t in lags: G=G/(t*s+1)
    wf=np.logspace(-4,2,200000); sf=1j*wf; Gf=K*np.exp(-sf*L)
    for t in lags: Gf=Gf/(t*sf+1)
    Ms=np.max(abs(1/(1+(Kp+Ki/sf+Kd*sf/(Tf*sf+1))*Gf)))   # true peak on a fine grid
    for b,c in [(1,1),(0.5,0),(0,0)]:
        y,u=sim(K,L,lags,Kp,Ki,Kd,b,c,dt,n)
        Sn=1+F(-y,dt,w,0.0)*0+F(1-y,dt,w,0.0)   # naive: S = 1 + F{de/dt}, e = 1 - y, e(0-) = 0
        Sn=F(1-y,dt,w,0.0)                        # includes the jump 0 -> 1 at t = 0
        Gr=F(y,dt,w,0.0)/F(u,dt,w,0.0); Sr=1/(1+Cfb*Gr)
        print(f"{name:6}{b:5.1f}{c:3d}{Ms:9.3f}{np.max(abs(Sn)):11.3f}{np.max(abs(Sr)):10.3f}")
