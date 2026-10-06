import numpy as np, math
from scipy.signal import tf2ss, cont2discrete
def Ms_k(k):
    w=np.linspace(1e-4,60,2_000_000); return 1/np.min(np.sqrt(w*w-2*k*w*np.sin(w)+k*k)/w)

# ---- plants: gain, delay, lags
PLANTS={'P1':(1.0,1.0,[10,1,1,1]),'P2':(1.25,8.0,[5,5,5,5]),
        'P3':(1.0,10.0,[2,1,1,1]),'P4':(1.0,4.0,[8]*6),
        'F-cancel':(1.0,1.0,[3.0]),         # control: exact FOPTD, tau/theta = 3
        'F-lagdom':(1.0,1.0,[20.0])}        # control: SIMC uses tauI = 8 theta < tau1

def half_rule(lags,theta0,order):
    l=sorted(lags,reverse=True)+[0,0,0]
    if order==1: return [l[0]+l[1]/2], theta0+l[1]/2+sum(l[2:])
    t1,t2=l[0],l[1]+l[2]/2
    return sorted([t1,t2],reverse=True), theta0+l[2]/2+sum(l[3:])

def simc(K,taus,theta,order):
    Kc=taus[0]/(K*(2*theta)); Ti=min(taus[0],8*theta)
    Td=taus[1] if order==2 else 0.0
    return Kc,Ti,Td

def C_tf(Kc,Ti,Td,N=10):
    if Td==0: return np.array([Kc*Ti,Kc]),np.array([Ti,0.0])
    return np.polymul([Kc*Ti,Kc],[Td,1.0]),np.polymul([Ti,0.0],[Td/N,1.0])

def freq(K,L,lags,num,den):
    w=np.logspace(-5,2,300000); s=1j*w
    G=K*np.exp(-s*L)
    for t in lags: G=G/(t*s+1)
    Lw=np.polyval(num,s)/np.polyval(den,s)*G
    ph=np.unwrap(np.angle(Lw)); i=np.argmin(abs(ph+np.pi))
    return 1/abs(Lw[i]), np.max(1/abs(1+Lw))

def simulate(K,L,lags,num,den,dt,tend):
    A,B,C,D=tf2ss(num,den); Ad,Bd,Cd,Dd,_=cont2discrete((A,B,C,D),dt,method='bilinear')
    xc=np.zeros((Ad.shape[0],1)); n=int(tend/dt); nd=int(round(L/dt))
    a=[math.exp(-dt/t) for t in lags]; x=[0.0]*len(lags)
    ub=np.zeros(n+nd+2); e=np.zeros(n); y=0.0
    for i in range(n):
        ek=1.0-y; e[i]=ek
        u=float((Cd@xc+Dd*ek).ravel()[0]); xc=Ad@xc+Bd*ek
        ub[i+nd]=K*u; v=ub[i]
        for j,aj in enumerate(a): x[j]=aj*x[j]+(1-aj)*v; v=x[j]
        y=v
    return e

def read(e,dt,Lg):
    y=1-e; E=np.cumsum(e)*dt; t=np.arange(len(e))*dt
    best=None
    for Ld in np.linspace(0.6*Lg,1.4*Lg,161):
        nd=int(round(Ld/dt)); m=(t>1.5*Ld)&(t<min(25*Ld,t[-1]))
        z=E[np.arange(len(e))-nd][m]/Ld; k=np.dot(z,y[m])/np.dot(z,z)
        r=math.sqrt(np.mean((y[m]-k*z)**2))
        if best is None or r<best[2]: best=(Ld,k,r)
    return best

def crossings(e,delta=0.02):
    kc=np.where(abs(e)>delta)[0][-1]+1; s=np.sign(e[:kc]); s=s[s!=0]
    return int(np.sum(s[1:]!=s[:-1]))

if __name__ == "__main__":
    print(f"{'plant':9}{'ctrl':4}{'theta':>7}{'GM true':>9}{'GM read':>9}{'Ms true':>9}{'Ms read':>9}"
          f"{'err Ms':>8}{'resid':>9}{'L_d read':>9}{'cross':>6}")
    rows=[]
    for name,(K,L,lags) in PLANTS.items():
        for order in ([1] if name.startswith('F') else [1,2]):
            taus,theta=half_rule(lags,L,order)
            Kc,Ti,Td=simc(K,taus,theta,order); num,den=C_tf(Kc,Ti,Td)
            GM,Ms=freq(K,L,lags,num,den)
            scale=max(theta,1.0); dt=scale/200
            e=simulate(K,L,lags,num,den,dt,40*scale+10*max(lags))
            Ld,k,r=read(e,dt,theta); GMr=math.pi/(2*k); Msr=Ms_k(min(k,1.55))
            n=crossings(e)
            rows.append((name,order,theta,GM,GMr,Ms,Msr,r,Ld,n))
            print(f"{name:9}{'PI' if order==1 else 'PID':4}{theta:7.2f}{GM:9.3f}{GMr:9.3f}{Ms:9.3f}{Msr:9.3f}"
                  f"{100*(Msr-Ms)/Ms:+7.1f}%{r:9.1e}{Ld:9.2f}{n:6d}")
    np.save('battery.npy',np.array([r[2:] for r in rows]))
