import numpy as np, math
from scipy.special import lambertw
from scipy.optimize import brentq
from exactlaw import Ms, woa
T=np.arange(1.02,400.0,0.004)
def erec(k,t,N=120):
    out=np.zeros_like(t)
    for n in range(N):
        s=lambertw(-k,k=n); term=np.exp(s*t)/(1+s)
        out=out+(2*term.real if abs(s.imag)>1e-12 else term.real)
    return out
def ext_n(k,n):
    e=erec(k,T); d=np.diff(e)
    idx=np.where(np.sign(d[1:])!=np.sign(d[:-1]))[0]+1
    return abs(e[idx[n-1]]) if len(idx)>=n else 0.0
if __name__ == "__main__":
    print("sanity:", [round(ext_n(0.5236,n),4) for n in [1,2,3]], " (gm=3)")
    print("\n delta  n   k          w/a      Ms exact   Ms simulated   diff")
    meas={0.05:[1.613,2.056,2.559,3.082],0.02:[1.531,1.839,2.206,2.598],
          0.01:[1.495,1.738,2.038,2.363],0.005:[1.472,1.667,1.917,2.192]}
    for d,ms in meas.items():
        for n,m in zip([1,2,3,4],ms):
            f=lambda kk: ext_n(kk,n)-d
            k=brentq(f,0.37,1.50,xtol=1e-10)
            print(f" {d:<6} {n}  {k:.6f}  {woa(k):7.3f}  {Ms(k):8.3f}   {m:8.3f}   {Ms(k)-m:+.3f}")
