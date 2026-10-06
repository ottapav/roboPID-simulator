import numpy as np, math
from scipy.special import lambertw
from scipy.optimize import brentq

def woa(k):
    "omega/alpha of the dominant closed-loop pole, exact: s = W_{-1}(-k)"
    s=lambertw(-k,k=-1)
    return abs(s.imag/s.real)

def Ms(k):
    "1/Ms = min_w sqrt(w^2 - 2 k w sin w + k^2)/w   (exact)"
    w=np.linspace(1e-6,60,4_000_000)
    return 1/np.min(np.sqrt(w*w-2*k*w*np.sin(w)+k*k)/w)

def k_from_woa(r):
    return brentq(lambda k: woa(k)-r, 1/math.e+1e-9, math.pi/2-1e-12, xtol=1e-14)

if __name__ == "__main__":
    print("check against the frequency-response / root-finding values")
    print("  gm     k      w/a exact   Ms exact")
    for gm in [4,3,2.5,2.2,2.0,1.8,1.6,1.5]:
        k=math.pi/(2*gm)
        print(f" {gm:4}  {k:.4f}   {woa(k):8.3f}   {Ms(k):8.3f}")

    print("\nEXACT certificate: limit N = n/2  ->  w/a = n*pi/ln(1/delta)  ->  k  ->  Ms")
    print(" delta    n   w/a       k        Ms exact    Ms measured")
    meas={0.05:[1.613,2.056,2.559,3.082],0.02:[1.531,1.839,2.206,2.598],
          0.01:[1.495,1.738,2.038,2.363],0.005:[1.472,1.667,1.917,2.192]}
    for d,ms in meas.items():
        for n,m in zip([1,2,3,4],ms):
            r=n*math.pi/math.log(1/d); k=k_from_woa(r)
            print(f" {d:<7} {n}  {r:7.3f}  {k:.5f}  {Ms(k):8.3f}     {m:8.3f}")

    print("\nREFINED edge law: residue amplitude and phase from the same Lambert root")
    print(" e(t) = sum_p exp(s_p t)/(1+s_p);  dominant A = 2/|1+s_p|, phi = -arg(1+s_p)")
    print(" delta  n   Ms exact(refined)   Ms measured   err%")
    for d,ms in meas.items():
        for n,m in zip([1,2,3,4],ms):
            def F(k):
                s=lambertw(-k,k=-1); a,w=-s.real,abs(s.imag)
                A=2/abs(1+s); phi=-np.angle(1+s); psi=math.atan2(a,w)
                tn=(n*math.pi-phi+psi)/w                 # n-th extremum after the start
                return A*math.exp(-a*tn)-d
            try:
                k=brentq(F,1/math.e+1e-9,math.pi/2-1e-12,xtol=1e-14)
                print(f" {d:<6} {n}      {Ms(k):8.3f}        {m:8.3f}   {100*(Ms(k)-m)/m:+5.2f}")
            except ValueError as e:
                print(f" {d:<6} {n}      no bracket")
