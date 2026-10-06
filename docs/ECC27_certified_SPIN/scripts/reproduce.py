"""
Reproduces the Section 2 numbers of the ECC 2027 paper (Prop. 1-2, Table I, the 630-loop grid).
(SPIN runs of Section 4: designed.py).

Part A needs no simulation: the loop  de/dt = -k e(t-1)  is solved exactly
(Lambert W poles, residue series, one-line peak).  A zero-order-hold
simulation is used only to cross-check, and for the Part B grid.

Run:   python3 reproduce.py             (Part A and whitening, a few minutes)
       python3 reproduce.py --battery   (adds the SPIN battery test, Table 5)
       python3 reproduce.py --grid      (adds the 630-loop alarm grid, slow)
Needs: numpy, scipy.
"""
import sys, math
import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq

K_MIN, K_MAX = 1 / math.e + 1e-9, math.pi / 2 - 1e-9


# ---------------------------------------------------------------- exact loop
def dominant(k):
    """dominant closed-loop pole s = W_{-1}(-k) = -alpha + j omega"""
    s = lambertw(-k, k=-1)
    return s, -s.real, abs(s.imag)


def q_of(k):
    """decay per half cycle, q = exp(-pi alpha / omega)"""
    _, a, w = dominant(k)
    return math.exp(-math.pi * a / w)


def Ms(k):
    """1/Ms = min_w sqrt(w^2 - 2 k w sin w + k^2) / w"""
    w = np.linspace(1e-4, 60, 2_000_000)
    return 1 / np.min(np.sqrt(w * w - 2 * k * w * np.sin(w) + k * k) / w)


def record(k, t, branches=80):
    """exact error record: sum_j exp(s_j t)/(1 + s_j), s_j = W_j(-k)"""
    e = np.zeros_like(t)
    for j in range(branches):
        s = lambertw(-k, k=j)
        term = np.exp(s * t) / (1 + s)
        e = e + (2 * term.real if abs(s.imag) > 1e-12 else term.real)
    return e


def extrema(t, e):
    d = np.diff(e)
    idx = np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0] + 1
    return idx


# ---------------------------------------------------------- ZOH cross-check
def zoh(k, dt, tend):
    """PI with Ti = T on a FOPTD plant, L_d = T = K = 1, so the loop is k e^{-s}/s"""
    n, nd = int(tend / dt), int(round(1 / dt))
    a = math.exp(-dt)
    x = I = 0.0
    ub = np.zeros(n + nd + 1)
    e = np.zeros(n)
    for i in range(n):
        ek = 1.0 - x
        e[i] = ek
        ub[i + nd] = k * (ek + I)
        I += ek * dt
        x = a * x + (1 - a) * ub[i]
    return e


def section(title):
    print("\n" + title + "\n" + "-" * len(title))


def main():
    # ---- series vs simulation
    section("Proposition 3: the ZOH simulation converges to the exact series (k = 0.632)")
    k = 0.632
    ex = record(k, np.array([5.0, 8.0]))
    for n in [100, 400, 1600, 6400]:
        dt = 1 / n
        e = zoh(k, dt, 9)
        print(f"  {n:5d} steps per dead time:  error at t=5 {e[int(round(5 / dt))] - ex[0]:+.1e}"
              f"   at t=8 {e[int(round(8 / dt))] - ex[1]:+.1e}")


    # ---- the reading: y(t) = k z(t),  z(t) = (1/L_d) int_0^{t-L_d} e
    section("Section 4: reading k by one least-squares ratio")

    def pi_sim(T, Kp, Ti, dt, tend):
        n, nd = int(tend / dt), int(round(1 / dt))
        a = math.exp(-dt / T)
        x = I = 0.0
        ub = np.zeros(n + nd + 1)
        e = np.zeros(n)
        for i in range(n):
            ek = 1.0 - x
            e[i] = ek
            ub[i + nd] = Kp * (ek + I / Ti)
            I += ek * dt
            x = a * x + (1 - a) * ub[i]
        return e

    def gm_ms(T, Kp, Ti):
        w = np.logspace(-4, 3, 200000)
        s = 1j * w
        L = Kp * (1 + 1 / (Ti * s)) * np.exp(-s) / (T * s + 1)
        ph = np.unwrap(np.angle(L))
        i = np.argmin(abs(ph + np.pi))
        return 1 / abs(L[i]), np.max(1 / abs(1 + L))

    def read(e, Ld, dt):
        y = 1 - e
        E = np.cumsum(e) * dt
        nd = int(round(Ld / dt))
        t = np.arange(len(e)) * dt
        m = (t > 1.5 * Ld) & (t < 25)
        z = E[np.arange(len(e)) - nd][m] / Ld
        k = np.dot(z, y[m]) / np.dot(z, z)
        return k, np.sqrt(np.mean((y[m] - k * z) ** 2))

    dt = 1 / 400
    for T in [0.3, 1.0, 5.0]:
        for gm in [3.0, 2.0, 1.6]:
            g1 = gm_ms(T, 1.0, T)[0]
            e = pi_sim(T, g1 / gm, T, dt, 40)
            kk, res = read(e, 1.0, dt)
            print(f"  T/L={T:<4} true GM {gm:4}  read GM {math.pi / (2 * kk):.3f}   residual {res:.1e}")
    g1 = gm_ms(1.0, 1.0, 1.0)[0]
    e = pi_sim(1.0, g1 / 2.0, 1.0, dt, 40)
    print("  dead time from the residual minimum (true 1.0):",
          "  ".join(f"{Ld}:{read(e, Ld, dt)[1]:.1e}" for Ld in [0.9, 0.95, 1.0, 1.05, 1.1]))
    for sd in [0.01, 0.02]:
        en = e + sd * np.random.default_rng(1).standard_normal(len(e))
        print(f"  noise {sd:.2f}: read GM {math.pi / (2 * read(en, 1.0, dt)[0]):.3f} (true 2.000)")
    print("  integral time mismatch, true GM 2.5:")
    for r_ in [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]:
        g1 = gm_ms(1.0, 1.0, r_)[0]
        Kp = g1 / 2.5
        ms_true = gm_ms(1.0, Kp, r_)[1]
        kk, res = read(pi_sim(1.0, Kp, r_, dt, 60), 1.0, dt)
        print(f"    Ti/T={r_:<5} residual {res:.1e}   Ms error {100 * (Ms(min(kk, 1.55)) / ms_true - 1):+.1f}%")
    section("Table 2: gain margin -> Ms, exact")
    for gm in [4.0, math.pi, 3.0, 2.5, 2.0, 1.7, 1.5]:
        print(f"  GM {gm:5.3f}   Ms {Ms(math.pi / (2 * gm)):.3f}")

    # ---- crossings and extrema
    section("Proposition 4: crossings, extrema, and the constant ratio q (k = 0.632)")
    t = np.arange(0.0, 30, 0.0005)
    e = record(k, t)
    z = t[np.where(np.sign(e[1:]) != np.sign(e[:-1]))[0]]
    x = extrema(t, e)
    x = x[t[x] > 1.05]
    for zi, xi in list(zip(z, t[x]))[:3]:
        print(f"  crossing {zi:7.4f}   extremum {xi:7.4f}   gap {xi - zi:.4f} dead times")
    ev = abs(e[x[:5]])
    print("  successive ratios:", "  ".join(f"{b / a:.6f}" for a, b in zip(ev, ev[1:])),
          f"   q = {q_of(k):.6f}")
    rs = []
    for kk in np.linspace(0.40, 1.50, 23):
        s1, s2 = lambertw(-kk, k=-1), lambertw(-kk, k=1)
        rs.append(s2.real / s1.real)
    print(f"  next pair decays r = alpha2/alpha >= {min(rs):.2f} times faster (up to {max(rs):.0f} near the limit)")

    # ---- decay-ratio anchors
    section("Reference values: decay ratio q^2 -> Ms")
    for dr in [1 / 100, 1 / 25, 1 / 4]:
        kk = brentq(lambda kk: q_of(kk) - math.sqrt(dr), K_MIN, K_MAX, xtol=1e-13)
        print(f"  q^2 = {dr:.4f}   Ms = {Ms(kk):.3f}")

    # ---- certificate table from exact edges  o q^(n-1) = delta
    section("Table 1: fewer than n crossings certify Ms <= entry (exact edges)")
    T = np.arange(1.02, 250.0, 0.01)

    def nth_ext(kk, n):
        ee = record(kk, T, 60)
        idx = extrema(T, ee)
        return abs(ee[idx[n - 1]]) if len(idx) >= n else 0.0

    print("  delta    n=1     n=2     n=3     (Ms(q) at q = delta^(1/n))")
    for d in [0.05, 0.02, 0.01, 0.005, 0.002]:
        row, coll = [], []
        for n in [1, 2, 3]:
            kk = brentq(lambda kk: nth_ext(kk, n) - d, 0.37, 1.50, xtol=1e-9)
            row.append(Ms(kk))
            kq = brentq(lambda kk: q_of(kk) - d ** (1 / n), K_MIN, K_MAX, xtol=1e-13)
            coll.append(Ms(kq))
        print(f"  {d:<6} " + "  ".join(f"{m:6.3f}" for m in row)
              + "     (" + "  ".join(f"{m:.3f}" for m in coll) + ")")

    print("  crossing thresholds at delta=0.02 (Table 3):")
    for n in [1, 2, 3]:
        kk = brentq(lambda kk: nth_ext(kk, n) - 0.02, 0.37, 1.50, xtol=1e-9)
        print(f"    at most {n - 1} crossings: GM >= {math.pi / (2 * kk):.3f}   Ms <= {Ms(kk):.3f}")

    # ---- V0: the dead-time corner
    section("The dead-time corner: virtual starting peak V0 of the dominant mode")
    for kk in [0.40, 0.60, 1.00, 1.50]:
        s, _, _ = dominant(kk)
        c = 1 / (1 + s)
        tt = np.linspace(0.2, 1.2, 100001)
        v = 2 * np.real(c * np.exp(s * tt))
        i = np.argmax(abs(v))
        print(f"  k={kk:.2f}   V0 = {abs(v[i]):.4f}  at t = {tt[i]:.3f} dead times")

    # ---- whitening on an analytic mode
    section("Proposition 6: spread between the three counts on one analytic mode")

    def wind(p, q):
        ang = np.unwrap(np.arctan2(q, p))
        return (ang[-1] - ang[0]) / (2 * math.pi)

    def whiten(p, q):
        Z = np.vstack([p, q])
        w, V = np.linalg.eigh(Z @ Z.T / Z.shape[1])
        Y = V @ np.diag(1 / np.sqrt(w)) @ V.T @ Z
        return Y[0], Y[1]

    for zeta in [0.1, 0.3, 0.5, 0.7]:
        a, w = zeta, math.sqrt(1 - zeta ** 2)
        t = np.linspace(0, 2.62 * 2 * math.pi / w, 50000)
        x = np.exp(-a * t) * np.cos(w * t)
        dx = np.exp(-a * t) * (-a * np.cos(w * t) - w * np.sin(w * t))
        d2 = np.exp(-a * t) * ((a * a - w * w) * np.cos(w * t) + 2 * a * w * np.sin(w * t))
        X = np.exp(-a * t) * (-a * np.cos(w * t) + w * np.sin(w * t)) / (a * a + w * w)
        P = [(x, X), (dx, x), (d2, dx)]
        peak = [wind(p / abs(p).max(), q / abs(q).max()) for p, q in P]
        white = [wind(*whiten(p, q)) for p, q in P]
        print(f"  zeta={zeta}: per-axis spread {max(peak) - min(peak):.4f}"
              f"   whitened spread {max(white) - min(white):.6f}")

    section("Theorem 1: the sensitivity peak from one step record")
    record_certificate()

    if "--battery" in sys.argv:
        section("Section 8: SPIN battery, SIMC tunings (slow)")
        battery()

    # ---- Part B grid
    if "--grid" in sys.argv:
        section("Part B: 630 first-order loops, crossing counts (slow)")
        grid()


def grid():
    def freq(T, Kp, Ti):
        w = np.logspace(-4, 3, 200000)
        s = 1j * w
        L = Kp * (1 + 1 / (Ti * s)) * np.exp(-s) / (T * s + 1)
        ph = np.unwrap(np.angle(L))
        i = np.argmin(abs(ph + np.pi))
        return np.max(1 / abs(1 + L)), 1 / abs(L[i])

    def sim(T, Kp, Ti, dt, tend):
        n, nd = int(tend / dt), int(round(1 / dt))
        a = math.exp(-dt / T)
        x = I = 0.0
        ub = np.zeros(n + nd + 1)
        e = np.zeros(n)
        for i in range(n):
            ek = 1.0 - x
            e[i] = ek
            ub[i + nd] = Kp * (ek + I / Ti)
            I += ek * dt
            x = a * x + (1 - a) * ub[i]
        return e

    def crossings(e, delta=0.02):
        kc = np.where(abs(e) > delta)[0][-1] + 1
        s = np.sign(e[:kc])
        s = s[s != 0]
        return int(np.sum(s[1:] != s[:-1]))

    rows = []
    for T in [0.3, 1.0, 5.0]:
        for r in [0.5, 0.75, 1, 1.5, 2, 4, 8]:
            Ti = r * T
            g1 = freq(T, 1.0, Ti)[1]
            for gm in np.linspace(6, 1.3, 30):
                Kp = g1 / gm
                rows.append((T, r, freq(T, Kp, Ti)[0],
                             crossings(sim(T, Kp, Ti, 0.01, min(1500, 60 + 30 * Ti)))))
    A = np.array(rows)
    for n in [1, 2, 3, 4]:
        m = A[:, 3] >= n
        print(f"  {n}+ crossings: min Ms = {A[m, 2].min():.2f}")
    print(f"  0 crossings:  max Ms = {A[A[:, 3] == 0, 2].max():.2f}")
    m = A[:, 1] == 1
    v = sum(((A[m, 3] < n) & (A[m, 2] > l)).sum() for n, l in [(1, 1.535), (2, 1.845), (3, 2.215)])
    print(f"  cancelling loops: {m.sum()}, certificate violations: {v}")



def record_certificate():
    """Theorem 1: S(jw) = 1 + int de/dt exp(-jwt) dt, from the error record alone."""
    from scipy.signal import tf2ss, cont2discrete

    def sim(K, L, lags, Kc, Ti, dt, tend):
        A, B, C, D = tf2ss([Kc * Ti, Kc], [Ti, 0.0])
        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method="bilinear")
        xc = np.zeros((Ad.shape[0], 1)); n = int(tend / dt); nd = int(round(L / dt))
        a = [math.exp(-dt / t) for t in lags]; x = [0.0] * len(lags)
        ub = np.zeros(n + nd + 2); e = np.zeros(n); y = 0.0
        for i in range(n):
            ek = 1.0 - y; e[i] = ek
            u = float((Cd @ xc + Dd * ek).ravel()[0]); xc = Ad @ xc + Bd * ek
            ub[i + nd] = K * u; v = ub[i]
            for j, aj in enumerate(a):
                x[j] = aj * x[j] + (1 - aj) * v; v = x[j]
            y = v
        return e

    def S_true(K, L, lags, Kc, Ti, w):
        s = 1j * w; G = K * np.exp(-s * L)
        for t in lags: G = G / (t * s + 1)
        return 1 / (1 + Kc * (1 + 1 / (Ti * s)) * G)

    def S_rec(e, dt, w):
        de = np.diff(e); t = (np.arange(len(de)) + 0.5) * dt; out = []
        for wc in np.array_split(w, 15):
            out.append(1 + np.exp(-1j * np.outer(wc, t)) @ de)
        return np.concatenate(out)

    cases = []
    for name, (K, L, lags) in {"P1": (1.0, 1.0, [10, 1, 1, 1]), "P2": (1.25, 8.0, [5, 5, 5, 5]),
                               "P3": (1.0, 10.0, [2, 1, 1, 1]), "P4": (1.0, 4.0, [8] * 6)}.items():
        l = sorted(lags, reverse=True); tau = l[0] + l[1] / 2; th = L + l[1] / 2 + sum(l[2:])
        cases.append((name + " SIMC PI", K, L, lags, tau / (K * 2 * th), min(tau, 8 * th)))
    # slow integral action: T = 0.3 L, Ti = 8T, gain margin 1.3 (the zero-crossing case is in --grid)
    w = np.logspace(-4, 2, 200000); T = 0.3; Ti = 8 * T
    Lw = (1 + 1 / (Ti * 1j * w)) * np.exp(-1j * w) / (T * 1j * w + 1)
    ph = np.unwrap(np.angle(Lw)); g1 = 1 / abs(Lw[np.argmin(abs(ph + np.pi))])
    cases.append(("slow integral, Ti = 8T", 1.0, 1.0, [T], g1 / 1.3, Ti))
    rng = np.random.default_rng(0)
    for lab, K, L, lags, Kc, Ti in cases:
        sc = max(L, 1.0); dt = sc / 100
        e = sim(K, L, lags, Kc, Ti, dt, 80 * sc + 15 * max(lags) + 10 * Ti)
        wg = np.linspace(1e-3, 3 / sc, 1500)
        Mt = np.max(abs(S_true(K, L, lags, Kc, Ti, np.logspace(-4, 1, 100000))))
        Mr = np.max(abs(S_rec(e, dt, wg)))
        Mn = np.max(abs(S_rec(e + 0.01 * rng.standard_normal(len(e)), dt, wg)))
        print(f"  {lab:24} Ms true {Mt:.3f}   from record {Mr:.3f} ({100 * (Mr - Mt) / Mt:+.1f}%)"
              f"   with 1% noise {Mn:.3f}")


def battery():
    """Section 8: SPIN battery plants tuned by SIMC (half rule, tau_c = theta)."""
    from scipy.signal import tf2ss, cont2discrete
    _Ms = Ms
    Ms_k = lambda k: _Ms(min(k, 1.55))
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

    print(f"{'plant':9}{'ctrl':4}{'theta':>7}{'GM true':>9}{'GM read':>9}{'Ms true':>9}{'Ms read':>9}"
          f"{'err Ms':>8}{'resid':>9}{'L_d read':>9}{'cross':>6}")
    rows=[]
    for name,(K,L,lags) in PLANTS.items():
        for order in ([1] if name.startswith('F') else [1,2]):
            taus,theta=half_rule(lags,L,order)
            Kc,Ti,Td=simc(K,taus,theta,order); num,den=C_tf(Kc,Ti,Td)
            GM,Mst=freq(K,L,lags,num,den)
            scale=max(theta,1.0); dt=scale/200
            e=simulate(K,L,lags,num,den,dt,40*scale+10*max(lags))
            Ld,k,r=read(e,dt,theta); GMr=math.pi/(2*k); Msr=Ms_k(min(k,1.55))
            n=crossings(e)
            rows.append((name,order,theta,GM,GMr,Mst,Msr,r,Ld,n))
            print(f"{name:9}{'PI' if order==1 else 'PID':4}{theta:7.2f}{GM:9.3f}{GMr:9.3f}{Mst:9.3f}{Msr:9.3f}"
                  f"{100*(Msr-Mst)/Mst:+7.1f}%{r:9.1e}{Ld:9.2f}{n:6d}")



if __name__ == "__main__":
    main()
