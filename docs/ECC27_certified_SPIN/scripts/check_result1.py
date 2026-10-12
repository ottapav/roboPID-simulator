"""Numerical confirmation of the Result 1 statements (lag-cancelled loop, L_d = 1)."""
import math, numpy as np
from scipy.optimize import brentq
from lagcancel import records, Ms as Ms_brute

def Ms_param(k):
    """Prop. 1: unique minimiser x* in (0, pi/2) of |1+L|, k = x sin x - x^2 cos x"""
    x = brentq(lambda x: x*math.sin(x) - x*x*math.cos(x) - k, 1e-9, math.pi/2)
    return 1/math.sqrt(1 - 2*k*math.sin(x)/x + k*k/(x*x)), x
print("Ms param vs brute:")
for k in [0.2, 1/math.e, 0.5, 0.7, 1.0, 1.4, 1.5]:
    mp, x = Ms_param(k); print(f"  k={k:.4f} param {mp:.6f} brute {Ms_brute(k):.6f} x*={x:.4f}")
ks = np.linspace(0.01, 1.56, 300); m = [Ms_param(k)[0] for k in ks]
print("  strictly increasing:", bool(np.all(np.diff(m) > 0)))

def r(k): return k*k/2 if k <= 1 else k - 0.5
print("Extremum ratio bound |E_{i+1}| <= r(k)|E_i|  (E_0 = 1):")
ks = np.linspace(0.38, 1.49, 112)
t, E = records(ks, dt=1e-3, tend=200.0)
worst = 0; okall = True; lobes_ok = True; half_ok = True
for j, k in enumerate(ks):
    e = E[:, j]; s = np.sign(e); z = np.where(s[1:] * s[:-1] < 0)[0]          # crossings
    if len(z) < 3: continue
    ext = [1.0] + [e[z[i]:z[i+1]+1][np.argmax(abs(e[z[i]:z[i+1]+1]))] for i in range(len(z)-1)]
    ext = np.abs(ext); ratios = ext[1:] / ext[:-1]
    sig = ext > 1e-6                                                    # above round-off
    rr = ratios[sig[1:]] if np.any(sig[1:]) else ratios[:1]
    worst = max(worst, float(np.max(rr / r(k)))); okall &= bool(np.all(rr <= r(k) * (1 + 1e-3)))
    # lobe areas strictly decreasing
    A = [np.sum(e[:z[0]+1])] + [abs(np.sum(e[z[i]+1:z[i+1]+1])) for i in range(len(z)-1)]
    A = np.array(A)[: int(np.sum(sig))]
    lobes_ok &= bool(np.all(np.diff(A) < 0))
    # half turns of Gamma_0 between crossings, in raw and in whitened coordinates
    zt = np.cumsum(e) * 1e-3; zt = zt - zt[-1]
    nz = min(len(z), int(np.sum(sig))) - 1
    for M in (np.eye(2), np.array([[2.0, 0.7], [0.3, 0.5]])):
        def at(i):                                   # exact crossing by linear interpolation
            f = e[z[i]] / (e[z[i]] - e[z[i]+1]); return M @ np.array([0.0, zt[z[i]] + f*(zt[z[i]+1]-zt[z[i]])])
        for i in range(nz):
            P = np.hstack([at(i)[:, None], M @ np.vstack([e[z[i]+1:z[i+1]+1], zt[z[i]+1:z[i+1]+1]]), at(i+1)[:, None]])
            ang = np.unwrap(np.arctan2(P[1], P[0])); d = (ang[-1] - ang[0]) / (2*math.pi)
            if abs(d - 0.5) > 1e-6: half_ok = False; print("   k", round(k,3), "i", i, d)
print(f"  max observed ratio / r(k) = {worst:.4f}  (bound holds: {okall})")
print("  lobe areas strictly decreasing:", lobes_ok)
print("  half turn between crossings (raw and linear map):", half_ok)
print("  r(k)<1 for k<3/2; Ms at k=1/e, 1, 3/2:", [round(Ms_param(k)[0], 4) for k in (1/math.e, 1.0, 1.5)])
import spin_run as SR
print("Corollary: whitened count N0 in ((n-1)/2, (n+1)/2), n = crossings before settling (delta = 0.02):")
ks = np.linspace(0.2, 1.45, 126); t, E = records(ks, dt=1e-3, tend=120.0); ok = True; step = 10
for j, k in enumerate(ks):
    e = E[::step, j]; N0 = SR.counts(e, 1e-2, 0.02)[0]
    kd = min(len(e), np.where(abs(e) > 0.02 * abs(e).max())[0][-1] + 2); s = np.sign(e[:kd]); s = s[s != 0]
    n = int(np.sum(s[1:] != s[:-1]))
    if not ((n - 1) / 2 < N0 < (n + 1) / 2): ok = False; print("   fail", round(k, 3), n, N0)
print("  holds on all", len(ks), "values of k:", ok)
