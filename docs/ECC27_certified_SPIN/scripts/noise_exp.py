"""For proposal.md (follow-up, not the ECC paper). Noise term of the record-based |S| estimate: S^ = sum a_n e^{-jn th}, a_n = diff(e).
Measurement noise v enters as e = S (r - v), so S^ = S_N + D_N, D_N = DFT of diff(e_w), e_w = -S v.
Prediction (relative, near frequency w): |D_N|/|S| ~ w * sqrt(T_rec * Phi_v(w)) / A."""
import sys, math, numpy as np
sys.path.insert(0, "/home/claude/robopid-simulator/docs/ECC27_certified_SPIN/scripts")
from cert import L_d

K, L, lags = 1.0, 10.0, [2.0, 1.0, 1.0, 1.0]
Ki, Kp, Kd = 0.0516, 0.4113, 0.25            # final basic-SPIN tuning of P3 PID (Fig. 1)
dt = 0.02                                      # coarser than the paper (0.0025) for speed

def loop(r, v):
    """error e = r - (y + v) of the digital PID loop of the experiment (derivative on e, Tf = Td/10)."""
    n = len(r); nd = int(round(L / dt)); a = [math.exp(-dt / t) for t in lags]; x = [0.0] * len(lags)
    Tf = Kd / Kp / 10; af = math.exp(-dt / Tf); ub = np.zeros(n + nd + 1); e = np.zeros(n); I = xf = y = 0.0
    for i in range(n):
        ek = r[i] - (y + v[i]); e[i] = ek
        D = (Kd / Tf) * (ek - xf); xf = af * xf + (1 - af) * ek
        u = Kp * ek + Ki * I + D; I += ek * dt
        ub[i + nd] = K * u; vv = ub[i]
        for j, aj in enumerate(a): x[j] = aj * x[j] + (1 - aj) * vv; vv = x[j]
        y = vv
    return e

def ar1(n, sigma, tau_n, rng):
    phi = math.exp(-dt / tau_n); w = rng.standard_normal(n) * sigma * math.sqrt(1 - phi**2)
    v = np.zeros(n); v[0] = sigma * rng.standard_normal()
    for i in range(1, n): v[i] = phi * v[i-1] + w[i]
    return v

def S_hat(e, th):
    a = np.diff(np.concatenate([[0.0], e])); n = np.arange(len(a))
    return np.array([np.sum(a * np.exp(-1j * t * n)) for t in th])

w = np.linspace(0.03, 0.6, 300); th = w * dt
S_true = 1 / (1 + L_d(K, L, lags, Ki, Kp, Kd, dt, th)); Ms = abs(S_true).max(); wp = w[np.argmax(abs(S_true))]
rng = np.random.default_rng(1)
sigma, tau_n, A = 0.01, 0.5, 1.0                # noise std 1 % of the step, filter 0.5 s
Phi = lambda om: 2 * sigma**2 * tau_n / (1 + (om * tau_n)**2)   # two-sided PSD of v
print(f"true M_s {Ms:.3f} at w = {wp:.3f} rad/s")
for T_rec in (225.0, 100.0):
    n = int(T_rec / dt); r = np.ones(n); e0 = loop(r, np.zeros(n)); S0 = S_hat(e0, th)
    errs, peaks = [], []
    for k in range(40):
        ew = loop(np.zeros(n), ar1(n, sigma, tau_n, rng))          # noise-only part (superposition)
        D = S_hat(ew, th); Sn = S0 + D
        errs.append(np.max(abs(D) / abs(S_true))); peaks.append(abs(Sn).max())
        if k == 0: rel_at_peak = []
        rel_at_peak.append(abs(D[np.argmax(abs(S_true))]) / Ms)
    pred = wp * math.sqrt(T_rec * Phi(wp)) / A
    print(f"T_rec {T_rec:5.0f} s: clean max|S^| {abs(S0).max():.3f};  noisy max|S^| mean {np.mean(peaks):.3f} "
          f"sd {np.std(peaks):.3f} max {np.max(peaks):.3f};  rel. error at peak rms {math.sqrt(np.mean(np.square(rel_at_peak))):.4f} "
          f"(predicted {pred:.4f});  sup over band: median {np.median(errs):.3f}, max {np.max(errs):.3f}")
    # averaging K records (independent noise)
    Kavg = 4; peaks4 = []
    for k in range(20):
        D = sum(S_hat(loop(np.zeros(n), ar1(n, sigma, tau_n, rng)), th) for _ in range(Kavg)) / Kavg
        peaks4.append(abs(S0 + D).max())
    print(f"        averaged over {Kavg} steps: max|S^| sd {np.std(peaks4):.3f}")
