"""
SPIN, original rule versus certified rule, on chosen plants, with a log of
which band stops the gains in the final limit cycle.

    original : per-axis counts (SPIN Def. 1), limits (0.5, 0.75, 1.0)
    certified: whitened counts, limits (0.5, 1.0, 1.5), acceptance test M* = 1.7

The controller is simulated finely (PID: Tf/20, PI: Ta/1000) so the running
loop matches its continuous design; counts are taken at SPIN's usual
resolution (1500 samples per 15 apparent time constants).

Run:  python3 designed.py <worker> <n_workers>     results -> designed_<worker>.jsonl
"""
import json, math, sys
import numpy as np
import spin_run as S
from compare import counts_old, load_iae
from spin_cert_run import sim, elasticities

FAMILY_LAGS = [1.0, 0.5, 0.25]
PLANTS = [(f"F L={L}", 1.0, L, FAMILY_LAGS, True) for L in [0.25, 0.5, 1, 2, 4, 8, 16]] + [
    ("P1 PID", 1.0, 1.0, [10, 1, 1, 1], True),
    ("P2 PID", 1.25, 8.0, [5, 5, 5, 5], True),
    ("P3 PID", 1.0, 10.0, [2, 1, 1, 1], True),
    ("P4 PID", 1.0, 4.0, [8] * 6, True),
    ("P1 PI", 1.0, 1.0, [10, 1, 1, 1], False),
    ("P2 PI", 1.25, 8.0, [5, 5, 5, 5], False),
    ("P3 PI", 1.0, 10.0, [2, 1, 1, 1], False),
    ("P4 PI", 1.0, 4.0, [8] * 6, False),
    ("delay-dominant PI", 1.0, 5.0, [1.0], False),
    ("lag-dominant PI", 1.0, 1.0, [10.0], False),
    ("lag-dominant 2-lag PI", 1.0, 0.5, [10.0, 1.0], False),
    ("fast PID", 1.0, 0.1, [1.0, 0.5], True),
]
import os as _os
# SPIN's own multiplier box is [0.001, 10]; the paper removes it (the limits alone decide).
# Set SPIN_BOX=1 to run with the box.
FMIN, FMAX = (1e-3, 10.0) if _os.environ.get("SPIN_BOX") else (1e-4, 1e4)
RULES = {"original": (counts_old, (0.5, 0.75, 1.0), None),
         "certified": (S.counts, (0.5, 1.0, 1.5), 1.7),
         # ablation: the original counts and limits, with only the acceptance test added
         "original+test": (counts_old, (0.5, 0.75, 1.0), 1.7),
         # no counts at all: the same rule driven by the M_s read from the step alone
         "ms-only": (None, None, 1.7),
         # counts replaced by band-wise M_s: the peak of |S| read from the step over the
         # frequencies where band k dominates |C| (largest elasticity); limits per band
         # SPIN counts and limits shape the gain ratios; if the step reads M_s > M*, all
         # gains are lowered together by one step (uniform back-off, no band attribution)
         "original+uniform": (counts_old, (0.5, 0.75, 1.0), 1.7),
         # SPIN counts; if the step reads M_s > M*, predict from the same record the M_s after
         # each SPIN move (lower band k, raise the bands below it) and after a uniform back-off,
         # using L = 1/S - 1 and G = L/C; apply the band move with the lowest prediction if it
         # lowers M_s, else back off uniformly
         "original+shape": (counts_old, (0.5, 0.75, 1.0), 1.7),
         # same, but among the band moves predicted to lower M_s take the highest band: its move
         # raises the lower gains (K_i last), which costs the least load rejection
         "original+shape-hi": (counts_old, (0.5, 0.75, 1.0), 1.7),
         # Algorithm 2 of the paper: accept iff the certified upper bound of Theorem 1 is <= M*;
         # otherwise predict from the same record (discrete loop, exact) the M_s after each SPIN band
         # move, apply the highest band predicted to lower it (at most NMAX in a row), else back off
         # all gains uniformly
         # the count limits of cert-shape can be set with CERT_LIMITS (default: SPIN's published limits)
         "cert-shape": (counts_old, tuple(float(x) for x in _os.environ.get("CERT_LIMITS", "0.5,0.75,1.0").split(",")), 1.7),
         # certify before applying (Theorem 3): a move is applied only if the record of the running
         # controller certifies the next one (stable, M_s <= M*); else the next candidate (band moves,
         # uniform back-off), else hold. Before the first certified tuning, back off uniformly.
         "cert-ahead": (counts_old, tuple(float(x) for x in _os.environ.get("CERT_LIMITS", "0.5,0.75,1.0").split(",")), 1.7),
         "ms-bands": ("bands", tuple(float(x) for x in _os.environ.get("MS_LIMITS", "1.7,1.7,1.7").split(",")), None)}


def S_rec(e, dt, w):
    """S(jw) = sum of the error increments times exp(-jwt); chunked to bound memory."""
    d = np.diff(np.concatenate([[0.0], e])); t = np.arange(len(d)) * dt
    out = np.empty(len(w), complex); step = max(1, int(2e7 // len(d)))
    for i in range(0, len(w), step):
        out[i:i + step] = np.exp(-1j * np.outer(w[i:i + step], t)) @ d
    return out


def loop_fr(K, L, lags, Ki, Kp, Kd, w):
    s = 1j * w
    C = Kp + Ki / s + (Kd * s / ((Kd / Kp / 10) * s + 1) if Kd > 0 else 0)
    G = K * np.exp(-s * L)
    for t in lags: G = G / (t * s + 1)
    return C * G


def C_fr(Ki, Kp, Kd, w):
    s = 1j * w
    return Kp + Ki / s + (Kd * s / ((Kd / Kp / 10) * s + 1) if Kd > 0 else 0)


def shape_move(e, dt, Ta, Ki0, Kp0, Kd0, F, nb, g, Mstar, prefer_high=False):
    """Read S from the step; if M_s > M*, return (move, info). move: band index or 'all'."""
    w = np.logspace(math.log10(0.01 / Ta), math.log10(0.3 * math.pi / dt), 600)
    Sw = S_rec(e, dt, w); i = int(np.argmax(abs(Sw)))
    wf = np.linspace(w[max(i - 1, 0)], w[min(i + 1, len(w) - 1)], 41)
    Sf = S_rec(e, dt, wf)
    w = np.concatenate([w, wf]); Sw = np.concatenate([Sw, Sf]); o = np.argsort(w); w, Sw = w[o], Sw[o]
    i = int(np.argmax(abs(Sw)))
    read = float(abs(Sw[i]))
    if read <= Mstar: return None, dict(read=read)
    K0 = np.array([Ki0, Kp0, Kd0]); C = C_fr(*(K0 * F), w); L = 1 / Sw - 1
    # first-order robustness gradient g_k = -Re(T c_k / C) at the peak, c_k = K_k dC/dK_k
    T = 1 - Sw[i]; grad = []
    for k in range(nb):
        Fp = F.copy(); Fp[k] *= math.exp(1e-6)
        ck = (C_fr(*(K0 * Fp), w[i:i + 1])[0] - C[i]) / 1e-6
        grad.append(float(-(T * ck / C[i]).real))
    pred = {}
    for k in range(nb):
        Fn = F.copy(); Fn[:k] *= g; Fn[k] /= g
        pred[k] = float(np.max(abs(1 / (1 + L * C_fr(*(K0 * Fn), w) / C))))
    pred["all"] = float(np.max(abs(1 / (1 + L * C_fr(*(K0 * F / g), w) / C))))
    kb = min(range(nb), key=lambda k: pred[k])
    move = kb if pred[kb] < read else "all"
    if prefer_high:
        down = [k for k in range(nb) if pred[k] < read]
        move = max(down) if down else "all"
    dk = [grad[k] - sum(grad[:k]) for k in range(nb)]
    return move, dict(read=read, grad=grad, pred=pred, ReS=float(Sw[i].real),
                      grad_pick=int(np.argmax(dk)), grad_ok=bool(max(dk) > 0))


TAU, ETA, NMAX = 1e-3, 2e-3, 10


def cert_move(e, dt, Ki0, Kp0, Kd0, F, nb, g, Mstar):
    """Algorithm 2, lines 7a-7d. Returns (move or None, info)."""
    from cert import certificate, S_grid, increments, C_d
    c = certificate(e, TAU, ETA, Mhf=Mstar)
    if c["upper"] <= Mstar: return None, c
    a = increments(e); S, h = S_grid(a, c["th1"], c["M"])
    th = np.arange(c["M"]) * h; th[0] = 1e-3 * h                 # avoid z = 1 (integrator)
    S = np.where(np.arange(c["M"]) == 0, S_grid(a, h * 1e-3, 2)[0][1], S)
    z = np.exp(1j * th); K0 = np.array([Ki0, Kp0, Kd0])
    C = C_d(*(K0 * F), dt, z); Lr = 1 / S - 1
    pred = {}
    for k in range(nb):
        Fn = F.copy(); Fn[:k] *= g; Fn[k] /= g
        pred[k] = float(np.max(abs(1 / (1 + Lr * C_d(*(K0 * Fn), dt, z) / C))))
    down = [k for k in range(nb) if pred[k] < c["read"]]
    c["pred"] = pred
    return (max(down) if down else "all"), c


def run(K, L, lags, pid, rule, iters=200, beta=0.1):
    counter, limits, Mstar = RULES[rule]
    Ta = L + sum(lags); g = 1 / (1 - beta); nb = 3 if pid else 2
    l = sorted(lags, reverse=True) + [0.0]
    tau = l[0] + l[1] / 2; th = L + l[1] / 2 + sum(l[2:])
    Kp0 = tau / (K * 4 * th); Ki0 = Kp0 / tau; Kd0 = Kp0 * l[1] * 0.5 if pid else 0.0
    sc = float(_os.environ.get("START_SCALE", "1"))   # e.g. 2 = SIMC with tau_c = theta
    Kp0, Ki0, Kd0 = sc * Kp0, sc * Ki0, sc * Kd0
    F = np.ones(3); acc = []; log = []; shp = []; dts = 15 * Ta / 1500; streak = 0
    FIXED = bool(_os.environ.get("FIXED_TS")); dt0 = None; applied = []; safe = False; bad = set(); F_cert = None
    for it in range(iters):
        Ki, Kp, Kd = Ki0 * F[0], Kp0 * F[1], Kd0 * F[2]
        dt = min(dts, (Kd / Kp / 10) / 20) if pid else Ta / 1000
        if FIXED:                                      # a real controller keeps its sampling period
            dt0 = dt if dt0 is None else dt0; dt = dt0
        n = int(15 * Ta / dt); e = sim(K, L, lags, Ki, Kp, Kd, dt, n)
        unstable = bool(abs(e[int(0.8 * n):]).max() > 0.5 * abs(e).max())
        applied.append(dict(it=it, gains=[Ki, Kp, Kd], dt=dt, unstable=unstable, safe=safe))
        if unstable:
            F = np.clip(F / np.array([2, 4, 8]), FMIN, FMAX)
            if it >= iters - 60: log.append("unstable")
            continue
        step = max(1, int(round(dts / dt)))
        if counter is None: V = []; tag = None; read = None
        elif counter == "bands":
            # whole band up to the sampling limit (a derivative-band peak can lie far above
            # 3/L), then refine around the largest value in each band so sharp peaks are not missed
            w = np.logspace(math.log10(0.01 / Ta), math.log10(0.3 * math.pi / dt), 600)
            A = abs(S_rec(e, dt, w)); tag = "cert"
            own = np.argmax(np.array([elasticities(Ki, Kp, Kd, x)[:nb] for x in w]), axis=1)
            Mk = []
            for k in range(nb):
                idx = np.where(own == k)[0]
                if not len(idx): Mk.append(0.0); continue
                i = idx[np.argmax(A[idx])]
                wf = np.linspace(w[max(i - 1, 0)], w[min(i + 1, len(w) - 1)], 41)
                Af = abs(S_rec(e, dt, wf))
                of = np.argmax(np.array([elasticities(Ki, Kp, Kd, x)[:nb] for x in wf]), axis=1)
                Mk.append(float(max(A[i], Af[of == k].max() if np.any(of == k) else 0.0)))
            read = max(Mk)
            V = [k for k in range(nb) if Mk[k] > limits[k]]
        else:
            N = counter(e[::step], dt * step)
            V = [k for k in range(nb) if N[k] > limits[k]]; tag = None; read = None
        if rule == "cert-ahead":
            from cert import certificate
            from cert_ahead import ahead
            c = certificate(e, TAU, ETA, Mhf=Mstar)
            if c["upper"] > Mstar:
                if not safe:                           # before the first certified tuning: back off
                    F = np.clip(F / g, FMIN, FMAX)
                    if it >= iters - 60: log.append("warm")
                    continue
                # certified ahead but its own record does not certify it: return to the last
                # tuning certified by its own record and never propose this one again
                bad.add(tuple(np.round(np.log(F), 6))); F = F_cert.copy()
                if it >= iters - 60: log.append("fallback")
                continue
            safe = True; applied[-1]["safe"] = True; F_cert = F.copy()
            if it >= iters - 60: acc.append((Ki, Kp, Kd, c["upper"]))
            K0 = np.array([Ki0, Kp0, Kd0]); cand = []
            Fp = F.copy()
            if not V: Fp[:nb] *= g; tagp = "raise"
            else:
                km = min(V); Fp[:km] *= g; Fp[km] /= g; tagp = f"G{km}"
            cand.append((tagp, Fp))
            for k in reversed(range(nb)):
                Fn = F.copy(); Fn[:k] *= g; Fn[k] /= g; cand.append((f"band{k}", Fn))
            cand.append(("back", F / g))
            moved = False
            for tag_, Fn in cand:
                Fn = np.clip(Fn, FMIN, FMAX)
                if np.allclose(Fn, F) or tuple(np.round(np.log(Fn), 6)) in bad: continue
                A = ahead(e, dt, (Ki, Kp, Kd), tuple(K0 * Fn), TAU, Mstar, ETA)
                if A["stable"] and A["U"] <= Mstar:
                    F = Fn; moved = True
                    if it >= iters - 60: log.append(f"ahead:{tag_}")
                    break
            if not moved and it >= iters - 60: log.append("hold")
            continue
        if not V and rule == "cert-shape":
            mv, info = cert_move(e, dt, Ki0, Kp0, Kd0, F, nb, g, Mstar)
            read = max(info["read"] + info["grid_term"] + TAU, info["up_hf"])     # certified upper bound
            if mv is not None:
                streak = streak + 1 if mv != "all" else streak
                if streak > NMAX: mv = "all"
                Fn = F.copy()
                if mv == "all": Fn = Fn / g
                else: Fn[:mv] *= g; Fn[mv] /= g
                Fn = np.clip(Fn, FMIN, FMAX)
                if it >= iters - 60: log.append(f"shape{mv}")
                shp.append(dict(it=it, move=mv, read=info["read"], upper=info["upper"],
                                pred=(info["pred"][mv] if mv != "all" else None)))
                F = Fn; continue
            streak = 0
        if not V and rule in ("original+shape", "original+shape-hi"):
            mv, info = shape_move(e, dt, Ta, Ki0, Kp0, Kd0, F, nb, g, Mstar,
                                  prefer_high=(rule == "original+shape-hi")); read = info["read"]
            if mv is not None:
                Fn = F.copy()
                if mv == "all": Fn = Fn / g
                else: Fn[:mv] *= g; Fn[mv] /= g
                Fn = np.clip(Fn, FMIN, FMAX)
                if it >= iters - 60: log.append(f"shape{mv}")
                wt = np.logspace(-4, 2.5, 20000)
                true_new = float(np.max(abs(1 / (1 + loop_fr(K, L, lags, *(np.array([Ki0, Kp0, Kd0]) * Fn), wt)))))
                shp.append(dict(it=it, move=mv, read=info["read"], pred=info["pred"][mv], true_new=true_new,
                                grad=info["grad"], grad_pick=info["grad_pick"], grad_ok=info["grad_ok"], ReS=info["ReS"]))
                F = Fn; continue
        if not V and Mstar is not None and rule not in ("original+shape", "original+shape-hi", "cert-shape"):
            wmax = min(3 / L, 0.3 * math.pi / dt)
            w = np.logspace(math.log10(0.01 / Ta), math.log10(wmax), 500)
            Sw = S_rec(e, dt, w); i = int(np.argmax(abs(Sw))); read = float(abs(Sw[i]))
            if abs(Sw[i]) > Mstar:
                if rule == "original+uniform": V = ["all"]; tag = "cert"
                else: V = [int(np.argmax(elasticities(Ki, Kp, Kd, w[i])[:nb]))]; tag = "cert"
        if V == ["all"]:                                   # uniform back-off
            if it >= iters - 60: log.append("certU")
            F = np.clip(F / g, FMIN, FMAX); continue
        kmin = min(V) if V else 3
        if it >= iters - 60:
            log.append("none" if kmin == 3 else (f"cert{kmin}" if tag else f"G{kmin}"))
            if kmin == 3: acc.append((Ki, Kp, Kd, read))
        for k in range(nb):
            if k < kmin: F[k] *= g
            elif k == kmin: F[k] /= g
        F = np.clip(F, FMIN, FMAX)
    from cert import Ms_d as _Ms_d
    cache = {}
    for a_ in applied:
        key = tuple(round(x, 12) for x in a_["gains"]) + (round(a_["dt"], 12),)
        if key not in cache:
            cache[key] = math.inf if a_["unstable"] else _Ms_d(K, L, lags, *a_["gains"], a_["dt"])[0]
        a_["Ms_d"] = cache[key]
    traj = dict(n=len(applied), above=sum(a_["Ms_d"] > 1.7 for a_ in applied),
                above_safe=sum(a_["Ms_d"] > 1.7 for a_ in applied if a_["safe"]),
                unstable=sum(a_["unstable"] for a_ in applied),
                max_Ms=max(a_["Ms_d"] for a_ in applied),
                first_safe=next((a_["it"] for a_ in applied if a_["safe"]), None),
                Ms_it=[a_["Ms_d"] for a_ in applied])
    if not acc: return dict(worst=None, traj=traj)
    # SPIN ends in a limit cycle; every tuning it accepts in the final cycle is a
    # candidate output. Report all of them and the most aggressive one.
    w = np.logspace(-4, 2.5, 80000); seen = {}
    for Ki, Kp, Kd, read in acc:
        key = (round(Ki, 12), round(Kp, 12), round(Kd, 12))
        if key in seen: continue
        Lw = loop_fr(K, L, lags, Ki, Kp, Kd, w)
        ph = np.unwrap(np.angle(Lw)); GM = 1 / abs(Lw[np.argmin(abs(ph + np.pi))])
        from cert import Ms_d
        dtk = dt0 if FIXED else (min(dts, (Kd / Kp / 10) / 20) if pid else Ta / 1000)
        seen[key] = dict(gains=[Ki, Kp, Kd], Ms=float(np.max(1 / abs(1 + Lw))), GM=float(GM), read=read,
                         Ms_d=Ms_d(K, L, lags, Ki, Kp, Kd, dtk)[0], dt=dtk)
    cand = list(seen.values()); worst = max(cand, key=lambda c: c["Ms"])
    worst["IAE"] = float(load_iae(K, L, lags, *worst["gains"]))
    hist = {k: log.count(k) for k in sorted(set(log))}
    return dict(worst=worst, Ms_all=sorted(c["Ms"] for c in cand), accepted=cand,
                fired=hist, F=[float(x) for x in F], shape_log=shp, traj=traj)


if __name__ == "__main__":
    wk, nw = int(sys.argv[1]), int(sys.argv[2])
    cost = lambda p: (3 if p[4] else 0.05) * (1 + len(p[3]))
    order = sorted(PLANTS, key=cost, reverse=True)            # balance the workers
    mine = [p for i, p in enumerate(order) if i % nw == wk]
    import os
    only = os.environ.get("RULES_ONLY")                        # e.g. RULES_ONLY="original+test"
    rules = [r for r in RULES if not only or r in only.split(",")]
    jobs = [(p, r) for p in mine for r in rules]
    out = os.environ.get("OUT_PREFIX", "designed")
    with open(f"{out}_{wk}.jsonl", "a") as f:
        for (name, K, L, lags, pid), rule in jobs:
            res = run(K, L, lags, pid, rule)
            f.write(json.dumps(dict(plant=name, K=K, L=L, lags=lags, pid=pid, rule=rule, res=res)) + "\n")
            f.flush()
            print(name, rule, None if res is None or res.get("worst") is None else
                  (f"Ms accepted {res['Ms_all'][0]:.3f}-{res['Ms_all'][-1]:.3f}", res["fired"]), flush=True)
    print("DONE", flush=True)
