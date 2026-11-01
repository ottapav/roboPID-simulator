"""
Every number and figure of the limit-cycle paper, from the current code.

Setting: lc_setup.py (linear loop, derivative on the error, app defaults).
Run from the repository root:
    python docs/ECC27_limit_cycle/scripts/lc_results.py
Writes docs/ECC27_limit_cycle/lc_results.json and fig_index.pdf.

Sections of the output, by paper section:
  orbits      terminal orbits from 27 starts per plant (AMIGO x 8^{-1,0,1}):
              period, move counts, applied moves all table moves, half-spans,
              distinct orbits, M_s; for the derivative on the error and, for
              comparison, on the measurement (Sec. Robustness)
  fidelity    smooth count vs hard count on the states the rule visits:
              hard limit, share of identical moves for p = 8, 32, 128, and the
              orbits reached with the smooth count driving the rule
  event       where the orbits sit: whether the window end k_delta differs
              between the points of an orbit, which side the e-points are on,
              and the number of error extrema above delta * peak on each side
  limit       fixed step beta -> 0 from the AMIGO start, hard and smooth:
              period, orbit radius, and the distance between the orbit centres
              of successive beta
  schedule    step halved once every band has reversed (drift meter): the
              experiments until the step is below 1e-3 of the initial one, and
              the moves applied over the last 16 experiments
  spread      range of the distinct orbit centres per coordinate; Jacobian
              singular-value ratio of the continuous count at the centres
  meter       singular values of M^T on the share-difference subspace
  robust      M_s on the all-quiet states the rule visits
  line        hard and smooth count along a line in gain space (fig_index)
"""
from __future__ import annotations

import json
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings('ignore')

from lc_setup import (PLANTS, NBAR, LO, HI, MOVES, DESC, Loop, step, applied_move,  # noqa: E402
                      run_to_orbit, grid_starts, h_of, ROOT)
from core.features import compute_features                             # noqa: E402
from regindex_soft import soft_counts                                  # noqa: E402
from core.signals import add_derivatives                               # noqa: E402

H = h_of(0.1)
OUT = ROOT / 'docs' / 'ECC27_limit_cycle'
TABLE = ('e', 'd0', 'd1', 'd2')


def midrange(o):
    return 0.5 * (o.max(axis=0) + o.min(axis=0))


def cluster(cents, tol):
    groups = []
    for i, c in enumerate(cents):
        for g in groups:
            if np.linalg.norm(c - g['c']) < tol:
                g['idx'].append(i); g['c'] = np.mean([cents[j] for j in g['idx']], axis=0); break
        else:
            groups.append({'c': c.copy(), 'idx': [i]})
    return groups


def orbit_stats(L, starts):
    runs = []
    for th0 in starts:
        r = run_to_orbit(L, th0)
        o = r['orbit']
        cnt = Counter(r['moves'])
        runs.append(dict(n=r['n'], period=len(o), counts=[cnt.get(m, 0) for m in TABLE],
                         other=len(o) - sum(cnt.get(m, 0) for m in TABLE),
                         halfspan_h=((o.max(axis=0) - o.min(axis=0)) / 2 / H).tolist(),
                         centre=midrange(o).tolist(), centroid=o.mean(axis=0).tolist(),
                         Ms=max(L.Ms(t) for t in o),
                         Kd_floor=bool(np.any(np.isclose(o[:, 2], LO))),
                         at_ceiling=bool(np.any(np.isclose(o, HI))),
                ceil_coord=np.any(np.isclose(o, HI), axis=0).tolist()))
    return runs


def summarize(runs):
    g = cluster([np.array(r['centroid']) for r in runs], 2 * H)
    Ki = np.exp([r['centroid'][0] for r in runs])
    ne = [r['period'] // 8 for r in runs if not r['at_ceiling']]
    off = [r for r in runs if not r['at_ceiling']]
    balanced = all(r['other'] == 0 and r['period'] % 8 == 0 and
                   r['counts'] == [r['period'] // 8 * c for c in (1, 4, 2, 1)] for r in off)
    within = all(np.all(np.array(r['halfspan_h']) <= np.array([2, 1, 0.5]) * (r['period'] // 8) + 1e-9)
                 for r in off)
    return dict(n_orbits=len(g), cluster_sizes=sorted([len(x['idx']) for x in g], reverse=True),
                Ki=[float(Ki.min()), float(Ki.max())], Ms=[min(r['Ms'] for r in runs), max(r['Ms'] for r in runs)],
                n_e=sorted(set(ne)), off_ceiling=len(off), all_balanced=balanced, all_within_bound=within,
                periods_ceiling=sorted(set(r['period'] for r in runs if r['at_ceiling'])),
                Kd_floor=sum(r['Kd_floor'] for r in runs), ceiling=sum(r['at_ceiling'] for r in runs),
                n_median=float(np.median([r['n'] for r in runs])),
                n_e_hist={str(k): v for k, v in sorted(Counter(ne).items())},
                ceiling_coords=[int(k) for k in range(3) if any(r['ceil_coord'][k] for r in runs)],
                table_only=sum(r['other'] == 0 for r in runs),
                centres=[x['c'].tolist() for x in g])


def encirc_v1(x, y, eps=0.1):
    """core.features.encirc as published before the correction: endpoint angle
    corrected by crossings of the negative x-axis with both segment ends at
    x < 0, over segments 0..j-2. Kept here to count where it was wrong."""
    from core.features import find_disc_entrypoint
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    mx, my = np.max(np.abs(x)), np.max(np.abs(y))
    if mx < 1e-12 or my < 1e-12:
        return 0.0
    x, y = x / mx, y / my
    j = min(find_disc_entrypoint(x, y, eps), len(x) - 1)
    if j <= 1:
        return 0.0
    angle = np.arctan2(y[j], x[j]) - np.arctan2(y[0], x[0])
    xp, xn, yp, yn = x[:j - 1], x[1:j], y[:j - 1], y[1:j]
    inn = (xp < 0) & (xn < 0)
    return angle / (2 * np.pi) - float(np.sum(inn & (yp < 0) & (yn >= 0))) + float(np.sum(inn & (yp >= 0) & (yn < 0)))


def spread(name, centres):
    """Range of the distinct orbit centres per coordinate, and the singular-value
    ratio of the continuous count's Jacobian (central differences, step 0.05)
    at the centres off the K_d bounds."""
    C = np.array(centres)
    Ls = Loop(name, 'soft', p=32.0)
    ratio = []
    for c in C:
        if np.min(np.abs(c[2] - np.array([LO, HI]))) < 0.5:
            continue
        J = np.array([(Ls.counts(c + 0.05 * e)[0] - Ls.counts(c - 0.05 * e)[0]) / 0.1 for e in np.eye(3)]).T
        sv = np.linalg.svd(J, compute_uv=False)
        ratio.append(float(sv[-1] / sv[0]))
    return dict(range=(C.max(axis=0) - C.min(axis=0)).tolist(), sv_ratio_median=float(np.median(ratio)),
                n_interior=len(ratio))


def visited(L, starts, n_steps=120):
    pts = []
    for th0 in starts:
        th = np.clip(th0, LO, HI)
        for _ in range(n_steps):
            pts.append(th.copy()); th, _ = step(L, th)
    return np.unique(np.round(np.array(pts), 10), axis=0)


def move_of(N, unst):
    if unst:
        return 'U'
    v = [k for k in range(3) if N[k] > NBAR[k]]
    return 'e' if not v else f'd{v[0]}'


def fidelity(name, starts, hard_runs):
    Lh = Loop(name, 'hard')
    pts = visited(Lh, starts)
    exact, same = 0, {8: 0, 32: 0, 128: 0}
    why = Counter()
    v1_wrong = 0
    v1_sizes = set()
    for t in pts:
        Nh, u = Lh.counts(t)
        s = Lh.signals(t)
        ext = add_derivatives({'e': s['e']})['e']
        Nl, parts = soft_counts(ext, s['e'], s['k1'], s['k2'], sharp=None, return_parts=True)
        Nv1 = np.array([f['N'] for f in compute_features(DESC, add_derivatives({'e': s['e']}),
                                                          s['k1'], s['k_delta'], metric=encirc_v1)])
        if np.max(np.abs(Nv1 - Nh)) > 1e-6:
            v1_wrong += 1
            v1_sizes.add(round(float(np.max(np.abs(Nv1 - Nh))), 6))
        if np.max(np.abs(Nl - Nh)) <= 1e-6:
            exact += 1
        else:
            b = int(np.argmax(np.abs(Nl - Nh))); pr = parts[b]
            if pr is not None and abs(abs(Nl[b] - Nh[b]) - 1) < 1e-6 and \
                    np.max(np.abs(pr['dphi']) * pr['S'] * pr['V']) > 3.0:
                why['origin'] += 1
            elif Nh[b] == 0.0 and abs(Nl[b]) < 0.05:
                why['j<=1'] += 1
            else:
                why['other'] += 1
        for p in same:
            Ns = soft_counts(ext, s['e'], s['k1'], s['k2'], sharp=float(p))
            same[p] += move_of(Ns, u) == move_of(Nh, u)
    Ls = Loop(name, 'soft', p=32.0)
    soft_runs = orbit_stats(Ls, starts)
    gap = [np.linalg.norm(np.array(a['centroid']) - np.array(b['centroid'])) / H
           for a, b in zip(hard_runs, soft_runs)]
    return dict(n_states=len(pts), hard_limit_exact=exact, mismatch=dict(why), encirc_v1_wrong=v1_wrong, encirc_v1_error_sizes=sorted(v1_sizes),
                same_orbit=float(np.mean(np.array(gap) < 0.5)),
                same_move={str(p): same[p] / len(pts) for p in same},
                orbit_gap_h_max=float(np.max(gap)), orbit_gap_h_median=float(np.median(gap)),
                soft_summary=summarize(soft_runs))


def overshoot(e):
    """Overshoot of the step response, relative to the largest error: the
    largest excursion of y above the setpoint (the record e = y - r starts at
    -1), divided by max |e|."""
    e = np.asarray(e, float)
    return float(max(0.0, e.max()) / np.max(np.abs(e)))


def event(name, starts, delta=0.02):
    """Where the orbits sit: the overshoot at the points of every terminal
    orbit, split by the move taken there."""
    L = Loop(name, 'hard')
    n_orb = n_straddle = 0
    os_e, os_d = [], []
    for th0 in starts:
        r = run_to_orbit(L, th0)
        n_orb += 1
        os_ = [overshoot(L.signals(t)['e']) for t in r['orbit']]
        n_straddle += min(os_) < delta < max(os_)
        for v, m in zip(os_, r['moves']):
            (os_e if m == 'e' else os_d).append(v)
    os_e, os_d = np.array(os_e), np.array(os_d)
    return dict(n_orbits=n_orb, straddle=int(n_straddle),
                e_below=float(np.mean(os_e < delta)), e_max=float(os_e.max()),
                d_above=float(np.mean(os_d > delta)), d_median=float(np.median(os_d)))


def limit(name, betas=(0.1, 0.05, 0.025, 0.0125, 0.00625)):
    rows = []
    for idx in ('hard', 'soft'):
        prev = None
        for b in betas:
            L = Loop(name, idx, p=32.0)
            r = run_to_orbit(L, L.amigo(), beta=b, n_iter=20000)
            o = r['orbit']; c = midrange(o)
            rows.append(dict(beta=b, h=h_of(b), index=idx, n=r['n'], period=len(o),
                             ceiling=bool(np.any(np.isclose(o, HI))),
                             radius=float(np.max(o.max(axis=0) - o.min(axis=0)) / 2),
                             centre=c.tolist(),
                             step_from_prev=None if prev is None else float(np.linalg.norm(c - prev))))
            prev = c
    return rows


def schedule(name, n_exp=600):
    out = {}
    for idx in ('hard', 'soft'):
        L = Loop(name, idx, p=32.0)
        th, beta = np.clip(L.amigo(), LO, HI), 0.1
        h0 = h_of(beta)
        seen_sign = [set(), set(), set()]
        hs, moves, ths, halvings = [], [], [], []
        for _ in range(n_exp):
            hs.append(h_of(beta)); ths.append(th.copy())
            th_new, row = step(L, th, beta)
            mv = 'U' if row == 'unstable' else applied_move(th, th_new, h_of(beta))
            moves.append(mv)
            if mv in TABLE:
                d = MOVES[mv]
                for k in range(3):
                    if d[k] != 0:
                        seen_sign[k].add(d[k])
                if all(len(x) == 2 for x in seen_sign):
                    beta = 1 - (1 - beta) ** 0.5          # halves h = log(1/(1-beta))
                    seen_sign = [set(), set(), set()]
                    halvings.append(len(hs))
            th = th_new
        hs = np.array(hs)
        reach = {str(f): (int(np.argmax(hs < f * h0)) if np.any(hs < f * h0) else None) for f in (1e-1, 1e-2, 1e-3)}
        last = np.array(ths[-16:])
        out[idx] = dict(experiments_to_step_fraction=reach, n_halvings=len(halvings),
                        last_halving=halvings[-1] if halvings else None,
                        Ms_at_step_fraction={k: (L.Ms(ths[v]) if v is not None else None) for k, v in reach.items()},
                        last_moves=dict(Counter(moves[-16:])),
                        final_spread=float(np.max(last.max(axis=0) - last.min(axis=0))),
                        final_gains=np.exp(ths[-1]).tolist(), final_Ms=L.Ms(ths[-1]))
    return out


def robust(name, starts):
    L = Loop(name, 'hard')
    pts = visited(L, starts, n_steps=60)
    rng = np.random.default_rng(0)
    pts = pts[rng.choice(len(pts), min(500, len(pts)), replace=False)]
    q, ms_all, r_all = [], [], []
    for t in pts:
        N, u = L.counts(t)
        if u:
            continue
        m = L.Ms(t); ms_all.append(m); r_all.append(float(np.max(N / NBAR)))
        if np.all(N <= NBAR):
            q.append(m)
    ra = np.argsort(np.argsort(r_all)); rm = np.argsort(np.argsort(ms_all))
    return dict(n_quiet=len(q), Ms_quiet_median=float(np.median(q)), Ms_quiet_max=float(np.max(q)),
                rank_corr=float(np.corrcoef(ra, rm)[0, 1]))


def line(name='P1'):
    """Hard and smooth counts, all three bands, along the uniform direction
    through the AMIGO orbit centre, +-1.5h, 1201 points."""
    Lh = Loop(name, 'hard')
    c = midrange(run_to_orbit(Lh, Lh.amigo())['orbit'])
    s = np.linspace(-1.5, 1.5, 1201) * H
    hard, soft, kd, os_ = [], [], [], []
    for si in s:
        th = c + si * np.ones(3)
        hard.append(Lh.counts(th)[0])
        sg = Lh.signals(th); kd.append(int(sg['k_delta'])); os_.append(overshoot(sg['e']))
        ext = add_derivatives({'e': sg['e']})['e']
        soft.append(soft_counts(ext, sg['e'], sg['k1'], sg['k2'], sharp=32.0))
    hard, soft = np.array(hard), np.array(soft)
    jumps = np.nonzero(np.max(np.abs(np.diff(hard, axis=0)), axis=1) > 0.05)[0]
    return dict(plant=name, s_h=(s / H).tolist(), hard=hard.tolist(), soft=soft.tolist(), k_delta=kd,
                jumps_at_h=[float(s[j] / H) for j in jumps],
                k_delta_changes_at_h=[float(s[j] / H) for j in range(len(kd) - 1) if kd[j + 1] != kd[j]],
                overshoot_crosses_delta_at_h=[float(s[j] / H) for j in range(len(os_) - 1)
                                             if (os_[j] - 0.02) * (os_[j + 1] - 0.02) < 0],
                max_jump_hard=float(np.max(np.abs(np.diff(hard, axis=0)))),
                max_step_soft=float(np.max(np.abs(np.diff(soft, axis=0)))))


def meter():
    M = np.array([MOVES[m] for m in TABLE])
    B = np.linalg.svd(np.ones((1, 4)))[2][1:].T
    sv = np.linalg.svd(M.T @ B, compute_uv=False)
    w = np.array([1 / 8, 1 / 2, 1 / 4, 1 / 8])
    return dict(sv=sv.tolist(), ratio=float(sv[0] / sv[-1]), Mtw=(M.T @ w).tolist())


def figures(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    ln = res['line']
    x = np.array(ln['s_h']); hard = np.array(ln['hard']) / NBAR; soft = np.array(ln['soft']) / NBAR
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    col = ('#6b2e9e', '#c77a06', '#1f5fa9')
    for k in range(3):
        ax.plot(x, hard[:, k], color=col[k], lw=0.8, alpha=0.45)
        ax.plot(x, soft[:, k], color=col[k], lw=1.4, label=r'$N_%d/\bar N_%d$' % (k, k))
    marks = [j for i, j in enumerate(ln['jumps_at_h']) if i == 0 or j - ln['jumps_at_h'][i - 1] > 0.05]
    for j in marks:
        ax.axvline(j, color='#9aa7b4', lw=0.6, ls=':')
    ax.axhline(1.0, color='#b8112b', lw=0.8, ls='--')
    ax.set_xlabel(r'offset along $(1,1,1)$ from the orbit centre, in steps $h$')
    ax.set_ylabel('count / threshold')
    ax.legend(frameon=False, fontsize=7, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.08))
    fig.tight_layout(); fig.savefig(OUT / 'fig_index.pdf'); plt.close(fig)


if __name__ == '__main__':
    res = {'orbits': {}, 'orbits_D_on_y': {}, 'fidelity': {}, 'event': {}, 'limit': {},
           'schedule': {}, 'robust': {}}
    for name in PLANTS:
        L = Loop(name, 'hard')
        starts = grid_starts(L)
        runs = orbit_stats(L, starts)
        res['orbits'][name] = dict(summary=summarize(runs), amigo=runs[13])
        Ly = Loop(name, 'hard', dtype='y')
        res['orbits_D_on_y'][name] = summarize(orbit_stats(Ly, grid_starts(Ly)))
        sm = res['orbits'][name]['summary']
        print(name, 'orbits', {k: v for k, v in sm.items() if k != 'centres'}, '| D on y:',
              {k: res['orbits_D_on_y'][name][k] for k in ('n_orbits', 'Ki', 'Ms')}, flush=True)
        res['fidelity'][name] = fidelity(name, starts, runs)
        print(name, 'fidelity', {k: v for k, v in res['fidelity'][name].items() if k != 'soft_summary'}, flush=True)
        res.setdefault('spread', {})[name] = spread(name, sm['centres'])
        print(name, 'spread', res['spread'][name], flush=True)
        res['event'][name] = event(name, starts)
        print(name, 'event', res['event'][name], flush=True)
        res['limit'][name] = limit(name)
        print(name, 'limit', [(r['index'], r['beta'], r['period'], r['ceiling'], round(r['radius'], 4),
                               r['step_from_prev'] and round(r['step_from_prev'], 4)) for r in res['limit'][name]], flush=True)
        res['schedule'][name] = schedule(name)
        print(name, 'schedule', {k: {kk: vv for kk, vv in v.items() if kk != 'final_gains'} for k, v in res['schedule'][name].items()}, flush=True)
        res['robust'][name] = robust(name, starts)
        print(name, 'robust', res['robust'][name], flush=True)
    res['meter'] = meter()
    res['line'] = line()
    print('meter', res['meter'], '\nline', {k: v for k, v in res['line'].items() if k not in ('s_h', 'hard', 'soft', 'k_delta')})
    (OUT / 'lc_results.json').write_text(json.dumps(res, indent=1, default=float))
    figures(res)
    print('wrote', OUT / 'lc_results.json')
