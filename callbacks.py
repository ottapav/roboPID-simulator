"""
Dash callbacks for RoboPID.

Flow:
  1. User edits plant params (tau, K, Td) → no immediate effect
  2. User clicks Optimize → run_optimize → updates base-gains store → triggers full figure rebuild
  3. User drags sliders → update_figures_patch sends only changed trace data (fast)
  4. User clicks Tune → run_tune (background) → streams live sliders/plots/status via
     set_progress, reusing the same patch-building logic as update_figures_patch

Split into two figure-update callbacks for performance:
  update_figures_full  — triggered by base-gains / ctype / limits changes → returns go.Figure
  update_figures_patch — triggered by sliders only → returns dash.Patch (much faster)
"""

from __future__ import annotations
import ast
import os
import time

import numpy as np
from dash import Input, Output, State, Patch, no_update
import plotly.graph_objects as go

from core.config import read_config, build_disturbance_model
from core.features import standard_pid_features, loop_response_features
from core.optimizer import pid_design
from core.tuning import pid_tuning

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'robopid.config')

# Trace indices in the time-domain figure
_T_Y, _T_R, _T_U = 0, 1, 2

# Colors
_C = {
    'y': 'red', 'r': 'rgba(200,0,0,0.4)', 'u': 'royalblue',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_tau(tau_str) -> np.ndarray:
    try:
        val = ast.literal_eval(str(tau_str).strip())
        return np.atleast_1d(np.asarray(val, dtype=float))
    except Exception:
        return np.array([5.0])


def _f(val, default: float) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _load_cfg(Ts: float):
    cfg = read_config(CONFIG_FILE)
    dist_a, dist_b = build_disturbance_model(cfg, Ts)
    return cfg, dist_a, dist_b


def _simulate(tau, K, Td, Ts, Kp, Ki, Kd, ctype, lim1, lim2, lim3, cfg, dist_a, dist_b):
    """Run simulation and return (feats, sigs, gains_text)."""
    Kd_eff = 0.0 if ctype in ('I', 'PI') else Kd
    Ki_eff = 0.0 if ctype == 'I' else Ki
    Kp_eff = 0.0 if ctype == 'I' else Kp

    desc = standard_pid_features(limits=(lim1, lim2, lim3))
    feats, _, _, sigs, _ = loop_response_features(
        desc, ['uI', 'uP', 'uD'],
        tau, K, Td, Ts, Kp_eff, Ki_eff, Kd_eff, dtype='y',
        simtype=int(cfg.get('simtype', 0)),
        minu=float(cfg.get('minu', -1.0)),
        maxu=float(cfg.get('maxu', 1.0)),
        dist_a=dist_a, dist_b=dist_b,
    )

    if Ki_eff > 1e-12 and Kp_eff > 1e-12:
        gains_text = (f'Kp = {Kp_eff:.4g}   Ki = {Ki_eff:.4g}   Kd = {Kd_eff:.4g}'
                      f'   |   Ti = {Kp_eff/Ki_eff:.3g}   Td_c = {Kd_eff/Kp_eff:.3g}')
    else:
        gains_text = f'Kp = {Kp_eff:.4g}   Ki = {Ki_eff:.4g}   Kd = {Kd_eff:.4g}'

    return feats, sigs, gains_text


def _patch_figures(tau, K, Td, Ts, Kp, Ki, Kd, ctype, lim1, lim2, lim3, cfg, dist_a, dist_b):
    """Simulate once and return (patch_f1, patch_f2, patch_f3, patch_time, gains_text)."""
    feats, sigs, gains_text = _simulate(
        tau, K, Td, Ts, Kp, Ki, Kd, ctype,
        lim1, lim2, lim3, cfg, dist_a, dist_b)

    patches_f = []
    for i, feat in enumerate(feats):
        p = Patch()
        p['data'][0]['x'] = feat['xdata'].tolist()
        p['data'][0]['y'] = feat['ydata'].tolist()
        p['layout']['title']['text'] = (
            f'F{i+1}: {feat["phase"]:.2f} circles (limit {feat["limit"]})')
        patches_f.append(p)

    pt = Patch()
    t = sigs['t']
    pt['data'][_T_Y]['y'] = sigs['y'].tolist()
    pt['data'][_T_R]['y'] = np.ones(len(t)).tolist()
    pt['data'][_T_U]['y'] = sigs['u'].tolist()

    return patches_f[0], patches_f[1], patches_f[2], pt, gains_text


# ── Full-figure builders ──────────────────────────────────────────────────────

def _build_feature_fig(feat: dict, idx: int) -> go.Figure:
    title = f'F{idx+1}: {feat["phase"]:.2f} circles (limit {feat["limit"]})'
    fig = go.Figure(data=[
        go.Scatter(x=feat['xdata'].tolist(), y=feat['ydata'].tolist(),
                   mode='lines', line={'color': 'steelblue', 'width': 1.5},
                   name='trajectory'),
        go.Scatter(x=[0], y=[0], mode='markers',
                   marker={'symbol': 'cross', 'size': 10, 'color': 'red'},
                   showlegend=False),
    ])
    fig.add_hline(y=0, line={'color': 'gray', 'dash': 'dot', 'width': 0.7})
    fig.add_vline(x=0, line={'color': 'gray', 'dash': 'dot', 'width': 0.7})
    fig.update_layout(
        title={'text': title, 'font': {'size': 11}},
        xaxis_title=feat['xname'], yaxis_title=feat['yname'],
        margin={'l': 40, 'r': 8, 't': 38, 'b': 35},
        showlegend=False, plot_bgcolor='#f4f4f4',
    )
    return fig


def _build_time_fig(sigs: dict) -> go.Figure:
    t = sigs['t']
    fig = go.Figure(data=[
        go.Scatter(x=t.tolist(), y=sigs['y'].tolist(), mode='lines',
                   name='y (output)', line={'color': _C['y'], 'width': 2}),
        go.Scatter(x=t.tolist(), y=np.ones(len(t)).tolist(), mode='lines',
                   name='r (setpoint)', line={'color': _C['r'], 'dash': 'dash'}),
        go.Scatter(x=t.tolist(), y=sigs['u'].tolist(), mode='lines',
                   name='u (action)', line={'color': _C['u']}),
    ])
    fig.update_layout(
        title={'text': 'Step Response', 'font': {'size': 11}},
        xaxis_title='time', yaxis_title='value',
        margin={'l': 40, 'r': 8, 't': 38, 'b': 50},
        legend={'font': {'size': 10}, 'orientation': 'h', 'y': -0.3, 'x': 0},
        plot_bgcolor='#f4f4f4',
    )
    return fig


# ── Callbacks registration ────────────────────────────────────────────────────

def register_callbacks(app, default_Ts: float = 1.0):

    # ── 1a. Full figure rebuild ────────────────────────────────────────────
    # Triggered by: base-gains change, ctype change, feature limit changes.
    # Plant params (tau/K/Td) are State — they only affect plots after Optimize.
    @app.callback(
        Output('graph-f1', 'figure'),
        Output('graph-f2', 'figure'),
        Output('graph-f3', 'figure'),
        Output('graph-time', 'figure'),
        Output('gains-display', 'children'),
        Input('base-gains', 'data'),
        Input('dropdown-ctype', 'value'),
        Input('limit-1', 'value'),
        Input('limit-2', 'value'),
        Input('limit-3', 'value'),
        State('slider-kp', 'value'),
        State('slider-ki', 'value'),
        State('slider-kd', 'value'),
        State('input-tau', 'value'),
        State('input-K', 'value'),
        State('input-Td', 'value'),
        prevent_initial_call=False,
    )
    def update_figures_full(base_gains, ctype, lim1, lim2, lim3,
                            Fp, Fi, Fd, tau_str, K_val, Td_val):
        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _f(Td_val, 0.0)
        lim1, lim2, lim3 = _f(lim1, 0.5), _f(lim2, 0.75), _f(lim3, 1.0)
        Fp, Fi, Fd = _f(Fp, 1.0), _f(Fi, 1.0), _f(Fd, 1.0)

        if base_gains is None:
            base_gains = {'Kp': 1.0, 'Ki': 1.0, 'Kd': 0.0, 'optimized': False}

        Kp = Fp * float(base_gains.get('Kp', 1.0))
        Ki = Fi * float(base_gains.get('Ki', 1.0))
        Kd = Fd * float(base_gains.get('Kd', 0.0))
        optimized = base_gains.get('optimized', False)

        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        feats, sigs, gains_text = _simulate(
            tau, K, Td, default_Ts, Kp, Ki, Kd, ctype,
            lim1, lim2, lim3, cfg, dist_a, dist_b)

        if not optimized:
            gains_text = '⚠ Click Optimize to design a controller for this plant'

        figs_f = [_build_feature_fig(feats[i], i) for i in range(3)]
        fig_t = _build_time_fig(sigs)
        return figs_f[0], figs_f[1], figs_f[2], fig_t, gains_text

    # ── 1b. Patch update on slider move ────────────────────────────────────
    # Only trace data changes — no figure rebuild, very fast.
    @app.callback(
        Output('graph-f1', 'figure', allow_duplicate=True),
        Output('graph-f2', 'figure', allow_duplicate=True),
        Output('graph-f3', 'figure', allow_duplicate=True),
        Output('graph-time', 'figure', allow_duplicate=True),
        Output('gains-display', 'children', allow_duplicate=True),
        Input('slider-kp', 'value'),
        Input('slider-ki', 'value'),
        Input('slider-kd', 'value'),
        State('base-gains', 'data'),
        State('dropdown-ctype', 'value'),
        State('input-tau', 'value'),
        State('input-K', 'value'),
        State('input-Td', 'value'),
        State('limit-1', 'value'),
        State('limit-2', 'value'),
        State('limit-3', 'value'),
        prevent_initial_call=True,
    )
    def update_figures_patch(Fp, Fi, Fd, base_gains, ctype,
                             tau_str, K_val, Td_val, lim1, lim2, lim3):
        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _f(Td_val, 0.0)
        lim1, lim2, lim3 = _f(lim1, 0.5), _f(lim2, 0.75), _f(lim3, 1.0)
        Fp, Fi, Fd = _f(Fp, 1.0), _f(Fi, 1.0), _f(Fd, 1.0)

        if base_gains is None:
            return no_update, no_update, no_update, no_update, no_update

        Kp = Fp * float(base_gains.get('Kp', 1.0))
        Ki = Fi * float(base_gains.get('Ki', 1.0))
        Kd = Fd * float(base_gains.get('Kd', 0.0))

        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        p_f1, p_f2, p_f3, p_time, gains_text = _patch_figures(
            tau, K, Td, default_Ts, Kp, Ki, Kd, ctype,
            lim1, lim2, lim3, cfg, dist_a, dist_b)

        return p_f1, p_f2, p_f3, p_time, gains_text

    # ── 2. Optimize button ─────────────────────────────────────────────────
    @app.callback(
        Output('base-gains', 'data'),
        Output('slider-kp', 'value', allow_duplicate=True),
        Output('slider-ki', 'value', allow_duplicate=True),
        Output('slider-kd', 'value', allow_duplicate=True),
        Input('btn-optimize', 'n_clicks'),
        State('input-tau', 'value'),
        State('input-K', 'value'),
        State('input-Td', 'value'),
        State('dropdown-ctype', 'value'),
        prevent_initial_call=True,
    )
    def run_optimize(n_clicks, tau_str, K_val, Td_val, ctype):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _f(Td_val, 0.0)
        T   = 5.0 * (float(np.sum(tau)) + Td)
        Kp, Ki, Kd = pid_design(tau, K, Td, default_Ts, ctype, T=T)
        return {'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'optimized': True}, 1.0, 1.0, 1.0

    # ── 3. Tune button ────────────────────────────────────────────────────
    # Background callback: runs pid_tuning() in a worker process (DiskcacheManager)
    # and streams live progress (sliders, status text, and the same figure patches
    # update_figures_patch builds) back via set_progress.
    @app.callback(
        Output('slider-kp', 'value'),
        Output('slider-ki', 'value'),
        Output('slider-kd', 'value'),
        Output('tune-status', 'children'),
        Input('btn-tune', 'n_clicks'),
        State('slider-kp', 'value'),
        State('slider-ki', 'value'),
        State('slider-kd', 'value'),
        State('input-tau', 'value'),
        State('input-K', 'value'),
        State('input-Td', 'value'),
        State('dropdown-ctype', 'value'),
        State('limit-1', 'value'),
        State('limit-2', 'value'),
        State('limit-3', 'value'),
        State('base-gains', 'data'),
        background=True,
        progress=[
            Output('slider-kp', 'value', allow_duplicate=True),
            Output('slider-ki', 'value', allow_duplicate=True),
            Output('slider-kd', 'value', allow_duplicate=True),
            Output('tune-status', 'children', allow_duplicate=True),
            Output('graph-time', 'figure', allow_duplicate=True),
            Output('graph-f1', 'figure', allow_duplicate=True),
            Output('graph-f2', 'figure', allow_duplicate=True),
            Output('graph-f3', 'figure', allow_duplicate=True),
            Output('gains-display', 'children', allow_duplicate=True),
        ],
        running=[
            (Output('btn-tune', 'disabled'), True, False),
            (Output('btn-optimize', 'disabled'), True, False),
        ],
        interval=150,
        prevent_initial_call=True,
    )
    def run_tune(set_progress, n_clicks, Fp, Fi, Fd,
                 tau_str, K_val, Td_val, ctype,
                 lim1, lim2, lim3, base_gains):
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _f(Td_val, 0.0)
        Fp, Fi, Fd = _f(Fp, 1.0), _f(Fi, 1.0), _f(Fd, 1.0)
        lim1, lim2, lim3 = _f(lim1, 0.5), _f(lim2, 0.75), _f(lim3, 1.0)

        if base_gains is None:
            base_gains = {'Kp': 1.0, 'Ki': 1.0, 'Kd': 0.0}

        Kp = float(base_gains.get('Kp', 1.0))
        Ki = float(base_gains.get('Ki', 1.0))
        Kd = float(base_gains.get('Kd', 0.0))

        T   = max(10.0 * (float(np.sum(tau)) + Td), 50.0)
        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        n_iter     = int(cfg.get('n_iter', 100))
        tune_step  = float(cfg.get('tune_step', 0.05))

        desc = standard_pid_features(limits=(lim1, lim2, lim3))
        smin, smax = 0.01, 5.0

        last_push = [0.0]
        MIN_PUSH_INTERVAL = 0.08  # seconds; well under interval=150ms poll above

        def on_iteration(i, N, Fp_cur, Fi_cur, Fd_cur):
            now = time.monotonic()
            if i < N and (now - last_push[0]) < MIN_PUSH_INTERVAL:
                return
            last_push[0] = now
            # Fp_cur/Fi_cur/Fd_cur are multipliers relative to the (Kp*Fp, Ki*Fi, Kd*Fd)
            # baseline passed into pid_tuning below — re-multiply by the slider-scale
            # Fp/Fi/Fd to land back in slider.value units, same conversion as new_Fp below.
            p_Fp = float(np.clip(Fp_cur * Fp, smin, smax))
            p_Fi = float(np.clip(Fi_cur * Fi, smin, smax))
            p_Fd = float(np.clip(Fd_cur * Fd, smin, smax))

            p_f1, p_f2, p_f3, p_time, gains_text = _patch_figures(
                tau, K, Td, default_Ts, p_Fp * Kp, p_Fi * Ki, p_Fd * Kd, ctype,
                lim1, lim2, lim3, cfg, dist_a, dist_b)

            set_progress((p_Fp, p_Fi, p_Fd, f'Tuning… iter {i}/{N}',
                          p_time, p_f1, p_f2, p_f3, gains_text))

        Fp_hist, Fi_hist, Fd_hist = pid_tuning(
            desc, tau, K, Td, default_Ts,
            Kp * Fp, Ki * Fi, Kd * Fd,
            dtype='y', T=T, N=n_iter,
            Fp_limits=(smin / Fp, smax / Fp),
            Fi_limits=(smin / Fi, smax / Fi),
            Fd_limits=(smin / Fd, smax / Fd),
            feature_limits=(lim1, lim2, lim3),
            step=tune_step,
            simtype=int(cfg.get('simtype', 0)),
            minu=float(cfg.get('minu', -1.0)),
            maxu=float(cfg.get('maxu', 1.0)),
            dist_a=dist_a, dist_b=dist_b,
            on_iteration=on_iteration,
        )

        new_Fp = float(np.clip(Fp_hist[-1] * Fp, smin, smax))
        new_Fi = float(np.clip(Fi_hist[-1] * Fi, smin, smax))
        new_Fd = float(np.clip(Fd_hist[-1] * Fd, smin, smax))
        return new_Fp, new_Fi, new_Fd, f'Tuned ({n_iter} iter)'
