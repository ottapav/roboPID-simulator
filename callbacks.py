"""
Dash callbacks for RoboPID.

Flow:
  1. User edits plant params (tau, K, Td) or drags gain sliders →
     update_figures_patch sends only changed trace data (fast)
  2. User clicks Tune → run_tune (background) → streams live sliders/plots/status via
     set_progress, reusing the same patch-building logic as update_figures_patch,
     then reports the final gain trajectory in the Tuning History plot

Split into two figure-update callbacks for performance:
  update_figures_full  — triggered by base-gains / ctype / limits changes → returns go.Figure
  update_figures_patch — triggered by sliders or plant params → returns dash.Patch (much faster)
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
from core.signals import min_sim_time
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


def _td(val) -> float:
    """Parse dead time and clamp to >=0 (negative Td is a no-op in plant_tf's
    delay padding, so a raw negative value would be silently inconsistent
    with the T-floor/gain-scaling formulas that still subtract it)."""
    return max(_f(val, 0.0), 0.0)


def _parses(val) -> bool:
    try:
        return np.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _tau_parses(tau_str) -> bool:
    try:
        val = ast.literal_eval(str(tau_str).strip())
        np.atleast_1d(np.asarray(val, dtype=float))
        return True
    except Exception:
        return False


_cfg_cache: dict = {}


def _load_cfg(Ts: float):
    """Read robopid.config + build the disturbance model, cached by the
    config file's mtime so repeated calls (every slider drag) don't re-read
    the file or redo the expm/Lyapunov solve unless it actually changed on
    disk — this keeps the documented hot-reload behavior while avoiding
    per-callback recomputation."""
    try:
        mtime = os.path.getmtime(CONFIG_FILE)
    except OSError:
        mtime = None
    key = (mtime, Ts)
    if _cfg_cache.get('key') != key:
        cfg = read_config(CONFIG_FILE)
        dist_a, dist_b = build_disturbance_model(cfg, Ts)
        _cfg_cache['key'] = key
        _cfg_cache['value'] = (cfg, dist_a, dist_b)
    return _cfg_cache['value']


def _simulate(tau, K, Td, Ts, Kp, Ki, Kd, ctype, lim1, lim2, lim3, cfg, dist_a, dist_b):
    """Run simulation and return (feats, sigs)."""
    Kd_eff = 0.0 if ctype in ('I', 'PI') else Kd
    Ki_eff = Ki  # I is active in I, PI, and PID
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

    return feats, sigs


def _patch_figures(tau, K, Td, Ts, Kp, Ki, Kd, ctype, lim1, lim2, lim3, cfg, dist_a, dist_b):
    """Simulate once and return (patch_f1, patch_f2, patch_f3, patch_time)."""
    feats, sigs = _simulate(
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

    return patches_f[0], patches_f[1], patches_f[2], pt


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


def _build_gains_history_fig(Kp_traj, Ki_traj, Kd_traj, it=None) -> go.Figure:
    Kp_traj, Ki_traj, Kd_traj = np.asarray(Kp_traj), np.asarray(Ki_traj), np.asarray(Kd_traj)
    if it is None:
        it = list(range(len(Kp_traj)))
    fig = go.Figure(data=[
        go.Scatter(x=it, y=Kp_traj.tolist(), mode='lines',
                   name='Kp', line={'color': '#2a78d6', 'width': 2}),
        go.Scatter(x=it, y=Ki_traj.tolist(), mode='lines',
                   name='Ki', line={'color': '#008300', 'width': 2}),
        go.Scatter(x=it, y=Kd_traj.tolist(), mode='lines',
                   name='Kd', line={'color': '#4a3aa7', 'width': 2}),
    ])
    fig.update_layout(
        title={'text': 'Tuning History', 'font': {'size': 11}},
        xaxis_title='iteration', yaxis_title='gain value', yaxis_type='log',
        margin={'l': 40, 'r': 8, 't': 38, 'b': 50},
        legend={'font': {'size': 10}, 'orientation': 'h', 'y': -0.3, 'x': 0},
        plot_bgcolor='#f4f4f4',
    )
    return fig


# ── Callbacks registration ────────────────────────────────────────────────────

def register_callbacks(app, default_Ts: float = 1.0):

    # ── 1a. Full figure rebuild ────────────────────────────────────────────
    # Triggered by: base-gains change, ctype change, feature limit changes.
    # Plant params (tau/K/Td) are State here — they're handled as fast Inputs
    # in update_figures_patch below instead, to avoid a full figure rebuild.
    @app.callback(
        Output('graph-f1', 'figure'),
        Output('graph-f2', 'figure'),
        Output('graph-f3', 'figure'),
        Output('graph-time', 'figure'),
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
        Td  = _td(Td_val)
        lim1, lim2, lim3 = _f(lim1, 0.5), _f(lim2, 0.75), _f(lim3, 1.0)
        Fp, Fi, Fd = _f(Fp, 1.0), _f(Fi, 1.0), _f(Fd, 1.0)

        if base_gains is None:
            base_gains = {'Kp': 1.0, 'Ki': 1.0, 'Kd': 1.0}

        Kp = Fp * float(base_gains.get('Kp', 1.0))
        Ki = Fi * float(base_gains.get('Ki', 1.0))
        Kd = Fd * float(base_gains.get('Kd', 1.0))

        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        feats, sigs = _simulate(
            tau, K, Td, default_Ts, Kp, Ki, Kd, ctype,
            lim1, lim2, lim3, cfg, dist_a, dist_b)

        figs_f = [_build_feature_fig(feats[i], i) for i in range(3)]
        fig_t = _build_time_fig(sigs)
        return figs_f[0], figs_f[1], figs_f[2], fig_t

    # ── 1b. Patch update on slider move ────────────────────────────────────
    # Only trace data changes — no figure rebuild, very fast.
    @app.callback(
        Output('graph-f1', 'figure', allow_duplicate=True),
        Output('graph-f2', 'figure', allow_duplicate=True),
        Output('graph-f3', 'figure', allow_duplicate=True),
        Output('graph-time', 'figure', allow_duplicate=True),
        Input('slider-kp', 'value'),
        Input('slider-ki', 'value'),
        Input('slider-kd', 'value'),
        Input('input-tau', 'value'),
        Input('input-K', 'value'),
        Input('input-Td', 'value'),
        State('base-gains', 'data'),
        State('dropdown-ctype', 'value'),
        State('limit-1', 'value'),
        State('limit-2', 'value'),
        State('limit-3', 'value'),
        prevent_initial_call=True,
    )
    def update_figures_patch(Fp, Fi, Fd, tau_str, K_val, Td_val,
                             base_gains, ctype, lim1, lim2, lim3):
        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _td(Td_val)
        lim1, lim2, lim3 = _f(lim1, 0.5), _f(lim2, 0.75), _f(lim3, 1.0)
        Fp, Fi, Fd = _f(Fp, 1.0), _f(Fi, 1.0), _f(Fd, 1.0)

        if base_gains is None:
            return no_update, no_update, no_update, no_update

        Kp = Fp * float(base_gains.get('Kp', 1.0))
        Ki = Fi * float(base_gains.get('Ki', 1.0))
        Kd = Fd * float(base_gains.get('Kd', 1.0))

        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        p_f1, p_f2, p_f3, p_time = _patch_figures(
            tau, K, Td, default_Ts, Kp, Ki, Kd, ctype,
            lim1, lim2, lim3, cfg, dist_a, dist_b)

        return p_f1, p_f2, p_f3, p_time

    # ── 1c. Input validation feedback ──────────────────────────────────────
    # Text-only: re-runs the same parsing rules as the simulation callbacks
    # and surfaces a warning when a field fails to parse or gets clamped,
    # instead of silently substituting a default with no feedback.
    @app.callback(
        Output('input-warning', 'children'),
        Input('input-tau', 'value'),
        Input('input-K', 'value'),
        Input('input-Td', 'value'),
        Input('limit-1', 'value'),
        Input('limit-2', 'value'),
        Input('limit-3', 'value'),
        prevent_initial_call=False,
    )
    def validate_inputs(tau_str, K_val, Td_val, lim1, lim2, lim3):
        msgs = []
        if not _tau_parses(tau_str):
            msgs.append("tau: couldn't parse — using [5.0]")
        if not _parses(K_val):
            msgs.append("K: couldn't parse — using 1.0")
        if not _parses(Td_val):
            msgs.append("Td: couldn't parse — using 0.0")
        elif float(Td_val) < 0:
            msgs.append('Td: negative dead time clamped to 0')
        for label, val in (('F1 limit', lim1), ('F2 limit', lim2), ('F3 limit', lim3)):
            if not _parses(val):
                msgs.append(f"{label}: couldn't parse")

        return f'⚠ {" · ".join(msgs)}' if msgs else ''

    # ── 2. Controller-type gain-slider visibility ──────────────────────────
    @app.callback(
        Output('col-kp', 'className'),
        Output('col-kd', 'className'),
        Input('dropdown-ctype', 'value'),
        prevent_initial_call=False,
    )
    def toggle_gain_sliders(ctype):
        hidden = 'd-none'
        return (hidden if ctype == 'I' else '',
                hidden if ctype in ('I', 'PI') else '')

    # ── 3. Tune button ────────────────────────────────────────────────────
    # Background callback: runs pid_tuning() in a worker process (DiskcacheManager)
    # and streams live progress (sliders, status text, and the same figure patches
    # update_figures_patch builds) back via set_progress.
    @app.callback(
        Output('slider-kp', 'value'),
        Output('slider-ki', 'value'),
        Output('slider-kd', 'value'),
        Output('tune-status', 'children'),
        Output('graph-gains-history', 'figure'),
        Output('base-gains', 'data'),
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
            Output('graph-gains-history', 'figure', allow_duplicate=True),
        ],
        running=[
            (Output('btn-tune', 'disabled'), True, False),
        ],
        interval=150,
        prevent_initial_call=True,
    )
    def run_tune(set_progress, n_clicks, Fp, Fi, Fd,
                 tau_str, K_val, Td_val, ctype,
                 lim1, lim2, lim3, base_gains):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update

        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _td(Td_val)
        Fp, Fi, Fd = _f(Fp, 1.0), _f(Fi, 1.0), _f(Fd, 1.0)
        lim1, lim2, lim3 = _f(lim1, 0.5), _f(lim2, 0.75), _f(lim3, 1.0)

        if base_gains is None:
            base_gains = {'Kp': 1.0, 'Ki': 1.0, 'Kd': 1.0}

        Kp = float(base_gains.get('Kp', 1.0))
        Ki = float(base_gains.get('Ki', 1.0))
        Kd = float(base_gains.get('Kd', 1.0))

        # Gains the selected controller type doesn't use stay at zero throughout
        # the search, matching _simulate()'s ctype zeroing.
        Kp_base = 0.0 if ctype == 'I' else Kp
        Kd_base = 0.0 if ctype in ('I', 'PI') else Kd

        T   = min_sim_time(tau, Td)
        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        n_iter     = int(cfg.get('n_iter', 200))
        tune_step  = float(cfg.get('tune_step', 0.1))

        desc = standard_pid_features(limits=(lim1, lim2, lim3))
        smin, smax = 0.01, 5.0

        last_push = [0.0]
        MIN_PUSH_INTERVAL = 0.08  # seconds; well under interval=150ms poll above
        hist_iter, hist_kp, hist_ki, hist_kd = [], [], [], []

        def on_iteration(i, N, Fp_cur, Fi_cur, Fd_cur):
            now = time.monotonic()
            if i < N and (now - last_push[0]) < MIN_PUSH_INTERVAL:
                return
            last_push[0] = now
            # Fp_cur/Fi_cur/Fd_cur are multipliers relative to the (Kp_base*Fp, Ki*Fi,
            # Kd_base*Fd) baseline passed into pid_tuning below — re-multiply by the
            # slider-scale Fp/Fi/Fd to land back in slider.value units, same conversion
            # as new_Fp below.
            p_Fp = float(np.clip(Fp_cur * Fp, smin, smax))
            p_Fi = float(np.clip(Fi_cur * Fi, smin, smax))
            p_Fd = float(np.clip(Fd_cur * Fd, smin, smax))

            p_f1, p_f2, p_f3, p_time = _patch_figures(
                tau, K, Td, default_Ts, p_Fp * Kp, p_Fi * Ki, p_Fd * Kd, ctype,
                lim1, lim2, lim3, cfg, dist_a, dist_b)

            # Same (Kp_base, Ki, Kd_base) convention as Kp_traj/Ki_traj/Kd_traj below,
            # so the live curve lands exactly on the final trajectory at the last push.
            hist_iter.append(i)
            hist_kp.append(p_Fp * Kp_base)
            hist_ki.append(p_Fi * Ki)
            hist_kd.append(p_Fd * Kd_base)
            p_gains_hist = _build_gains_history_fig(hist_kp, hist_ki, hist_kd, it=hist_iter)

            set_progress((p_Fp, p_Fi, p_Fd, f'Tuning… iter {i}/{N}',
                          p_time, p_f1, p_f2, p_f3, p_gains_hist))

        Fp_hist, Fi_hist, Fd_hist = pid_tuning(
            desc, tau, K, Td, default_Ts,
            Kp_base * Fp, Ki * Fi, Kd_base * Fd,
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

        Kp_traj = Fp_hist * (Kp_base * Fp)
        Ki_traj = Fi_hist * (Ki * Fi)
        Kd_traj = Fd_hist * (Kd_base * Fd)
        fig_gains_hist = _build_gains_history_fig(Kp_traj, Ki_traj, Kd_traj)

        # Rebase: persist the freshly tuned absolute gains as the new baseline
        # (skipping terms this ctype doesn't tune, so they keep their prior
        # baseline instead of being zeroed out), and reset the multiplier
        # sliders to 1.0x on top of the new baseline.
        new_base_gains = dict(base_gains)
        new_base_gains['Ki'] = float(Ki_traj[-1])
        if ctype != 'I':
            new_base_gains['Kp'] = float(Kp_traj[-1])
        if ctype == 'PID':
            new_base_gains['Kd'] = float(Kd_traj[-1])

        return 1.0, 1.0, 1.0, f'Tuned ({n_iter} iter)', fig_gains_hist, new_base_gains
