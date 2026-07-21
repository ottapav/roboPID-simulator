"""
Dash callbacks for RoboPID.

Gain sliders are log-scale (value = log10(gain), range 0.01-10) so the
slider itself always shows the true absolute gain via a de-logging
tooltip transform — there's no separate "base gain" store to fall out of
sync with what's displayed.

Flow:
  1. User edits plant params (tau, K, Td) or drags gain sliders →
     update_figures_patch sends only changed trace data (fast)
  2. User clicks Tune → run_tune (background) → streams live sliders/plots/status via
     set_progress, reusing the same patch-building logic as update_figures_patch,
     then reports the final gain trajectory in the Tuning History plot

Split into two figure-update callbacks for performance:
  update_figures_full  — triggered by ctype / limits changes → returns go.Figure
  update_figures_patch — triggered by sliders or plant params → returns dash.Patch (much faster)
"""

from __future__ import annotations
import ast
import os
import time

import numpy as np
from dash import Input, Output, State, Patch, no_update, ctx
import plotly.graph_objects as go

from core.config import read_config, build_disturbance_model
from core.features import standard_pid_features, loop_response_features
from core.signals import min_sim_time
from core.tuning import pid_tuning, CRAFT_READING

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'robopid.config')

# Trace indices in the time-domain figure
_T_Y, _T_R, _T_U = 0, 1, 2

# Colors
_C = {
    'y': 'red', 'r': 'rgba(200,0,0,0.4)', 'u': 'royalblue',
}

# Default tuning iteration budget per controller structure — fewer terms to
# search converge faster, so I/PI don't need as many iterations as full PID.
# Used as a fallback when the GUI's iterations input is empty/invalid; the
# user can otherwise override the count directly from the GUI.
N_ITER_BY_CTYPE = {'I': 50, 'PI': 100, 'PID': 200}

# Battery plants from RoboPID_JPC_paper/main.tex, Section "Validation on a
# plant battery": (tau string for input-tau, K, Td). Keyed by button id.
BATTERY_PRESETS = {
    'btn-p1': ('[10, 1, 1, 1]', 1.0, 1.0),        # lag-dominant
    'btn-p2': ('[5, 5, 5, 5]', 1.25, 8.0),        # balanced
    'btn-p3': ('[2, 1, 1, 1]', 1.0, 10.0),        # delay-dominant
    'btn-p4': ('[8, 8, 8, 8, 8, 8]', 1.0, 4.0),   # high-order slow
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


def _gain(log_val, default_log: float = 0.0) -> float:
    """Convert a log-scale slider value (log10 of the gain) to an absolute gain."""
    return 10.0 ** _f(log_val, default_log)


def _log_gain(gain: float) -> float:
    """Convert an absolute gain back to its log-scale slider value."""
    return float(np.log10(gain)) if gain > 0 else -2.0


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


def _build_warning(tau_str, K_val, Td_val, lim1, lim2, lim3) -> str:
    msgs = []
    if not _tau_parses(tau_str):
        msgs.append("tau: couldn't parse — using [5.0]")
    if not _parses(K_val):
        msgs.append("K: couldn't parse — using 1.0")
    if not _parses(Td_val):
        msgs.append("Td: couldn't parse — using 0.0")
    elif float(Td_val) < 0:
        msgs.append('Td: negative dead time clamped to 0')
    for label, val in (('Γ0 limit', lim1), ('Γ1 limit', lim2), ('Γ2 limit', lim3)):
        if not _parses(val):
            msgs.append(f"{label}: couldn't parse")
    return f'⚠ {" · ".join(msgs)}' if msgs else ''


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


def _simulate(tau, K, Td, Ts, Kp, Ki, Kd, ctype, lim1, lim2, lim3,
             delta, eps, cfg, dist_a, dist_b):
    """Run simulation and return (feats, sigs)."""
    Kd_eff = 0.0 if ctype in ('I', 'PI') else Kd
    Ki_eff = Ki  # I is active in I, PI, and PID
    Kp_eff = 0.0 if ctype == 'I' else Kp

    desc = standard_pid_features(limits=(lim1, lim2, lim3))
    feats, _, _, sigs = loop_response_features(
        desc,
        tau, K, Td, Ts, Kp_eff, Ki_eff, Kd_eff, dtype='y',
        simtype=int(cfg.get('simtype', 0)),
        minu=float(cfg.get('minu', -1.0)),
        maxu=float(cfg.get('maxu', 1.0)),
        dist_a=dist_a, dist_b=dist_b,
        delta=delta, eps=eps,
    )

    return feats, sigs


def _patch_figures(tau, K, Td, Ts, Kp, Ki, Kd, ctype, lim1, lim2, lim3,
                   delta, eps, cfg, dist_a, dist_b):
    """Simulate once and return (patch_f1, patch_f2, patch_f3, patch_time)."""
    feats, sigs = _simulate(
        tau, K, Td, Ts, Kp, Ki, Kd, ctype,
        lim1, lim2, lim3, delta, eps, cfg, dist_a, dist_b)

    patches_f = []
    for i, feat in enumerate(feats):
        p = Patch()
        p['data'][0]['x'] = feat['xdata'].tolist()
        p['data'][0]['y'] = feat['ydata'].tolist()
        p['layout']['title']['text'] = (
            f'Γ{i}: N={feat["phase"]:.2f} (limit {feat["limit"]})')
        patches_f.append(p)

    pt = Patch()
    t = sigs['t']
    pt['data'][_T_Y]['y'] = sigs['y'].tolist()
    pt['data'][_T_R]['y'] = np.ones(len(t)).tolist()
    pt['data'][_T_U]['y'] = sigs['u'].tolist()

    return patches_f[0], patches_f[1], patches_f[2], pt


# ── Full-figure builders ──────────────────────────────────────────────────────

_AXIS_STYLE = {
    'showgrid': True, 'gridcolor': 'black', 'gridwidth': 0.5,
    'zeroline': True, 'zerolinecolor': 'black', 'zerolinewidth': 0.5,
    'showline': True, 'linecolor': 'black', 'linewidth': 2,
}


def _style_axes(fig: go.Figure, zeroline: bool = True) -> go.Figure:
    """Slim black gridlines with bolder black axis lines, capped with an
    arrowhead so each axis reads as a directional arrow rather than a bare
    border. zeroline=False when the figure already draws its own line
    through the origin, to avoid stacking two lines on top of each other."""
    fig.update_xaxes(**{**_AXIS_STYLE, 'zeroline': zeroline})
    fig.update_yaxes(**{**_AXIS_STYLE, 'zeroline': zeroline})
    fig.update_layout(annotations=list(fig.layout.annotations or ()) + [
        {  # x-axis arrowhead
            'x': 1, 'y': 0, 'xref': 'paper', 'yref': 'paper',
            'ax': -12, 'ay': 0, 'axref': 'pixel', 'ayref': 'pixel',
            'showarrow': True, 'arrowcolor': 'black', 'arrowwidth': 2,
            'arrowsize': 1, 'arrowhead': 2, 'text': '',
        },
        {  # y-axis arrowhead
            'x': 0, 'y': 1, 'xref': 'paper', 'yref': 'paper',
            'ax': 0, 'ay': 12, 'axref': 'pixel', 'ayref': 'pixel',
            'showarrow': True, 'arrowcolor': 'black', 'arrowwidth': 2,
            'arrowsize': 1, 'arrowhead': 2, 'text': '',
        },
    ])
    return fig


def _build_feature_fig(feat: dict, idx: int) -> go.Figure:
    title = f'Γ{idx}: N={feat["phase"]:.2f} (limit {feat["limit"]})'
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
        showlegend=False, plot_bgcolor='white',
    )
    return _style_axes(fig, zeroline=False)


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
        plot_bgcolor='white',
    )
    return _style_axes(fig)


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
        plot_bgcolor='white',
    )
    return _style_axes(fig)


# ── Callbacks registration ────────────────────────────────────────────────────

def register_callbacks(app, default_Ts: float = 1.0):

    # ── 0a. Battery preset buttons ───────────────────────────────────────
    # Fills tau/K/Td from RoboPID_JPC_paper/main.tex's P1-P4 battery; the
    # existing tau/K/Td Inputs on update_figures_patch pick up the change and
    # redraw automatically. Controller gains/type/limits are left alone.
    @app.callback(
        Output('input-tau', 'value', allow_duplicate=True),
        Output('input-K', 'value', allow_duplicate=True),
        Output('input-Td', 'value', allow_duplicate=True),
        Input('btn-p1', 'n_clicks'),
        Input('btn-p2', 'n_clicks'),
        Input('btn-p3', 'n_clicks'),
        Input('btn-p4', 'n_clicks'),
        prevent_initial_call=True,
    )
    def apply_battery_preset(n1, n2, n3, n4):
        preset = BATTERY_PRESETS.get(ctx.triggered_id)
        if preset is None:
            return no_update, no_update, no_update
        tau_str, K, Td = preset
        return tau_str, f'{K:.2f}', f'{Td:.2f}'

    # ── 0b. Guard mode toggle ────────────────────────────────────────────
    # Checked (Guarded): delta stays whatever the field holds (Definition 4).
    # Unchecked (Unguarded): delta pinned at 0 and the field disabled -- the
    # paper's guarded-vs-unguarded comparison (Section "Well-posedness of
    # the count", Fig. fig3_wellposed). delta=0 degenerates settling_index's
    # guard into a no-op, so the count runs on the raw window, exactly what
    # "unguarded" means there.
    @app.callback(
        Output('input-delta', 'value'),
        Output('input-delta', 'disabled'),
        Input('guard-mode', 'value'),
        prevent_initial_call=True,
    )
    def toggle_guard_mode(guarded):
        if not guarded:
            return 0.0, True
        return 0.02, False

    # ── 1a. Full figure rebuild ────────────────────────────────────────────
    # Triggered by: ctype change, feature limit changes.
    # Plant params (tau/K/Td) are State here — they're handled as fast Inputs
    # in update_figures_patch below instead, to avoid a full figure rebuild.
    @app.callback(
        Output('graph-f1', 'figure'),
        Output('graph-f2', 'figure'),
        Output('graph-f3', 'figure'),
        Output('graph-time', 'figure'),
        Output('input-warning', 'children'),
        Input('dropdown-ctype', 'value'),
        Input('limit-1', 'value'),
        Input('limit-2', 'value'),
        Input('limit-3', 'value'),
        Input('input-eps', 'value'),
        Input('input-delta', 'value'),
        State('slider-kp', 'value'),
        State('slider-ki', 'value'),
        State('slider-kd', 'value'),
        State('input-tau', 'value'),
        State('input-K', 'value'),
        State('input-Td', 'value'),
        prevent_initial_call=False,
    )
    def update_figures_full(ctype, lim1_raw, lim2_raw, lim3_raw, eps_raw, delta_raw,
                            Fp, Fi, Fd, tau_str, K_val, Td_val):
        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _td(Td_val)
        lim1, lim2, lim3 = _f(lim1_raw, 0.5), _f(lim2_raw, 0.75), _f(lim3_raw, 1.0)
        eps, delta = _f(eps_raw, 0.1), _f(delta_raw, 0.02)
        Kp, Ki, Kd = _gain(Fp), _gain(Fi), _gain(Fd)

        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        feats, sigs = _simulate(
            tau, K, Td, default_Ts, Kp, Ki, Kd, ctype,
            lim1, lim2, lim3, delta, eps, cfg, dist_a, dist_b)

        figs_f = [_build_feature_fig(feats[i], i) for i in range(3)]
        fig_t = _build_time_fig(sigs)
        warning = _build_warning(tau_str, K_val, Td_val, lim1_raw, lim2_raw, lim3_raw)
        return figs_f[0], figs_f[1], figs_f[2], fig_t, warning

    # ── 1b. Patch update on slider move ────────────────────────────────────
    # Only trace data changes — no figure rebuild, very fast.
    @app.callback(
        Output('graph-f1', 'figure', allow_duplicate=True),
        Output('graph-f2', 'figure', allow_duplicate=True),
        Output('graph-f3', 'figure', allow_duplicate=True),
        Output('graph-time', 'figure', allow_duplicate=True),
        Output('input-warning', 'children', allow_duplicate=True),
        Input('slider-kp', 'value'),
        Input('slider-ki', 'value'),
        Input('slider-kd', 'value'),
        Input('input-tau', 'value'),
        Input('input-K', 'value'),
        Input('input-Td', 'value'),
        State('dropdown-ctype', 'value'),
        State('limit-1', 'value'),
        State('limit-2', 'value'),
        State('limit-3', 'value'),
        State('input-eps', 'value'),
        State('input-delta', 'value'),
        prevent_initial_call=True,
    )
    def update_figures_patch(Fp, Fi, Fd, tau_str, K_val, Td_val,
                             ctype, lim1_raw, lim2_raw, lim3_raw, eps_raw, delta_raw):
        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _td(Td_val)
        lim1, lim2, lim3 = _f(lim1_raw, 0.5), _f(lim2_raw, 0.75), _f(lim3_raw, 1.0)
        eps, delta = _f(eps_raw, 0.1), _f(delta_raw, 0.02)
        Kp, Ki, Kd = _gain(Fp), _gain(Fi), _gain(Fd)

        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        p_f1, p_f2, p_f3, p_time = _patch_figures(
            tau, K, Td, default_Ts, Kp, Ki, Kd, ctype,
            lim1, lim2, lim3, delta, eps, cfg, dist_a, dist_b)

        warning = _build_warning(tau_str, K_val, Td_val, lim1_raw, lim2_raw, lim3_raw)
        return p_f1, p_f2, p_f3, p_time, warning

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

    # ── 2b. Controller-type iteration default ───────────────────────────────
    # Reset the Iter field to this ctype's recommended budget whenever the
    # controller structure changes; the user can still type a custom count
    # afterward and it'll stick until the next ctype change.
    @app.callback(
        Output('input-niter', 'value'),
        Input('dropdown-ctype', 'value'),
        prevent_initial_call=False,
    )
    def reset_niter_default(ctype):
        return N_ITER_BY_CTYPE.get(ctype, 200)

    # ── 2c. Snap K/Td display to two decimals ──────────────────────────────
    # Runs once at load (formatting the CLI-supplied default) and again after
    # every debounced edit (typed value + blur/Enter). Self-limiting: once the
    # value is already the 2-decimal string, the formatted output matches the
    # input and Dash stops the chain.
    def _make_round2(default: float, minimum: float | None = None):
        def _round2(val):
            v = _f(val, default)
            if minimum is not None:
                v = max(v, minimum)
            formatted = f'{v:.2f}'
            return no_update if formatted == str(val) else formatted
        return _round2

    app.callback(
        Output('input-K', 'value'),
        Input('input-K', 'value'),
        prevent_initial_call=False,
    )(_make_round2(1.0, minimum=0.01))

    app.callback(
        Output('input-Td', 'value'),
        Input('input-Td', 'value'),
        prevent_initial_call=False,
    )(_make_round2(0.0))

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
        State('input-niter', 'value'),
        State('input-eps', 'value'),
        State('input-delta', 'value'),
        State('input-step', 'value'),
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
                 lim1, lim2, lim3, niter_val, eps_val, delta_val, step_val):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update

        tau = _parse_tau(tau_str)
        K   = _f(K_val, 1.0)
        Td  = _td(Td_val)
        lim1, lim2, lim3 = _f(lim1, 0.5), _f(lim2, 0.75), _f(lim3, 1.0)

        Kp, Ki, Kd = _gain(Fp), _gain(Fi), _gain(Fd)

        # Gains the selected controller type doesn't use stay at zero throughout
        # the search, matching _simulate()'s ctype zeroing.
        Kp_base = 0.0 if ctype == 'I' else Kp
        Kd_base = 0.0 if ctype in ('I', 'PI') else Kd

        T   = min_sim_time(tau, Td)
        cfg, dist_a, dist_b = _load_cfg(default_Ts)
        n_iter     = int(_f(niter_val, N_ITER_BY_CTYPE.get(ctype, 200)))
        n_iter     = max(10, min(n_iter, 2000))
        eps_tune   = _f(eps_val, 0.1)
        delta_tune = _f(delta_val, 0.02)
        tune_step  = _f(step_val, 0.1)

        desc = standard_pid_features(limits=(lim1, lim2, lim3))
        smin, smax = 0.01, 10.0

        last_push = [0.0]
        MIN_PUSH_INTERVAL = 0.08  # seconds; well under interval=150ms poll above
        hist_iter, hist_kp, hist_ki, hist_kd = [], [], [], []

        def on_iteration(i, N, Fp_cur, Fi_cur, Fd_cur, row):
            now = time.monotonic()
            if i < N and (now - last_push[0]) < MIN_PUSH_INTERVAL:
                return
            last_push[0] = now
            # Fp_cur/Fi_cur/Fd_cur are multipliers relative to the (Kp_base, Ki,
            # Kd_base) baseline passed into pid_tuning below — apply them directly
            # to get absolute gains, clamped to the slider's fixed 0.01-10 range.
            p_Kp = float(np.clip(Fp_cur * Kp_base, smin, smax))
            p_Ki = float(np.clip(Fi_cur * Ki, smin, smax))
            p_Kd = float(np.clip(Fd_cur * Kd_base, smin, smax))

            p_f1, p_f2, p_f3, p_time = _patch_figures(
                tau, K, Td, default_Ts, p_Kp, p_Ki, p_Kd, ctype,
                lim1, lim2, lim3, delta_tune, eps_tune, cfg, dist_a, dist_b)

            hist_iter.append(i)
            hist_kp.append(p_Kp)
            hist_ki.append(p_Ki)
            hist_kd.append(p_Kd)
            p_gains_hist = _build_gains_history_fig(hist_kp, hist_ki, hist_kd, it=hist_iter)

            set_progress((_log_gain(p_Kp), _log_gain(p_Ki), _log_gain(p_Kd),
                          f'Tuning… iter {i}/{N} — {CRAFT_READING[row]}',
                          p_time, p_f1, p_f2, p_f3, p_gains_hist))

        Fp_hist, Fi_hist, Fd_hist = pid_tuning(
            desc, tau, K, Td, default_Ts,
            Kp_base, Ki, Kd_base,
            dtype='y', T=T, N=n_iter,
            Fp_limits=(smin / Kp_base, smax / Kp_base) if Kp_base > 0 else (smin, smax),
            Fi_limits=(smin / Ki, smax / Ki),
            Fd_limits=(smin / Kd_base, smax / Kd_base) if Kd_base > 0 else (smin, smax),
            feature_limits=(lim1, lim2, lim3),
            step=tune_step,
            simtype=int(cfg.get('simtype', 0)),
            minu=float(cfg.get('minu', -1.0)),
            maxu=float(cfg.get('maxu', 1.0)),
            dist_a=dist_a, dist_b=dist_b,
            delta=delta_tune, eps=eps_tune,
            on_iteration=on_iteration,
        )

        Kp_traj = np.clip(Fp_hist * Kp_base, smin, smax)
        Ki_traj = np.clip(Fi_hist * Ki, smin, smax)
        Kd_traj = np.clip(Fd_hist * Kd_base, smin, smax)
        fig_gains_hist = _build_gains_history_fig(Kp_traj, Ki_traj, Kd_traj)

        # Gains the selected controller type doesn't tune keep their prior
        # slider value instead of being overwritten.
        out_Fp = _log_gain(Kp_traj[-1]) if ctype != 'I' else Fp
        out_Fd = _log_gain(Kd_traj[-1]) if ctype == 'PID' else Fd

        return out_Fp, _log_gain(Ki_traj[-1]), out_Fd, f'Tuned ({n_iter} iter)', fig_gains_hist
