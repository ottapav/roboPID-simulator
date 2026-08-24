"""
Dash callbacks for RoboPID.

Gain sliders are log-scale (value = log10(gain), range 0.01-10) so the
slider itself always shows the true absolute gain via a de-logging
tooltip transform — there's no separate "base gain" store to fall out of
sync with what's displayed.

Flow:
  1. User edits plant params (tau, K, L) or drags gain sliders →
     update_figures_patch sends only changed trace data (fast)
  2. User clicks Tune → run_tune (background) → streams live sliders/plots/status via
     set_progress, reusing the same patch-building logic as update_figures_patch,
     then reports the final gain trajectory in the Tuning History plot

Split into two figure-update callbacks for performance:
  update_figures_full  — triggered by ctype / limits changes → returns go.Figure
  update_figures_patch — triggered by sliders or plant params → returns dash.Patch (much faster)
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass

import numpy as np
from dash import Input, Output, State, Patch, html, no_update, ctx
import plotly.graph_objects as go

from core.admissibility import check_plant, diagnose_run
from core.config import read_config, build_noise_model
from core.features import standard_pid_features, loop_response_features
from core.params import (
    BETA, DELTA, EPS, GAIN_BOX, NBAR, N_ITER_BY_CTYPE, N_POINTS,
    fmt2, gain_slider_marks, parse_tau,
)
from core.signals import auto_grid
from core.tuning import pid_tuning, MANUAL_READING

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'robopid.config')

# Trace indices in the time-domain figure
_T_Y, _T_R, _T_U = 0, 1, 2

# Colors
_C = {
    'y': 'red', 'r': 'rgba(200,0,0,0.4)', 'u': 'royalblue',
}

# Battery plants from docs/JPC26_basic/main.tex, Section "Validation on a
# plant battery": (tau string for input-tau, K, L). Keyed by button id.
BATTERY_PRESETS = {
    'btn-p1': ('[10, 1, 1, 1]', 1.0, 1.0),        # lag-dominant
    'btn-p2': ('[5, 5, 5, 5]', 1.25, 8.0),        # balanced
    'btn-p3': ('[2, 1, 1, 1]', 1.0, 10.0),        # delay-dominant
    'btn-p4': ('[8, 8, 8, 8, 8, 8]', 1.0, 4.0),   # high-order slow
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(val, default: float) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _deadtime(val) -> float:
    """Parse dead time and clamp to >=0 (negative L is a no-op in plant_tf's
    delay padding, so a raw negative value would be silently inconsistent with
    the span = sum(tau) + L that auto_grid and the gain scaling both add it
    into)."""
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


def _build_warning(tau_notes, grid_notes, K_val, L_val, Nbar0, Nbar1, Nbar2) -> str:
    """Assemble the warning line.

    tau_notes comes from the single core.params.parse_tau call the caller has
    already made, so the tau field is never parsed a second time just to find
    out whether it parsed; grid_notes likewise rides along in the grid store
    from the _clamp_grid call that produced it."""
    msgs = list(tau_notes) + list(grid_notes)
    if not _parses(K_val):
        msgs.append("K: couldn't parse — using 1.0")
    if not _parses(L_val):
        msgs.append("L: couldn't parse — using 0.0")
    elif float(L_val) < 0:
        msgs.append('L: negative dead time clamped to 0')
    for label, val in (('Γ0 limit', Nbar0), ('Γ1 limit', Nbar1), ('Γ2 limit', Nbar2)):
        if not _parses(val):
            msgs.append(f"{label}: couldn't parse")
    return f'⚠ {" · ".join(msgs)}' if msgs else ''


def _render_findings(findings, prefix: str = '⚠ '):
    """Render core.admissibility Findings for either display surface.

    The modal and the warning line share this so a message cannot be worded
    twice and drift; the Finding carries title/detail/fixes precisely so the
    same object can be shown at either density. Returns '' for an empty list,
    which is what both targets want as their cleared state.
    """
    findings = list(findings)
    if not findings:
        return ''

    blocks = []
    for f in findings:
        parts = [html.Div(f'{prefix}{f.title}',
                          style={'fontWeight': 'bold', 'marginBottom': '3px'})]
        parts += [html.Div(p, style={'marginBottom': '3px'}) for p in f.detail]
        if f.fixes:
            parts.append(html.Ul([html.Li(x) for x in f.fixes],
                                 style={'marginBottom': 0, 'paddingLeft': '18px'}))
        blocks.append(html.Div(parts, style={'marginBottom': '8px'}))
    return html.Div(blocks)


def _status_suffix(findings) -> str:
    """The terse form, for the one-line status slot next to the TUNE button."""
    tags = [f.status for f in findings if f.status]
    return f' {tags[0]}' if tags else ''


_cfg_cache: dict = {}


def _load_cfg():
    """Read robopid.config, cached by the config file's mtime so repeated
    calls (every slider drag) don't re-read the file unless it actually
    changed on disk — this keeps the documented hot-reload behavior while
    avoiding per-callback re-parsing."""
    try:
        mtime = os.path.getmtime(CONFIG_FILE)
    except OSError:
        mtime = None
    if 'key' not in _cfg_cache or _cfg_cache['key'] != mtime:
        _cfg_cache['key'] = mtime
        _cfg_cache['value'] = read_config(CONFIG_FILE)
    return _cfg_cache['value']


def _noise_coeffs(enabled, std_raw, tau_raw, Ts: float) -> tuple[float, float]:
    """AR(1) coefficients for the plant-output noise filter, or (0, 0) when
    the Plant card's noise checkbox is unchecked."""
    if not enabled:
        return 0.0, 0.0
    std = max(_f(std_raw, 1.0), 0.0) / 100.0
    tau = max(_f(tau_raw, 0.5), 1e-4)
    return build_noise_model(tau, std, Ts)


# ── Simulation grid ───────────────────────────────────────────────────────────
# The grid has two possible sources -- auto_grid's proposal, or whatever the
# user typed into the Tsim field -- and three consumers (update_figures_full,
# update_figures_patch, run_tune). The arbitration lives here so all three agree.
# N is fixed at params.N_POINTS throughout, so Ts is never a source of its own:
# it is Tsim/(N_POINTS - 1) whichever branch the horizon came from.

def _grid_sig(tau, L) -> list:
    """The plant identity a proposed grid belongs to.

    Deliberately the arguments auto_grid actually consumes, not the raw tau
    vector: reordering tau, or splitting [5] into [2.5, 2.5], leaves the span
    and therefore the grid untouched. A signature that changed anyway would
    strand update_figures_patch's staleness guard waiting for a store update
    that propose_grid has no reason to produce, and the figures would stop
    redrawing.
    """
    return [float(np.sum(tau)), float(L)]


def _propose_grid(tau, L) -> dict:
    Tsim, Ts = auto_grid(tau, L)
    return {'sig': _grid_sig(tau, L), 'Tsim': Tsim, 'Ts': Ts, 'notes': []}


def _grid_matches(store, tau, L) -> bool:
    return isinstance(store, dict) and store.get('sig') == _grid_sig(tau, L)


def _resolve_grid(store, tau, L) -> dict:
    """The grid to simulate on: the store's when it belongs to this plant, the
    auto proposal otherwise (first paint, or a tau/L edit commit_grid has not
    answered yet)."""
    return store if _grid_matches(store, tau, L) else _propose_grid(tau, L)


def _clamp_grid(tsim_raw, fallback: dict) -> dict:
    """Validate a hand-entered horizon against the last good grid.

    Tsim is the only grid value the user can type: N is the constant N_POINTS,
    so Ts is a consequence of the horizon rather than a second decision. An
    unusable horizon therefore has nothing to correct against except the grid
    that was already standing, which is what `fallback` is -- shortening it
    instead would end the record before the response settles, pinning
    settling_index's k_delta onto k2 and degenerating Definition 4's guard (see
    auto_grid).

    Corrections are reported through the warning line rather than written back
    into the field -- a callback cannot write a component property it also
    reads, and this matches how a bad K is already handled: the box keeps what
    was typed, the warning says what was simulated.
    """
    notes: list[str] = []

    Tsim = _f(tsim_raw, float('nan'))
    if not np.isfinite(Tsim) or Tsim <= 0:
        Tsim = fallback['Tsim']
        notes.append(f'Tsim: not a positive number — using {Tsim:.4g}')

    return {'sig': fallback.get('sig'), 'Tsim': Tsim,
            'Ts': Tsim / (N_POINTS - 1), 'notes': notes}


@dataclass(frozen=True)
class SimParams:
    """Everything the figure callbacks need, parsed once.

    update_figures_full and update_figures_patch differ only in what they do
    with the result — building this in one place stops a new input having to be
    parsed identically in two, which is the kind of edit that half-lands.
    """
    tau: np.ndarray
    K: float
    L: float
    Tsim: float
    Ts: float
    Kp: float
    Ki: float
    Kd: float
    ctype: str
    Nbar: tuple[float, float, float]
    delta: float
    eps: float
    cfg: dict
    dist_a: float
    dist_b: float
    warning: str

    @property
    def sim_args(self) -> tuple:
        """Positional arguments for _simulate/_patch_figures."""
        return (self.tau, self.K, self.L, self.Tsim, self.Ts,
                self.Kp, self.Ki, self.Kd, self.ctype, *self.Nbar,
                self.delta, self.eps, self.cfg, self.dist_a, self.dist_b)


def _read_inputs(grid_store, ctype, tau_str, K_val, L_val,
                 Kp_log, Ki_log, Kd_log,
                 nbar0_raw, nbar1_raw, nbar2_raw, eps_raw, delta_raw,
                 noise_enabled, noise_std_raw, noise_tau_raw) -> SimParams:
    tau, tau_notes = parse_tau(tau_str)
    K = _f(K_val, 1.0)
    L = _deadtime(L_val)
    Nbar = (_f(nbar0_raw, NBAR[0]), _f(nbar1_raw, NBAR[1]), _f(nbar2_raw, NBAR[2]))
    grid = _resolve_grid(grid_store, tau, L)
    Ts = grid['Ts']
    return SimParams(
        tau=tau, K=K, L=L, Tsim=grid['Tsim'], Ts=Ts,
        Kp=_gain(Kp_log), Ki=_gain(Ki_log), Kd=_gain(Kd_log),
        ctype=ctype, Nbar=Nbar,
        delta=_f(delta_raw, DELTA), eps=_f(eps_raw, EPS),
        cfg=_load_cfg(),
        **dict(zip(('dist_a', 'dist_b'),
                   _noise_coeffs(noise_enabled, noise_std_raw, noise_tau_raw, Ts))),
        warning=_build_warning(tau_notes, grid.get('notes', ()), K_val, L_val,
                               nbar0_raw, nbar1_raw, nbar2_raw),
    )


def _simulate(tau, K, L, Tsim, Ts, Kp, Ki, Kd, ctype, Nbar0, Nbar1, Nbar2,
             delta, eps, cfg, dist_a, dist_b):
    """Run simulation and return (feats, sigs)."""
    Kd_eff = 0.0 if ctype in ('I', 'PI') else Kd
    Ki_eff = Ki  # I is active in I, PI, and PID
    Kp_eff = 0.0 if ctype == 'I' else Kp

    desc = standard_pid_features(Nbar=(Nbar0, Nbar1, Nbar2))
    feats, _, _, sigs = loop_response_features(
        desc,
        tau, K, L, Ts, Kp_eff, Ki_eff, Kd_eff, dtype='y', Tsim=Tsim,
        simtype=int(cfg.get('simtype', 0)),
        minu=float(cfg.get('minu', -1.0)),
        maxu=float(cfg.get('maxu', 1.0)),
        dist_a=dist_a, dist_b=dist_b,
        delta=delta, eps=eps,
    )

    return feats, sigs


def _feature_title(feat: dict, idx: int) -> str:
    """The Γ-plot heading. Shared by the full rebuild and the patch path so the
    two can't drift into rendering the same count differently."""
    return f'Γ{idx}: N={fmt2(feat["N"])} (limit {fmt2(feat["Nbar"])})'


def _patches_from(feats, sigs):
    """Build the four figure patches from an already-computed simulation.

    Kept separate from _patch_figures so the tuner's progress stream can render
    the features and signals pid_tuning already produced for that iteration,
    rather than simulating the same gains a second time to draw them.
    """
    patches_f = []
    for i, feat in enumerate(feats):
        p = Patch()
        p['data'][0]['x'] = feat['xdata'].tolist()
        p['data'][0]['y'] = feat['ydata'].tolist()
        p['layout']['title']['text'] = _feature_title(feat, i)
        patches_f.append(p)

    # x must be patched together with y: tau/K/L and the Tsim/Ts fields are all
    # Inputs upstream of here, and any of them can change both the sample count
    # and the spacing, so patching y alone would draw the new response against
    # the old time grid.
    pt = Patch()
    t = sigs['t'].tolist()
    for idx, arr in ((_T_Y, sigs['y']), (_T_R, np.ones(len(t))), (_T_U, sigs['u'])):
        pt['data'][idx]['x'] = t
        pt['data'][idx]['y'] = np.asarray(arr).tolist()

    return patches_f[0], patches_f[1], patches_f[2], pt


def _patch_figures(tau, K, L, Tsim, Ts, Kp, Ki, Kd, ctype, Nbar0, Nbar1, Nbar2,
                   delta, eps, cfg, dist_a, dist_b):
    """Simulate once and return (patch_f1, patch_f2, patch_f3, patch_time)."""
    return _patches_from(*_simulate(
        tau, K, L, Tsim, Ts, Kp, Ki, Kd, ctype,
        Nbar0, Nbar1, Nbar2, delta, eps, cfg, dist_a, dist_b))


# ── Full-figure builders ──────────────────────────────────────────────────────

_AXIS_STYLE = {
    'showgrid': True, 'gridcolor': 'black', 'gridwidth': 0.5,
    'zeroline': True, 'zerolinecolor': 'black', 'zerolinewidth': 0.5,
    'showline': True, 'linecolor': 'black', 'linewidth': 2,
}


def _style_axes(fig: go.Figure, zeroline: bool = True,
                right_axis: bool = False) -> go.Figure:
    """Slim black gridlines with bolder black axis lines, capped with an
    arrowhead so each axis reads as a directional arrow rather than a bare
    border. zeroline=False when the figure already draws its own line
    through the origin, to avoid stacking two lines on top of each other.

    right_axis caps an overlaying secondary y-axis with the same arrowhead, so
    a two-scale figure doesn't end up with one arrow-tipped axis and one bare
    one. Call this *before* declaring the secondary axis (see _build_time_fig):
    update_yaxes writes to every y-axis present, and the secondary one wants
    its own grid and zeroline settings, not these."""
    fig.update_xaxes(**{**_AXIS_STYLE, 'zeroline': zeroline})
    fig.update_yaxes(**{**_AXIS_STYLE, 'zeroline': zeroline})
    arrows = [
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
    ]
    if right_axis:
        arrows.append({  # secondary y-axis arrowhead
            'x': 1, 'y': 1, 'xref': 'paper', 'yref': 'paper',
            'ax': 0, 'ay': 12, 'axref': 'pixel', 'ayref': 'pixel',
            'showarrow': True, 'arrowcolor': _C['u'], 'arrowwidth': 2,
            'arrowsize': 1, 'arrowhead': 2, 'text': '',
        })
    fig.update_layout(annotations=list(fig.layout.annotations or ()) + arrows)
    return fig


def _build_feature_fig(feat: dict, idx: int) -> go.Figure:
    title = _feature_title(feat, idx)
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
    """Step response: y and r on the left axis, u on its own axis at the right.

    u shares nothing with y dimensionally — it is whatever the actuator takes,
    bounded by robopid.config's minu/maxu, while y and r sit around 1 — so a
    shared scale either flattens the response or pushes the action off the top.
    Each axis is tinted to its traces, since a two-scale plot is unreadable
    without knowing which curve to read against which side.
    """
    t = sigs['t']
    fig = go.Figure(data=[
        go.Scatter(x=t.tolist(), y=sigs['y'].tolist(), mode='lines',
                   name='y (output)', line={'color': _C['y'], 'width': 2}),
        go.Scatter(x=t.tolist(), y=np.ones(len(t)).tolist(), mode='lines',
                   name='r (setpoint)', line={'color': _C['r'], 'dash': 'dash'}),
        go.Scatter(x=t.tolist(), y=sigs['u'].tolist(), mode='lines',
                   name='u (action)', line={'color': _C['u']}, yaxis='y2'),
    ])
    fig.update_layout(
        title={'text': 'Step Response', 'font': {'size': 11}},
        xaxis_title='time',
        yaxis_title={'text': 'y, r', 'font': {'color': _C['y']}},
        # Right margin has to make room for the tick labels and title that used
        # to have nowhere to go.
        margin={'l': 40, 'r': 44, 't': 38, 'b': 50},
        legend={'font': {'size': 10}, 'orientation': 'h', 'y': -0.3, 'x': 0},
        plot_bgcolor='white',
    )
    fig = _style_axes(fig, right_axis=True)

    # After _style_axes, which writes to every y-axis that exists: this one
    # carries no grid (a second set of lines at the right axis's tick positions
    # would not align with the left's) and no zeroline (u's zero and y's zero
    # land at different heights, so two of them read as a plotting error).
    fig.update_layout(yaxis2={
        'title': {'text': 'u', 'font': {'color': _C['u']}},
        'overlaying': 'y', 'side': 'right',
        'showgrid': False, 'zeroline': False,
        'showline': True, 'linewidth': _AXIS_STYLE['linewidth'],
        'linecolor': _C['u'], 'tickfont': {'color': _C['u']},
    })
    return fig


def _build_gains_history_fig(Kp_traj, Ki_traj, Kd_traj, it=None) -> go.Figure:
    """Gain trajectory plot. A trajectory passed as None is omitted entirely —
    gains the selected controller structure doesn't use have a base of 0, which
    the [Kmin, Kmax] clamp would otherwise lift to a flat line at Kmin that no
    iteration ever evaluated."""
    series = [('Kp', Kp_traj, '#2a78d6'),
              ('Ki', Ki_traj, '#008300'),
              ('Kd', Kd_traj, '#4a3aa7')]
    n = max((len(t) for _, t, _ in series if t is not None), default=0)
    if it is None:
        it = list(range(n))
    fig = go.Figure(data=[
        go.Scatter(x=it, y=np.asarray(traj).tolist(), mode='lines',
                   name=name, line={'color': color, 'width': 2})
        for name, traj, color in series if traj is not None
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

def register_callbacks(app):

    # ── 0a. Battery preset buttons ───────────────────────────────────────
    # Fills tau/K/L from docs/JPC26_basic/main.tex's P1-P4 battery; the
    # existing tau/K/L Inputs on update_figures_patch pick up the change and
    # redraw automatically. Controller gains/type/limits are left alone.
    @app.callback(
        Output('input-tau', 'value', allow_duplicate=True),
        Output('input-K', 'value', allow_duplicate=True),
        Output('input-L', 'value', allow_duplicate=True),
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
        tau_str, K, L = preset
        return tau_str, fmt2(K), fmt2(L)

    # ── 0a2. Simulation grid: propose, then commit ───────────────────────
    # Two callbacks rather than one because Dash forbids a callback from both
    # reading and writing the same component property, and the Tsim field is
    # written by the app and read from the user. The split turns out to be the
    # right shape anyway: neither half has to work out which of the two it is
    # looking at.
    #
    #   propose_grid   τ/L/↺        → the Tsim field          (the proposal)
    #   commit_grid    the field    → grid-store              (the verdict)
    #   show_ts        grid-store   → the Ts readout          (the consequence)
    #
    # So a τ edit reaches the store the long way round, one callback behind the
    # figure callbacks that also watch τ. update_figures_patch's staleness guard
    # is what keeps that from drawing the new plant on the old time base.
    @app.callback(
        Output('input-tsim', 'value'),
        Input('input-tau', 'value'),
        Input('input-L', 'value'),
        Input('btn-grid-auto', 'n_clicks'),
        prevent_initial_call=True,
    )
    def propose_grid(tau_str, L_val, _n_auto):
        tau, _ = parse_tau(tau_str)
        grid = _propose_grid(tau, _deadtime(L_val))
        return f'{grid["Tsim"]:.4g}'

    @app.callback(
        Output('grid-store', 'data'),
        Input('input-tsim', 'value'),
        State('input-tau', 'value'),
        State('input-L', 'value'),
        State('grid-store', 'data'),
        prevent_initial_call=False,
    )
    def commit_grid(tsim_raw, tau_str, L_val, store):
        tau, _ = parse_tau(tau_str)
        L = _deadtime(L_val)
        fallback = _resolve_grid(store, tau, L)
        grid = _clamp_grid(tsim_raw, fallback)
        grid['sig'] = _grid_sig(tau, L)
        return grid

    # Ts is read out of the committed grid rather than computed alongside the
    # proposal, so what the header shows is the period actually simulated --
    # including when _clamp_grid answered a bad horizon with the old one.
    @app.callback(
        Output('display-ts', 'children'),
        Input('grid-store', 'data'),
        prevent_initial_call=False,
    )
    def show_ts(store):
        if not isinstance(store, dict) or 'Ts' not in store:
            return no_update
        return fmt2(store['Ts'])

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

    # ── 0c. Output-noise checkbox toggle ─────────────────────────────────
    # Greys the sigma/tau fields out when unchecked; their values are left
    # alone (unlike guard-mode's delta, there's no "pinned" state to apply —
    # the checkbox itself, not the field values, gates whether noise is
    # simulated at all, see _noise_coeffs).
    @app.callback(
        Output('input-noise-std', 'disabled'),
        Output('input-noise-tau', 'disabled'),
        Input('noise-enabled', 'value'),
        prevent_initial_call=True,
    )
    def toggle_noise_mode(enabled):
        return not enabled, not enabled

    # ── 0d. Reset controller gains ───────────────────────────────────────
    @app.callback(
        Output('slider-kp', 'value', allow_duplicate=True),
        Output('slider-ki', 'value', allow_duplicate=True),
        Output('slider-kd', 'value', allow_duplicate=True),
        Input('btn-reset-controller', 'n_clicks'),
        prevent_initial_call=True,
    )
    def reset_controller(n_clicks):
        return 0.0, 0.0, 0.0

    # ── 0e. Reset tuner parameters ───────────────────────────────────────
    # Restores the paper's dimensionless constants and tuner settings to
    # their make_layout() defaults and clears the Tuning History plot from
    # any previous run. Iter resets to the current controller structure's
    # budget (matching reset_niter_default's ctype-based logic) rather than
    # a single fixed number.
    @app.callback(
        Output('input-nbar0', 'value', allow_duplicate=True),
        Output('input-nbar1', 'value', allow_duplicate=True),
        Output('input-nbar2', 'value', allow_duplicate=True),
        Output('input-kmin', 'value', allow_duplicate=True),
        Output('input-kmax', 'value', allow_duplicate=True),
        Output('input-eps', 'value', allow_duplicate=True),
        Output('input-beta', 'value', allow_duplicate=True),
        Output('input-niter', 'value', allow_duplicate=True),
        Output('guard-mode', 'value', allow_duplicate=True),
        Output('graph-gains-history', 'figure', allow_duplicate=True),
        Input('btn-reset-tuner', 'n_clicks'),
        State('dropdown-ctype', 'value'),
        prevent_initial_call=True,
    )
    def reset_tuner(n_clicks, ctype):
        niter = N_ITER_BY_CTYPE.get(ctype, N_ITER_BY_CTYPE['PID'])
        empty_history = _build_gains_history_fig([], [], [])
        return (*NBAR, *GAIN_BOX, EPS, BETA, niter, True, empty_history)

    # ── 0f. Gain slider range follows the gain box [Kmin, Kmax] ────────────
    # The Kp/Ki/Kd sliders are log-scale over [Kmin, Kmax] (paper Section 6's
    # gain boundary), so widening/narrowing the box has to re-scale the
    # sliders' own range, not just the tuning search/clamp bounds. Any
    # slider currently outside the new range gets pulled back inside it.
    @app.callback(
        Output('slider-kp', 'min'),
        Output('slider-kp', 'max'),
        Output('slider-kp', 'marks'),
        Output('slider-ki', 'min'),
        Output('slider-ki', 'max'),
        Output('slider-ki', 'marks'),
        Output('slider-kd', 'min'),
        Output('slider-kd', 'max'),
        Output('slider-kd', 'marks'),
        Output('slider-kp', 'value', allow_duplicate=True),
        Output('slider-ki', 'value', allow_duplicate=True),
        Output('slider-kd', 'value', allow_duplicate=True),
        Input('input-kmin', 'value'),
        Input('input-kmax', 'value'),
        State('slider-kp', 'value'),
        State('slider-ki', 'value'),
        State('slider-kd', 'value'),
        prevent_initial_call=True,
    )
    def update_gain_slider_range(kmin_raw, kmax_raw, kp_log, ki_log, kd_log):
        kmin = max(_f(kmin_raw, GAIN_BOX[0]), 1e-6)
        kmax = _f(kmax_raw, GAIN_BOX[1])
        if kmax <= kmin:
            return (no_update,) * 12

        lo, hi = np.log10(kmin), np.log10(kmax)
        marks = gain_slider_marks(kmin, kmax)

        def clamp(v):
            if v is None:
                return no_update
            c = float(np.clip(v, lo, hi))
            return no_update if abs(c - v) < 1e-9 else c

        return (lo, hi, marks, lo, hi, marks, lo, hi, marks,
                clamp(kp_log), clamp(ki_log), clamp(kd_log))

    # ── 1a. Full figure rebuild ────────────────────────────────────────────
    # Triggered by: ctype change, feature limit changes.
    # Plant params (tau/K/L) are State here — they're handled as fast Inputs
    # in update_figures_patch below instead, to avoid a full figure rebuild.
    @app.callback(
        Output('graph-f1', 'figure'),
        Output('graph-f2', 'figure'),
        Output('graph-f3', 'figure'),
        Output('graph-time', 'figure'),
        Output('input-warning', 'children'),
        Input('dropdown-ctype', 'value'),
        Input('input-nbar0', 'value'),
        Input('input-nbar1', 'value'),
        Input('input-nbar2', 'value'),
        Input('input-eps', 'value'),
        Input('input-delta', 'value'),
        State('slider-kp', 'value'),
        State('slider-ki', 'value'),
        State('slider-kd', 'value'),
        State('input-tau', 'value'),
        State('input-K', 'value'),
        State('input-L', 'value'),
        State('noise-enabled', 'value'),
        State('input-noise-std', 'value'),
        State('input-noise-tau', 'value'),
        # State, not Input: this callback fires on ctype and the feature limits,
        # none of which move the grid. _resolve_grid's fallback is only there to
        # cover the initial paint, before commit_grid has written the store.
        State('grid-store', 'data'),
        prevent_initial_call=False,
    )
    def update_figures_full(ctype, nbar0_raw, nbar1_raw, nbar2_raw, eps_raw, delta_raw,
                            Kp_log, Ki_log, Kd_log, tau_str, K_val, L_val,
                            noise_enabled, noise_std_raw, noise_tau_raw, grid_store):
        p = _read_inputs(grid_store, ctype, tau_str, K_val, L_val,
                         Kp_log, Ki_log, Kd_log,
                         nbar0_raw, nbar1_raw, nbar2_raw, eps_raw, delta_raw,
                         noise_enabled, noise_std_raw, noise_tau_raw)
        feats, sigs = _simulate(*p.sim_args)

        figs_f = [_build_feature_fig(feats[i], i) for i in range(3)]
        return (*figs_f, _build_time_fig(sigs), p.warning)

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
        Input('input-L', 'value'),
        Input('noise-enabled', 'value'),
        Input('input-noise-std', 'value'),
        Input('input-noise-tau', 'value'),
        Input('grid-store', 'data'),
        State('dropdown-ctype', 'value'),
        State('input-nbar0', 'value'),
        State('input-nbar1', 'value'),
        State('input-nbar2', 'value'),
        State('input-eps', 'value'),
        State('input-delta', 'value'),
        State('tuning-active', 'data'),
        prevent_initial_call=True,
    )
    def update_figures_patch(Kp_log, Ki_log, Kd_log, tau_str, K_val, L_val,
                             noise_enabled, noise_std_raw, noise_tau_raw, grid_store,
                             ctype, nbar0_raw, nbar1_raw, nbar2_raw, eps_raw, delta_raw,
                             tuning_active):
        # A Tune run streams slider values back through set_progress, and those
        # sliders are Inputs here. Without this gate every progress push would
        # trigger a full simulation in this process whose figure output then
        # races the patch the tuner is streaming to the same four graphs.
        if tuning_active:
            return (no_update,) * 5

        # tau/L are Inputs here and also drive propose_grid -> commit_grid, so a
        # tau edit reaches this callback one pass before the grid derived from
        # it does. Drawing that pass would put the new plant on the old time
        # base — exactly what the x-with-y note in _patches_from warns about.
        # The store lands a beat later and re-triggers us with the two
        # consistent, so skipping here costs a frame, not a redraw. `is not
        # None` keeps the very first patch renderable if the store hasn't
        # arrived at all.
        if grid_store is not None and not _grid_matches(
                grid_store, parse_tau(tau_str)[0], _deadtime(L_val)):
            return (no_update,) * 5

        p = _read_inputs(grid_store, ctype, tau_str, K_val, L_val,
                         Kp_log, Ki_log, Kd_log,
                         nbar0_raw, nbar1_raw, nbar2_raw, eps_raw, delta_raw,
                         noise_enabled, noise_std_raw, noise_tau_raw)
        return (*_patch_figures(*p.sim_args), p.warning)

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
        return N_ITER_BY_CTYPE.get(ctype, N_ITER_BY_CTYPE['PID'])

    # ── 2c. Snap K/L display to two decimals ──────────────────────────────
    # Runs once at load (formatting the CLI-supplied default) and again after
    # every debounced edit (typed value + blur/Enter). Self-limiting: once the
    # value is already the 2-decimal string, the formatted output matches the
    # input and Dash stops the chain.
    def _make_round2(default: float, minimum: float | None = None):
        def _round2(val):
            v = _f(val, default)
            if minimum is not None:
                v = max(v, minimum)
            formatted = fmt2(v)
            return no_update if formatted == str(val) else formatted
        return _round2

    app.callback(
        Output('input-K', 'value'),
        Input('input-K', 'value'),
        prevent_initial_call=False,
    )(_make_round2(1.0, minimum=0.01))

    app.callback(
        Output('input-L', 'value'),
        Input('input-L', 'value'),
        prevent_initial_call=False,
    )(_make_round2(0.0))

    app.callback(
        Output('input-noise-std', 'value'),
        Input('input-noise-std', 'value'),
        prevent_initial_call=False,
    )(_make_round2(1.0, minimum=0.0))

    app.callback(
        Output('input-noise-tau', 'value'),
        Input('input-noise-tau', 'value'),
        prevent_initial_call=False,
    )(_make_round2(0.5, minimum=0.01))

    # ── 2d. Dismiss the admissibility modal ───────────────────────────────
    # The header × and the backdrop close it client-side; the footer button
    # needs this. allow_duplicate because run_tune owns is_open as a primary
    # output.
    @app.callback(
        Output('tune-error-modal', 'is_open', allow_duplicate=True),
        Input('btn-tune-error-close', 'n_clicks'),
        prevent_initial_call=True,
    )
    def close_tune_error(_n_clicks):
        return False

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
        Output('tune-error-modal', 'is_open'),
        Output('tune-error-body', 'children'),
        Output('tune-findings', 'children'),
        Input('btn-tune', 'n_clicks'),
        State('slider-kp', 'value'),
        State('slider-ki', 'value'),
        State('slider-kd', 'value'),
        State('input-tau', 'value'),
        State('input-K', 'value'),
        State('input-L', 'value'),
        State('noise-enabled', 'value'),
        State('input-noise-std', 'value'),
        State('input-noise-tau', 'value'),
        State('dropdown-ctype', 'value'),
        State('input-nbar0', 'value'),
        State('input-nbar1', 'value'),
        State('input-nbar2', 'value'),
        State('input-niter', 'value'),
        State('input-eps', 'value'),
        State('input-delta', 'value'),
        State('input-beta', 'value'),
        State('input-kmin', 'value'),
        State('input-kmax', 'value'),
        State('grid-store', 'data'),
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
            Output('tune-findings', 'children', allow_duplicate=True),
        ],
        running=[
            (Output('btn-tune', 'disabled'), True, False),
            (Output('tuning-active', 'data'), True, False),
        ],
        interval=150,
        prevent_initial_call=True,
    )
    def run_tune(set_progress, n_clicks, Kp_log, Ki_log, Kd_log,
                 tau_str, K_val, L_val, noise_enabled, noise_std_raw, noise_tau_raw,
                 ctype, Nbar0, Nbar1, Nbar2, niter_val, eps_val, delta_val, beta_val,
                 kmin_val, kmax_val, grid_store):
        tau, tau_notes = parse_tau(tau_str)
        # nan, not 1.0, as the fallback: an unreadable K is a thing to refuse,
        # not to substitute for. The figure callbacks can afford to draw
        # *something*; a tuning run that silently searched against K = 1 would
        # hand back gains for a plant nobody asked about.
        K   = _f(K_val, float('nan'))
        L   = _deadtime(L_val)
        Nbar0, Nbar1, Nbar2 = _f(Nbar0, 0.5), _f(Nbar1, 0.75), _f(Nbar2, 1.0)

        Kp, Ki, Kd = _gain(Kp_log), _gain(Ki_log), _gain(Kd_log)

        # Gains the selected controller type doesn't use stay at zero throughout
        # the search, matching _simulate()'s ctype zeroing.
        Kp_base = 0.0 if ctype == 'I' else Kp
        Kd_base = 0.0 if ctype in ('I', 'PI') else Kd

        Kmin       = _f(kmin_val, GAIN_BOX[0])
        Kmax       = _f(kmax_val, GAIN_BOX[1])

        # Admissibility, before anything is simulated. A blocking finding means
        # the search has no answer to converge to, so the run is refused and
        # explained in the modal rather than spending n_iter iterations walking
        # to a bound and reporting "Tuned".
        verdict = check_plant(tau, K, L, ctype,
                              {'Kp': Kp_base, 'Ki': Ki, 'Kd': Kd_base},
                              gain_box=(Kmin, Kmax), K_raw=K_val)
        if not verdict.ok:
            # The findings line is cleared rather than left alone: whatever the
            # previous run advised is about gains this click did not produce.
            return (no_update, no_update, no_update,
                    _status_suffix(verdict.blocking).strip() or '⚠ Not tunable',
                    no_update, True, _render_findings(verdict.blocking, prefix=''),
                    '')

        # Advisory findings ride along with the run: shown from the first
        # progress push, then re-emitted alongside whatever the run itself
        # turned out to say.
        gate_warnings = list(verdict.warnings)
        warn_children = _render_findings(gate_warnings)

        # The grid the user is looking at, so a tuning run scores the same
        # response the plots show rather than a second, independent one.
        grid = _resolve_grid(grid_store, tau, L)
        Tsim, Ts = grid['Tsim'], grid['Ts']
        cfg = _load_cfg()
        dist_a, dist_b = _noise_coeffs(noise_enabled, noise_std_raw, noise_tau_raw, Ts)
        n_iter     = int(_f(niter_val, N_ITER_BY_CTYPE.get(ctype, N_ITER_BY_CTYPE['PID'])))
        n_iter     = max(10, min(n_iter, 2000))
        eps_tune   = _f(eps_val, 0.1)
        delta_tune = _f(delta_val, 0.02)
        beta_tune  = _f(beta_val, 0.1)

        desc = standard_pid_features(Nbar=(Nbar0, Nbar1, Nbar2))

        last_push = [0.0]
        MIN_PUSH_INTERVAL = 0.08  # seconds; well under interval=150ms poll above
        hist_iter, hist_kp, hist_ki, hist_kd = [], [], [], []
        last_feats = [None]

        def on_iteration(i, n_iter, Fp_cur, Fi_cur, Fd_cur, row, feats, sigs):
            # Captured above the throttle, so the counts the run-diagnosis reads
            # are the last ones actually scored rather than the last ones that
            # happened to be drawn. pid_tuning's post-loop call is never
            # throttled (i == n_iter), so this always ends up holding the final
            # iteration's features.
            last_feats[0] = feats

            now = time.monotonic()
            if i < n_iter and (now - last_push[0]) < MIN_PUSH_INTERVAL:
                return
            last_push[0] = now
            # Fp_cur/Fi_cur/Fd_cur are multipliers relative to the (Kp_base, Ki,
            # Kd_base) baseline passed into pid_tuning below — apply them directly
            # to get absolute gains, clamped to the gain boundary [Kmin, Kmax].
            p_Kp = float(np.clip(Fp_cur * Kp_base, Kmin, Kmax))
            p_Ki = float(np.clip(Fi_cur * Ki, Kmin, Kmax))
            p_Kd = float(np.clip(Fd_cur * Kd_base, Kmin, Kmax))

            # Render what the tuner just scored rather than re-simulating it —
            # one simulation per iteration instead of two, and the plots are
            # guaranteed to be the record the decision was actually made on.
            p_f1, p_f2, p_f3, p_time = _patches_from(feats, sigs)

            hist_iter.append(i)
            hist_kp.append(p_Kp)
            hist_ki.append(p_Ki)
            hist_kd.append(p_Kd)
            p_gains_hist = _build_gains_history_fig(
                hist_kp if Kp_base > 0 else None,
                hist_ki,
                hist_kd if Kd_base > 0 else None,
                it=hist_iter)

            set_progress((_log_gain(p_Kp), _log_gain(p_Ki), _log_gain(p_Kd),
                          f'Tuning… iter {i}/{n_iter} — {MANUAL_READING[row]}',
                          p_time, p_f1, p_f2, p_f3, p_gains_hist,
                          warn_children))

        # The multiplier box, kept in one place: pid_tuning searches it and
        # diagnose_run reads which of its bounds the run ended on.
        limits = {
            'Kp': (Kmin / Kp_base, Kmax / Kp_base) if Kp_base > 0 else (Kmin, Kmax),
            'Ki': (Kmin / Ki, Kmax / Ki),
            'Kd': (Kmin / Kd_base, Kmax / Kd_base) if Kd_base > 0 else (Kmin, Kmax),
        }

        Fp_hist, Fi_hist, Fd_hist = pid_tuning(
            desc, tau, K, L, Ts,
            Kp_base, Ki, Kd_base,
            dtype='y', Tsim=Tsim, n_iter=n_iter,
            Fp_limits=limits['Kp'],
            Fi_limits=limits['Ki'],
            Fd_limits=limits['Kd'],
            Nbar=(Nbar0, Nbar1, Nbar2),
            beta=beta_tune,
            simtype=int(cfg.get('simtype', 0)),
            minu=float(cfg.get('minu', -1.0)),
            maxu=float(cfg.get('maxu', 1.0)),
            dist_a=dist_a, dist_b=dist_b,
            delta=delta_tune, eps=eps_tune,
            on_iteration=on_iteration,
        )

        Kp_traj = np.clip(Fp_hist * Kp_base, Kmin, Kmax)
        Ki_traj = np.clip(Fi_hist * Ki, Kmin, Kmax)
        Kd_traj = np.clip(Fd_hist * Kd_base, Kmin, Kmax)
        # A gain the structure doesn't use has base 0, which the clamp lifts to
        # Kmin — plot nothing rather than a flat line at a value never used.
        fig_gains_hist = _build_gains_history_fig(
            Kp_traj if Kp_base > 0 else None,
            Ki_traj,
            Kd_traj if Kd_base > 0 else None)

        # Gains the selected controller type doesn't tune keep their prior
        # slider value instead of being overwritten.
        out_Kp_log = _log_gain(Kp_traj[-1]) if ctype != 'I' else Kp_log
        out_Kd_log = _log_gain(Kd_traj[-1]) if ctype == 'PID' else Kd_log

        # What the run itself turned out to say. A multiplier that finished on a
        # bound is the difference between "converged" and "ran out of box", and
        # the status line has claimed the former unconditionally until now.
        run_findings = diagnose_run(
            ctype,
            {'Kp': float(Fp_hist[-1]), 'Ki': float(Fi_hist[-1]),
             'Kd': float(Fd_hist[-1])},
            limits,
            {'Kp': float(Kp_traj[-1]), 'Ki': float(Ki_traj[-1]),
             'Kd': float(Kd_traj[-1])},
            last_feats[0] or [],
            gain_box=(Kmin, Kmax),
        )
        status = f'Tuned ({n_iter} iter)' + _status_suffix(run_findings)

        return (out_Kp_log, _log_gain(Ki_traj[-1]), out_Kd_log, status,
                fig_gains_hist, False, no_update,
                _render_findings(gate_warnings + run_findings))
