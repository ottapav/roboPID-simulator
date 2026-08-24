"""Dash layout — RoboPID educational web app."""

from __future__ import annotations

import math

from dash import dcc, html
import dash_bootstrap_components as dbc

from core.params import (
    BETA, DELTA, EPS, GAIN_BOX, NBAR, N_ITER_BY_CTYPE, fmt2, gain_slider_marks,
)

# The gain sliders are log-scale over the gain box, so their range and marks
# are derived from it rather than restated — update_gain_slider_range rebuilds
# both with the same helper when the box changes at runtime.
GAIN_LOG_MIN = math.log10(GAIN_BOX[0])
GAIN_LOG_MAX = math.log10(GAIN_BOX[1])
SLIDER_MARKS = gain_slider_marks(*GAIN_BOX)
GRAPH_STYLE = {'height': '300px'}
GRAPH_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d',
        'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
    ],
}
BTN = 'btn btn-sm'
CARD_HEADER_STYLE = {
    'minHeight': '48px', 'display': 'flex', 'alignItems': 'center',
    'paddingTop': '0.35rem', 'paddingBottom': '0.35rem',
}
CARD_TITLE_STYLE = {'fontSize': '14px'}
PARAM_GROUP_LABEL_WIDTH = '78px'


def _slider(sid: str, label: str, col_id: str | None = None) -> dbc.Col:
    return dbc.Col([
        html.Label(label, style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '2px'}),
        dcc.Slider(
            id=sid, min=GAIN_LOG_MIN, max=GAIN_LOG_MAX, step=0.01, value=0.0,
            marks=SLIDER_MARKS, updatemode='drag',
            tooltip={
                'placement': 'bottom', 'always_visible': True,
                'transform': 'logGain',
            },
        ),
    ], width=12, id=col_id)


def _plant_card(default_tau, default_K, default_L, default_noise_tau,
                default_tsim, default_ts) -> dbc.Card:
    formula = html.P([
        'P(s) = K · e',
        html.Sup('−L·s'),
        ' / ∏ (τᵢ·s + 1)',
    ], style={'fontStyle': 'italic', 'fontSize': '13px', 'color': '#555', 'marginBottom': '10px'})

    noise_formula = html.P([
        'n(s) = σ·√(2τ) / (τ·s + 1) · w(s)',
    ], style={'fontStyle': 'italic', 'fontSize': '13px', 'color': '#555', 'marginBottom': '10px'})

    def _row(label, input_elem):
        return dbc.Row([
            dbc.Col(html.Label(label, style={'fontSize': '13px'}), width=5),
            dbc.Col(input_elem, width=7),
        ], className='mb-2 align-items-center')

    def _preset_button(pid: str, tooltip: str) -> dbc.Col:
        return dbc.Col(html.Button(
            pid, id=f'btn-{pid.lower()}', n_clicks=0, title=tooltip,
            className=f'{BTN} btn-outline-secondary py-0',
            style={'fontSize': '11px'},
        ), width='auto')

    def _grid_field(label, field_id, value, tooltip, cls='') -> dbc.Col:
        return dbc.Col(html.Div([
            html.Small(f'{label} ', style={'color': '#777'}),
            dcc.Input(
                id=field_id, type='text', value=value, debounce=True,
                style={'width': '64px', 'fontSize': '12px', 'padding': '0 4px'},
            ),
        ], className='d-flex align-items-center gap-1', title=tooltip),
            width='auto', className=f'{cls} align-self-center'.strip())

    presets_row = dbc.Row([
        dbc.Col(html.Small('Battery presets:', style={'color': '#777'}),
                width='auto', className='align-self-center'),
        _preset_button('P1', 'Lag-dominant: e^-s / ((10s+1)(s+1)^3)'),
        _preset_button('P2', 'Balanced: 1.25 e^-8s / (5s+1)^4'),
        _preset_button('P3', 'Delay-dominant: e^-10s / ((2s+1)(s+1)^3)'),
        _preset_button('P4', 'High-order slow: e^-4s / (8s+1)^6'),
    ], className='mb-2 g-1 align-items-center')

    return dbc.Card([
        # The simulation grid describes how the plant is *observed*, not what it
        # is, so it sits in the header alongside the card title rather than in
        # the body among the τ/K/L fields that define the plant itself. The
        # sample count is a constant (params.N_POINTS) and is not shown at all,
        # which leaves Tsim as the one editable grid value: it is proposed from
        # τ and L (see auto_grid), rewritten whenever those change — hence ↺ to
        # get the proposal back — and Ts follows as Tsim/499, read-only.
        #
        # The Row needs w-100 because CARD_HEADER_STYLE makes the header a flex
        # container — without it the Row won't span and ms-auto can't push the
        # grid to the right edge (same reason the Controller header carries it).
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Strong('Plant', style=CARD_TITLE_STYLE),
                        width='auto', className='align-self-center'),
                _grid_field('Tsim', 'input-tsim', default_tsim,
                            'Simulation horizon — proposed as 10 plant spans '
                            '(sum τ + L), editable', cls='ms-auto'),
                dbc.Col(html.Small(
                    ['Ts ', html.Span(id='display-ts', children=default_ts,
                                      style={'fontWeight': 'bold', 'color': '#333'})],
                    style={'color': '#777', 'whiteSpace': 'nowrap'},
                    title='Sampling period — Tsim / 499, shown to two decimals'),
                    width='auto', className='align-self-center'),
                dbc.Col(html.Button(
                    '↺', id='btn-grid-auto', n_clicks=0,
                    title='Back to the automatic grid',
                    className=f'{BTN} btn-link p-0',
                    style={'fontSize': '14px', 'lineHeight': 1, 'color': '#777'},
                ), width='auto', className='align-self-center'),
            ], align='center', className='g-2 flex-nowrap w-100'),
            style=CARD_HEADER_STYLE,
        ),
        dbc.CardBody([
            formula,
            presets_row,
            _row('τ (time constants)', dcc.Input(
                id='input-tau', type='text', value=default_tau,
                debounce=True, style={'width': '100%'},
                placeholder='e.g. [5,5,5,5] or 10',
            )),
            _row('K (static gain)', dcc.Input(
                id='input-K', type='text', value=fmt2(default_K),
                debounce=True, style={'width': '100%'},
            )),
            _row('L (dead time)', dcc.Input(
                id='input-L', type='text', value=fmt2(default_L),
                debounce=True, style={'width': '100%'},
            )),
            html.Hr(style={'borderColor': '#e0e0e0', 'margin': '10px 0'}),
            dbc.Row([
                dbc.Col(html.Div([
                    dbc.Checkbox(
                        id='noise-enabled', value=False,
                        className='mb-0', style={'marginRight': '6px'},
                    ),
                    html.Label('Output noise', style={'fontSize': '13px', 'marginBottom': 0}),
                ], className='d-flex align-items-center'), width=12),
            ], className='mb-2'),
            noise_formula,
            _row('σ (noise std, %)', dcc.Input(
                id='input-noise-std', type='text', value=fmt2(1.0),
                debounce=True, disabled=True, style={'width': '100%'},
            )),
            _row('τ (noise filter)', dcc.Input(
                id='input-noise-tau', type='text', value=fmt2(default_noise_tau),
                debounce=True, disabled=True, style={'width': '100%'},
            )),
        ]),
    ], className='h-100')


def _controller_card(default_ctype, default_limits, default_niter,
                     default_tuner_params, default_box=GAIN_BOX) -> dbc.Card:
    Nbar0, Nbar1, Nbar2 = default_limits
    default_eps, default_delta, default_beta = default_tuner_params
    default_kmin, default_kmax = default_box

    controller_formula = html.P([
        'C(s) = Kp + Ki / s + Kd · s',
    ], style={'fontStyle': 'italic', 'fontSize': '13px', 'color': '#555', 'marginBottom': '0'})

    def _reset_button(bid: str, label: str, tooltip: str) -> html.Button:
        return html.Button(
            label, id=bid, n_clicks=0, title=tooltip,
            className=f'{BTN} btn-outline-secondary py-0',
            style={'fontSize': '11px'},
        )

    controller_header_row = dbc.Row([
        dbc.Col(controller_formula, className='flex-grow-1', style={'minWidth': 0}),
        dbc.Col(html.Div([
            _reset_button('btn-reset-controller', 'Reset controller',
                         'Reset Kp/Ki/Kd gains to their defaults'),
            _reset_button('btn-reset-tuner', 'Reset tuner',
                         'Reset tuning parameters to their defaults and clear the Tuning History'),
        ], className='d-flex', style={'gap': '6px'}), width='auto'),
    ], align='center', justify='between', className='g-2 flex-nowrap mb-2')

    def _limit_input(sid, val):
        return dcc.Input(id=sid, type='number', value=val,
                         debounce=True, step=0.05, min=0.0, max=5.0,
                         style={'width': '60px'})

    def _tuner_param_input(sid, val, step):
        return dcc.Input(id=sid, type='number', value=val,
                         debounce=True, step=step, min=0.0, max=5.0,
                         style={'width': '60px'})

    def _box_input(sid, val):
        return dcc.Input(id=sid, type='number', value=val,
                         debounce=True, step='any', min=0.0001, max=1000.0,
                         style={'width': '70px'})

    def _param_row(label, *pairs):
        """One labeled group; pairs is a sequence of (short_label, input)
        joined with ', ' delimiters. The label column has a fixed width so
        the first parameter of every group lines up across rows."""
        children = []
        for i, (plabel, inp) in enumerate(pairs):
            if i > 0:
                children.append(html.Span(', ', style={'color': '#777'}))
            children.append(html.Small(f'{plabel} ', style={'color': '#777'}))
            children.append(inp)
        return dbc.Row([
            dbc.Col(html.Small(f'{label}:', style={'color': '#777'}),
                    width='auto', style={'width': PARAM_GROUP_LABEL_WIDTH}),
            dbc.Col(children, width='auto',
                    className='d-flex align-items-center flex-wrap gap-1'),
        ], align='center', className='mb-2 g-2')

    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Strong('Controller', style=CARD_TITLE_STYLE),
                        width='auto', className='align-self-center'),
                dbc.Col(html.Button(
                    'TUNE', id='btn-tune', n_clicks=0,
                    className='btn btn-success',
                    style={'fontSize': '14px', 'fontWeight': 'bold', 'padding': '4px 18px'},
                ), width='auto'),
                dbc.Col(html.Div(id='tune-status',
                                 style={
                                     'fontSize': '11px', 'color': '#555',
                                     'whiteSpace': 'nowrap', 'overflow': 'hidden',
                                     'textOverflow': 'ellipsis',
                                 }),
                        className='flex-grow-1', style={'minWidth': 0}),
                dbc.Col(dcc.Dropdown(
                    id='dropdown-ctype',
                    options=[{'label': t, 'value': t} for t in ('I', 'PI', 'PID')],
                    value=default_ctype, clearable=False,
                    style={'minWidth': '68px', 'fontSize': '12px'},
                ), width='auto', className='ms-auto'),
            ], align='center', justify='between', className='g-2 flex-nowrap w-100'),
            style=CARD_HEADER_STYLE,
        ),
        dbc.CardBody([
            controller_header_row,
            # Sliders
            dbc.Row([
                _slider('slider-kp', 'Kp', col_id='col-kp'),
                _slider('slider-ki', 'Ki', col_id='col-ki'),
                _slider('slider-kd', 'Kd', col_id='col-kd'),
            ], className='mb-3'),

            # Feature limits (N̄0, N̄1, N̄2)
            _param_row('Limits',
                      ('Γ0', _limit_input('input-nbar0', Nbar0)),
                      ('Γ1', _limit_input('input-nbar1', Nbar1)),
                      ('Γ2', _limit_input('input-nbar2', Nbar2))),

            # Gain boundary [Kmin, Kmax]: bounds the tuning search box and
            # the live gain clamp during a run (paper Section 6: "widening
            # the box is the first remedy" when a multiplier stalls at its
            # bound).
            _param_row('Gain box',
                      ('Kmin', _box_input('input-kmin', default_kmin)),
                      ('Kmax', _box_input('input-kmax', default_kmax))),

            # Tuner settings
            _param_row('Settings',
                      ('Trunc ε', _tuner_param_input('input-eps', default_eps, 0.01)),
                      ('Step γ', _tuner_param_input('input-beta', default_beta, 0.05))),

            # Simulation (left) | Guard mode (right): checked = Guarded (delta
            # settable, Definition 4's settling-anchored window); unchecked =
            # Unguarded (delta pinned at 0, the raw-window count of
            # Definition 1) -- paper Section "Well-posedness of the count".
            # Settle δ lives next to the checkbox since unchecking
            # pins/disables it directly.
            dbc.Row([
                dbc.Col(html.Small('Simulation:', style={'color': '#777'}),
                        width='auto', style={'width': PARAM_GROUP_LABEL_WIDTH}),
                dbc.Col(html.Div([
                    html.Small('Iter', style={'color': '#777'}),
                    dcc.Input(id='input-niter', type='number', value=default_niter,
                             debounce=True, step=10, min=10, max=2000,
                             style={'width': '70px'}),
                ], className='d-flex align-items-center', style={'gap': '4px'}),
                        width='auto'),
                dbc.Col(html.Div([
                    dbc.Checkbox(
                        id='guard-mode', value=True,
                        className='mb-0', style={'marginRight': '0'},
                    ),
                    html.Small('Guard:', style={'color': '#777'}),
                    html.Small('Settle δ', style={'color': '#777'}),
                    _tuner_param_input('input-delta', default_delta, 0.01),
                ], className='d-flex align-items-center', style={'gap': '4px'}),
                        className='ms-auto',
                        width='auto'),
            ], align='center', className='mb-2 g-2 flex-wrap'),
        ]),
    ], className='h-100')


def _step_response_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Strong('Step Response', style=CARD_TITLE_STYLE), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dcc.Graph(id='graph-time', style=GRAPH_STYLE,
                      config=GRAPH_CONFIG),
        ]),
    ])


def _features_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Strong('Pachner plots (Γ0, Γ1, Γ2)', style=CARD_TITLE_STYLE), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Graph(id='graph-f1', style=GRAPH_STYLE,
                                  config=GRAPH_CONFIG), width=12, sm=6, md=4),
                dbc.Col(dcc.Graph(id='graph-f2', style=GRAPH_STYLE,
                                  config=GRAPH_CONFIG), width=12, sm=6, md=4),
                dbc.Col(dcc.Graph(id='graph-f3', style=GRAPH_STYLE,
                                  config=GRAPH_CONFIG), width=12, sm=6, md=4),
            ]),
        ]),
    ])


def _gains_history_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Strong('Tuning History', style=CARD_TITLE_STYLE), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dcc.Graph(id='graph-gains-history', style=GRAPH_STYLE,
                      config=GRAPH_CONFIG),
        ]),
    ])


_BIBTEX = """@misc{pachner2026robopid,
  author = {Pachner, Daniel and Otta, Pavel and Dostál, Jiří and Havlena, Vladimír},
  title  = {Model-Free PID Tuning by Step-Response Inspection},
  note   = {Submitted to Journal of Process Control},
  year   = {2026},
  url    = {https://github.com/ottapav/roboPID-simulator}
}"""


def _tune_error_modal() -> dbc.Modal:
    """Where a rejected Tune click explains itself.

    A blocking finding means the tuning problem has no answer for this plant,
    which is more than the one-line `tune-status` slot can carry and more than
    the warning line -- which the next slider drag overwrites -- should be
    trusted to hold. It is the only modal in the app; everything advisory still
    goes to `input-warning`.

    size='lg' because the phase message runs to three paragraphs plus a
    three-item fix list, and the point of it is that it is readable.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('Plant outside the tunable class'),
                        close_button=True),
        dbc.ModalBody(id='tune-error-body'),
        dbc.ModalFooter(dbc.Button('Close', id='btn-tune-error-close',
                                   className='ms-auto', n_clicks=0)),
    ], id='tune-error-modal', is_open=False, centered=True, size='lg',
        scrollable=True)


def _footer() -> html.Div:
    return html.Div([
        html.Hr(style={'borderColor': '#e0e0e0', 'margin': '18px 0 10px'}),
        html.P([
            'If roboPID is useful in your research, please consider citing '
            'the paper it implements: ',
            html.Strong('D. Pachner, P. Otta, J. Dostál, '
                        'V. Havlena, “Model-Free PID Tuning by Step-Response '
                        'Inspection,” submitted to Journal of Process Control.'),
        ], style={'fontSize': '12px', 'color': '#777', 'marginBottom': '6px'}),
        html.Pre(_BIBTEX, style={
            'fontSize': '11px', 'color': '#555', 'backgroundColor': '#f8f9fa',
            'border': '1px solid #e0e0e0', 'borderRadius': '4px',
            'padding': '8px 10px', 'whiteSpace': 'pre-wrap',
            'marginBottom': '14px',
        }),
    ])


def make_layout(default_tau: str = '[5,5,5,5]',
                default_K: str = '1.25',
                default_L: str = '8.0',
                default_noise_tau: float = 0.5,
                default_tsim: str = '280',
                default_ts: str = '0.5611',
                default_ctype: str = 'PID',
                default_limits: tuple = NBAR,
                default_niter: int = N_ITER_BY_CTYPE['PID'],
                default_tuner_params: tuple = (EPS, DELTA, BETA),
                default_box: tuple = GAIN_BOX):
    """Build and return the full app layout."""
    return dbc.Container(fluid=False, style={
        'width': '100%', 'maxWidth': '800px', 'overflowX': 'hidden',
    }, children=[

        # Set for the duration of a Tune run via run_tune's `running=` list.
        # update_figures_patch reads it as State and short-circuits, so the
        # slider values the tuner streams don't each trigger a redundant
        # simulation racing the patches the tuner is already pushing.
        dcc.Store(id='tuning-active', data=False),

        # The resolved simulation grid: {'sig', 'Tsim', 'Ts', 'notes'}. Written
        # by commit_grid from whatever the header fields hold, read by all three
        # simulating callbacks, so the proposal/override arbitration happens
        # once rather than in each of them. 'sig' stamps the plant the grid was
        # derived for, which is how update_figures_patch tells a fresh grid from
        # one belonging to the previous τ/L.
        dcc.Store(id='grid-store'),

        # ── Title ──────────────────────────────────────────────────────────
        dbc.Row(dbc.Col(html.H4('roboPID', className='mt-2 mb-2'))),

        # ── Description ───────────────────────────────────────────────────
        dbc.Row(dbc.Col(html.P([
            'roboPID is the open academic reference implementation of '
            'SPIN-based PID tuning — a browser-hosted simulator and Python '
            'library, released alongside the paper so readers can inspect '
            'and reproduce the method. It closes a loop around a simulated '
            'process, steps the setpoint, forms the three Pachner plots of '
            'the recorded error, computes each band’s turn index, and '
            'applies the triangular rule, iterating with fixed default '
            'constants. Because it displays the step response, the '
            'portraits with their counts and limits, and the gain '
            'trajectory, every decision the algorithm makes is visible '
            'rather than inferred. It reproduces every figure and table in '
            'the paper exactly, and is available at ',
            html.A('github.com/ottapav/roboPID-simulator',
                  href='https://github.com/ottapav/roboPID-simulator',
                  target='_blank', rel='noopener noreferrer'),
            '.',
        ], style={'fontSize': '13px', 'color': '#555', 'marginBottom': '14px'}))),

        # ── Plant | Controller cards ────────────────────────────────────────
        dbc.Row([
            dbc.Col(_plant_card(default_tau, default_K, default_L, default_noise_tau,
                                default_tsim, default_ts),
                    width=12, md=5, className='mb-2'),
            dbc.Col(_controller_card(default_ctype, default_limits, default_niter,
                                     default_tuner_params, default_box),
                    width=12, md=7, className='mb-2'),
        ], className='mb-1'),

        # ── Input validation warning ─────────────────────────────────────────
        dbc.Row(dbc.Col(html.Div(
            id='input-warning',
            style={'fontSize': '12px', 'color': '#b45309', 'minHeight': '16px'},
        ), width=12), className='mb-1'),

        # ── Advisory findings from the last Tune run ─────────────────────────
        # A separate surface from input-warning, and deliberately so: run_tune's
        # final return moves the sliders, which re-triggers update_figures_patch
        # as soon as `tuning-active` clears -- and that callback owns
        # input-warning, so anything the tuner wrote there would be overwritten
        # the moment it finished. This Div has exactly one writer.
        dbc.Row(dbc.Col(html.Div(
            id='tune-findings',
            style={'fontSize': '12px', 'color': '#b45309'},
        ), width=12), className='mb-1'),

        # Opened by run_tune when a gate rejects the click; see
        # core.admissibility.
        _tune_error_modal(),

        # ── Step Response ──────────────────────────────────────────────────
        dbc.Row(dbc.Col(_step_response_card(), width=12), className='mb-1'),

        # ── Features ──────────────────────────────────────────────────────
        dbc.Row(dbc.Col(_features_card(), width=12), className='mb-1'),

        # ── Tuning History ──────────────────────────────────────────────────
        dbc.Row(dbc.Col(_gains_history_card(), width=12), className='mb-1'),

        # ── Footer: citation note ─────────────────────────────────────────
        dbc.Row(dbc.Col(_footer(), width=12)),
    ])
