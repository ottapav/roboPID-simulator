"""Dash layout — RoboPID educational web app."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

SLIDER_MARKS = {-2: '0.01', -1: '0.1', 0: '1', 1: '10'}
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
            id=sid, min=-2, max=1, step=0.01, value=0.0,
            marks=SLIDER_MARKS, updatemode='drag',
            tooltip={
                'placement': 'bottom', 'always_visible': True,
                'transform': 'logGain',
            },
        ),
    ], width=12, id=col_id)


def _plant_card(default_tau, default_K, default_L, default_noise_tau) -> dbc.Card:
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

    presets_row = dbc.Row([
        dbc.Col(html.Small('Battery presets:', style={'color': '#777'}),
                width='auto', className='align-self-center'),
        _preset_button('P1', 'Lag-dominant: e^-s / ((10s+1)(s+1)^3)'),
        _preset_button('P2', 'Balanced: 1.25 e^-8s / (5s+1)^4'),
        _preset_button('P3', 'Delay-dominant: e^-10s / ((2s+1)(s+1)^3)'),
        _preset_button('P4', 'High-order slow: e^-4s / (8s+1)^6'),
    ], className='mb-2 g-1 align-items-center')

    return dbc.Card([
        dbc.CardHeader(html.Strong('Plant', style=CARD_TITLE_STYLE), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            formula,
            presets_row,
            _row('τ (time constants)', dcc.Input(
                id='input-tau', type='text', value=default_tau,
                debounce=True, style={'width': '100%'},
                placeholder='e.g. [5,5,5,5] or 10',
            )),
            _row('K (static gain)', dcc.Input(
                id='input-K', type='text', value=f'{float(default_K):.2f}',
                debounce=True, style={'width': '100%'},
            )),
            _row('L (dead time)', dcc.Input(
                id='input-L', type='text', value=f'{float(default_L):.2f}',
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
                id='input-noise-std', type='text', value=f'{1.0:.2f}',
                debounce=True, disabled=True, style={'width': '100%'},
            )),
            _row('τ (noise filter)', dcc.Input(
                id='input-noise-tau', type='text', value=f'{float(default_noise_tau):.2f}',
                debounce=True, disabled=True, style={'width': '100%'},
            )),
        ]),
    ], className='h-100')


def _controller_card(default_ctype, default_limits, default_niter,
                     default_tuner_params, default_box=(0.01, 10.0)) -> dbc.Card:
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


def make_layout(default_tau: str = '[5,5,5,5]',
                default_K: str = '1.25',
                default_L: str = '8.0',
                default_noise_tau: float = 0.5,
                default_ctype: str = 'PID',
                default_limits: tuple = (0.5, 0.75, 1.0),
                default_niter: int = 200,
                default_tuner_params: tuple = (0.1, 0.02, 0.1),
                default_box: tuple = (0.01, 10.0)):
    """Build and return the full app layout."""
    return dbc.Container(fluid=False, style={
        'width': '100%', 'maxWidth': '800px', 'overflowX': 'hidden',
    }, children=[

        # ── Title ──────────────────────────────────────────────────────────
        dbc.Row(dbc.Col(html.H4('RoboPID', className='mt-2 mb-2'))),

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
            dbc.Col(_plant_card(default_tau, default_K, default_L, default_noise_tau),
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

        # ── Step Response ──────────────────────────────────────────────────
        dbc.Row(dbc.Col(_step_response_card(), width=12), className='mb-1'),

        # ── Features ──────────────────────────────────────────────────────
        dbc.Row(dbc.Col(_features_card(), width=12), className='mb-1'),

        # ── Tuning History ──────────────────────────────────────────────────
        dbc.Row(dbc.Col(_gains_history_card(), width=12), className='mb-1'),
    ])
