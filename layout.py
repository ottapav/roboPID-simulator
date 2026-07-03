"""Dash layout — RoboPID educational web app."""

from dash import dcc, html
import dash_bootstrap_components as dbc

SLIDER_MARKS = {0.01: '0.01', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
GRAPH_STYLE = {'height': '300px'}
BTN = 'btn btn-sm'


def _slider(sid: str, label: str) -> dbc.Col:
    return dbc.Col([
        html.Label(label, style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '2px'}),
        dcc.Slider(
            id=sid, min=0.01, max=5.0, step=0.01, value=1.0,
            marks=SLIDER_MARKS, updatemode='mouseup',
            tooltip={'placement': 'bottom', 'always_visible': True},
        ),
    ], width=12)


def _plant_card(default_tau, default_K, default_Td) -> dbc.Card:
    formula = html.P([
        'P(s) = K · e',
        html.Sup('−Td·s'),
        ' / ∏ (τᵢ·s + 1)',
    ], style={'fontStyle': 'italic', 'fontSize': '13px', 'color': '#555', 'marginBottom': '10px'})

    def _row(label, input_elem):
        return dbc.Row([
            dbc.Col(html.Label(label, style={'fontSize': '13px'}), width=5),
            dbc.Col(input_elem, width=7),
        ], className='mb-2 align-items-center')

    return dbc.Card([
        dbc.CardHeader(html.Strong('Plant')),
        dbc.CardBody([
            formula,
            _row('τ (time constants)', dcc.Input(
                id='input-tau', type='text', value=default_tau,
                debounce=True, style={'width': '100%'},
                placeholder='e.g. [5,5,5,5] or 10',
            )),
            _row('K (static gain)', dcc.Input(
                id='input-K', type='number', value=float(default_K),
                debounce=True, step=0.01, min=0.001, style={'width': '100%'},
            )),
            _row('Td (dead time)', dcc.Input(
                id='input-Td', type='number', value=float(default_Td),
                debounce=True, step=0.1, min=0.0, style={'width': '100%'},
            )),
        ]),
    ], className='h-100')


def _controller_card(default_ctype, default_limits) -> dbc.Card:
    lim1, lim2, lim3 = default_limits

    def _limit_input(sid, val):
        return dcc.Input(id=sid, type='number', value=val,
                         debounce=True, step=0.05, min=0.0, max=5.0,
                         style={'width': '60px'})

    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Strong('Controller'), width='auto', className='align-self-center'),
                dbc.Col(dcc.Dropdown(
                    id='dropdown-ctype',
                    options=[{'label': t, 'value': t} for t in ('I', 'PI', 'PID')],
                    value=default_ctype, clearable=False,
                    style={'minWidth': '80px'},
                ), width='auto', className='ms-auto'),
            ], align='center', justify='between', className='g-2'),
        ),
        dbc.CardBody([
            # Sliders
            dbc.Row([
                _slider('slider-kp', 'Kp ×'),
                _slider('slider-ki', 'Ki ×'),
                _slider('slider-kd', 'Kd ×'),
            ], className='mb-3'),

            # Buttons + results
            dbc.Row([
                dbc.Col(html.Button(
                    'Optimize', id='btn-optimize', n_clicks=0,
                    className=f'{BTN} btn-primary', style={'marginRight': '6px'},
                ), width='auto'),
                dbc.Col(html.Button(
                    'Tune', id='btn-tune', n_clicks=0,
                    className=f'{BTN} btn-success',
                ), width='auto'),
                dbc.Col(html.Div(id='gains-display',
                                 style={'fontFamily': 'monospace', 'fontSize': '13px',
                                        'color': '#333'}),
                        width='auto', className='align-self-center'),
                dbc.Col(html.Div(id='tune-status',
                                 style={'fontSize': '12px', 'color': '#555'}),
                        width='auto', className='align-self-center'),
            ], align='center', className='mb-2'),

            # Feature limits
            dbc.Row([
                dbc.Col(html.Small('Limits:', style={'color': '#777'}), width='auto'),
                dbc.Col([html.Small('F1: ', style={'color': '#777'}), _limit_input('limit-1', lim1)],
                        width='auto', className='d-flex align-items-center gap-1'),
                dbc.Col([html.Small('F2: ', style={'color': '#777'}), _limit_input('limit-2', lim2)],
                        width='auto', className='d-flex align-items-center gap-1'),
                dbc.Col([html.Small('F3: ', style={'color': '#777'}), _limit_input('limit-3', lim3)],
                        width='auto', className='d-flex align-items-center gap-1'),
            ], align='center'),
        ]),
    ], className='h-100')


def _step_response_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Strong('Step Response')),
        dbc.CardBody([
            dcc.Graph(id='graph-time', style=GRAPH_STYLE,
                      config={'displayModeBar': False}),
        ]),
    ])


def _features_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Strong('Features')),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Graph(id='graph-f1', style=GRAPH_STYLE,
                                  config={'displayModeBar': False}), width=6, md=4),
                dbc.Col(dcc.Graph(id='graph-f2', style=GRAPH_STYLE,
                                  config={'displayModeBar': False}), width=6, md=4),
                dbc.Col(dcc.Graph(id='graph-f3', style=GRAPH_STYLE,
                                  config={'displayModeBar': False}), width=6, md=4),
            ]),
        ]),
    ])


def make_layout(default_tau: str = '[5,5,5,5]',
                default_K: str = '1.25',
                default_Td: str = '8.0',
                default_ctype: str = 'PID',
                default_limits: tuple = (0.5, 0.75, 1.0)):
    """Build and return the full app layout."""
    return dbc.Container(fluid=False, style={'width': '800px', 'maxWidth': '800px'}, children=[

        # ── Title ──────────────────────────────────────────────────────────
        dbc.Row(dbc.Col(html.H4('RoboPID', className='mt-2 mb-2'))),

        # ── Plant | Controller cards ────────────────────────────────────────
        dbc.Row([
            dbc.Col(_plant_card(default_tau, default_K, default_Td),
                    width=12, md=4, className='mb-2'),
            dbc.Col(_controller_card(default_ctype, default_limits),
                    width=12, md=8, className='mb-2'),
        ], className='mb-1'),

        # ── Step Response ──────────────────────────────────────────────────
        dbc.Row(dbc.Col(_step_response_card(), width=12), className='mb-1'),

        # ── Features ──────────────────────────────────────────────────────
        dbc.Row(dbc.Col(_features_card(), width=12), className='mb-1'),

        # ── Hidden state ────────────────────────────────────────────────────
        dcc.Store(id='base-gains', data={'Kp': 1.0, 'Ki': 1.0, 'Kd': 0.0,
                                         'optimized': False}),
    ])
