"""
RoboPID — Dash web app entry point.

Usage:
    python app.py [--tau "5,5,5,5"] [--K 1.25] [--Td 8] [--Ts 1] [--ctype PID]

Then open http://localhost:8050 in your browser.
"""

import argparse
import numpy as np
import dash
import dash_bootstrap_components as dbc

from layout import make_layout
from callbacks import register_callbacks


def parse_args():
    p = argparse.ArgumentParser(description='RoboPID')
    p.add_argument('--tau', default='[5,5,5,5]',
                   help='Time constants, e.g. "[5,5,5,5]" or "10"')
    p.add_argument('--K', type=float, default=1.25, help='Plant gain')
    p.add_argument('--Td', type=float, default=8.0, help='Dead time')
    p.add_argument('--Ts', type=float, default=1.0, help='Sampling period')
    p.add_argument('--ctype', default='PID', choices=['I', 'PI', 'PID'],
                   help='Controller type')
    p.add_argument('--port', type=int, default=8050, help='Port')
    p.add_argument('--debug', action='store_true', help='Enable Dash debug mode')
    return p.parse_args()


def main():
    args = parse_args()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title='RoboPID',
    )

    app.layout = make_layout(
        default_tau=args.tau,
        default_K=str(args.K),
        default_Td=str(args.Td),
        default_ctype=args.ctype,
    )

    register_callbacks(app, default_Ts=args.Ts)

    print(f'\n  RoboPID running at http://localhost:{args.port}')
    print(f'  Plant: tau={args.tau}, K={args.K}, Td={args.Td}, Ts={args.Ts}')
    print(f'  Controller: {args.ctype}\n')

    app.run(debug=args.debug, port=args.port)


if __name__ == '__main__':
    main()
