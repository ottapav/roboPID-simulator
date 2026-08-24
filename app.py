"""
RoboPID — Dash web app entry point.

Local usage:
    python app.py [--tau "5,5,5,5"] [--K 1.25] [--L 8] [--ctype PID]

Then open http://localhost:8050 in your browser.

Render (or any Gunicorn host) imports this module and serves the `server`
object defined below, e.g.:
    gunicorn app:server --bind 0.0.0.0:$PORT
"""

import argparse
import glob
import os
import shutil
import numpy as np
import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import DiskcacheManager

from core.params import N_POINTS, fmt2, parse_tau
from core.signals import auto_grid
from layout import make_layout
from callbacks import register_callbacks


def parse_args():
    p = argparse.ArgumentParser(description='RoboPID')
    p.add_argument('--tau', default='[5,5,5,5]',
                   help='Time constants, e.g. "[5,5,5,5]" or "10"')
    p.add_argument('--K', type=float, default=1.25, help='Plant gain')
    p.add_argument('--L', type=float, default=8.0, help='Dead time')
    p.add_argument('--ctype', default='PID', choices=['I', 'PI', 'PID'],
                   help='Controller type')
    p.add_argument('--port', type=int, default=8050, help='Port')
    p.add_argument('--debug', action='store_true', help='Enable Dash debug mode')
    # parse_known_args so this doesn't choke on argv it doesn't own (e.g. Gunicorn's).
    args, _ = p.parse_known_args()
    return args


args = parse_args()

_CACHE_ROOT = os.path.join(os.path.dirname(__file__), '.cache')


def _make_cache() -> diskcache.Cache:
    """
    Per-process background-callback cache.

    DiskcacheManager keys jobs by (callback source + arg values), not by time,
    and the cache persists across restarts. Since the app's default inputs are
    identical on every launch, the first Tune click of a new session can hash
    to a stale, already-completed job from a previous run: Dash then returns
    that cached result instantly and kills the freshly spawned worker before
    it ever streams progress, so the sliders/status jump straight to the
    (correct but stale) final values while the plots never get patched.

    Giving each process its own directory avoids those cross-session
    collisions without a shared clear(). The Procfile runs gunicorn with
    --workers 2 and no --preload, so each worker imports this module
    separately; a single shared cache.clear() at import time would let a
    recycled worker (max-requests, timeout) wipe the cache out from under
    another worker's in-flight tuning job.
    """
    for stale in glob.glob(os.path.join(_CACHE_ROOT, 'pid-*')):
        # Best effort: a live worker holds its own directory open, and on
        # Windows that makes the removal fail. Skipping it is fine — the
        # sweep is housekeeping, not correctness.
        shutil.rmtree(stale, ignore_errors=True)
    return diskcache.Cache(os.path.join(_CACHE_ROOT, f'pid-{os.getpid()}'))


cache = _make_cache()
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='RoboPID',
    background_callback_manager=background_callback_manager,
)
# WSGI entry point for Gunicorn/Render: `gunicorn app:server`.
server = app.server

_default_tau = parse_tau(args.tau)[0]
# Seed the header's grid fields with the same proposal propose_grid would make,
# so the first paint is already consistent with what any τ/L edit produces.
_default_tsim, _default_ts = auto_grid(_default_tau, args.L)

app.layout = make_layout(
    default_tau=args.tau,
    default_K=str(args.K),
    default_L=str(args.L),
    default_noise_tau=0.1 * float(np.mean(_default_tau)),
    default_tsim=f'{_default_tsim:.4g}',
    default_ts=fmt2(_default_ts),
    default_ctype=args.ctype,
)

register_callbacks(app)


def main():
    # Render assigns the listen port dynamically via $PORT and expects the
    # app to bind on 0.0.0.0, not localhost.
    port = int(os.environ.get('PORT', args.port))

    print(f'\n  RoboPID running at http://localhost:{port}')
    print(f'  Plant: tau={args.tau}, K={args.K}, L={args.L}')
    # Same rendering as the header fields, so the banner and the GUI never
    # disagree about the grid the app started on. N is the fixed sample count
    # every simulation runs on; the GUI doesn't show it because it never moves.
    print(f'  Grid: N={N_POINTS}, Tsim={_default_tsim:.4g}, Ts={fmt2(_default_ts)}')
    print(f'  Controller: {args.ctype}\n')

    app.run(debug=args.debug, host='0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    main()
