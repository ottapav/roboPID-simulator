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
import contextlib
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

_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
# The diskcache itself lives in its own subdirectory, so the bookkeeping files
# below can sit next to it without diskcache ever seeing them.
_CACHE_DIR = os.path.join(_CACHE_ROOT, 'jobs')
_STAMP = os.path.join(_CACHE_ROOT, 'invocation')
_LOCK = os.path.join(_CACHE_ROOT, 'startup.lock')
_CACHE_KEEP = frozenset(os.path.basename(p) for p in (_CACHE_DIR, _STAMP, _LOCK))

try:  # POSIX
    import fcntl

    def _flock(fd, exclusive):
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_UN)
except ImportError:  # Windows
    import msvcrt

    def _flock(fd, exclusive):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK if exclusive else msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def _startup_lock():
    """
    Serialize cache setup across workers importing this module at the same time.

    Best effort: if the lock can't be taken (unsupported filesystem, or the
    Windows 10-second retry running out) we proceed anyway. Losing the lock
    only risks two workers clearing the same already-empty cache at startup,
    which is harmless; blocking the app's import would not be.
    """
    fd = os.open(_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    locked = False
    try:
        with contextlib.suppress(OSError):
            _flock(fd, True)
            locked = True
        yield
    finally:
        if locked:
            with contextlib.suppress(OSError):
                _flock(fd, False)
        os.close(fd)


def _invocation_id() -> str:
    """
    Token identifying one launch of the app, identical across all its workers.

    systemd exports a fresh `INVOCATION_ID` per service start and Gunicorn
    passes its environment down, so every worker of one start agrees on it —
    including workers respawned later by max-requests or a timeout.
    """
    inv = os.environ.get('INVOCATION_ID')
    if inv:
        return inv
    if __name__ == '__main__':
        # Direct `python app.py`: one process, and a fresh cache every run.
        return f'run-{os.getpid()}'
    # Bare Gunicorn with no systemd: the master's pid is shared by this
    # launch's workers and changes when the host is restarted.
    return f'ppid-{os.getppid()}'


def _sweep_legacy_layouts() -> None:
    """
    Drop the leftovers of the cache layouts this app used before `jobs/`.

    Earlier versions put a diskcache straight into `.cache/` and, before that,
    one per process in `.cache/pid-<pid>/`; neither is read any more, so their
    files would otherwise sit there forever. `.cache/` is this app's private
    scratch directory, so anything in it we don't own is by definition stale.
    Best effort throughout: on Windows a directory a live worker still holds
    open cannot be removed, and that is fine — this is housekeeping, not
    correctness.
    """
    for name in os.listdir(_CACHE_ROOT):
        if name in _CACHE_KEEP:
            continue
        stale = os.path.join(_CACHE_ROOT, name)
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                os.unlink(stale)


def _make_cache() -> diskcache.Cache:
    """
    One background-callback cache shared by every worker, cleared once per launch.

    It has to be shared. Gunicorn runs `--workers 2` with no `--preload`, so
    the worker that starts a Tune job is often not the worker that serves the
    browser's follow-up progress/result polls. With a per-process cache that
    second worker looks into a different directory, finds nothing, and the
    click appears to do nothing at all (server-side this also surfaced as
    `sqlite3.OperationalError: no such table: Cache` while writing progress).

    It also has to be cleared, but only once. DiskcacheManager keys jobs by
    (callback source + arg values), not by time, and the cache outlives a
    restart. The app's default inputs are identical on every launch, so the
    first Tune click of a new session can hash to a stale completed job from a
    previous run: Dash returns that result instantly and kills the freshly
    spawned worker before it streams any progress, leaving the sliders/status
    at their (correct but stale) final values while the plots never get
    patched. Clearing per *launch* instead of per *process* avoids that
    without letting a recycled worker wipe another worker's in-flight job.
    """
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    invocation = _invocation_id()

    with _startup_lock():
        cache = diskcache.Cache(_CACHE_DIR)
        stamped = None
        with contextlib.suppress(OSError):
            with open(_STAMP, encoding='utf-8') as fh:
                stamped = fh.read().strip()
        if stamped != invocation:
            cache.clear()
            _sweep_legacy_layouts()
            with open(_STAMP, 'w', encoding='utf-8') as fh:
                fh.write(invocation)
    return cache


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
