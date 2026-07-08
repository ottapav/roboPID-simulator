"""Config file reader and disturbance model builder."""

import os
import numpy as np
import scipy.linalg

DEFAULTS = {
    'simtype': 0,
    'minu': -1.0,
    'maxu': 1.0,
    'dist_tau': 120.0,
    'dist_std': 0.05,
    'lipsch_const': 0.0,
    'tune_step': 0.1,
}


def read_config(path: str) -> dict:
    """Parse whitespace-separated key float pairs; merge with defaults."""
    cfg = dict(DEFAULTS)
    if path and os.path.isfile(path):
        with open(path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        cfg[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    return cfg


def build_disturbance_model(cfg: dict, Ts: float) -> tuple[float, float]:
    """
    Compute first-order disturbance model matrices (dist_A, dist_B).

    Mirrors MATLAB:
        A = [-1/dist_tau, 1/dist_tau; 0, 0];
        M = expm(A * Ts);
        dist_A = M(1,1); dist_B = M(1,2);
        X = dlyap(dist_A, dist_B*dist_B');
        dist_B = dist_B / sqrt(X) * dist_std;
    """
    dist_tau = cfg['dist_tau']
    dist_std = cfg['dist_std']

    A_cont = np.array([[-1.0 / dist_tau, 1.0 / dist_tau],
                        [0.0, 0.0]])
    M = scipy.linalg.expm(A_cont * Ts)
    dist_A = float(M[0, 0])
    dist_B = float(M[0, 1])

    if dist_std == 0.0:
        return dist_A, 0.0

    X = scipy.linalg.solve_discrete_lyapunov(
        np.array([[dist_A]]), np.array([[dist_B ** 2]])
    )
    dist_B = dist_B / float(np.sqrt(X[0, 0])) * dist_std
    return dist_A, dist_B
