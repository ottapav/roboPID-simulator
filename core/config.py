"""Config file reader and disturbance model builder."""

import os
import numpy as np
import scipy.linalg

DEFAULTS = {
    'simtype': 0,
    'minu': -1.0,
    'maxu': 1.0,
    'lipsch_const': 0.0,
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


def build_noise_model(tau: float, std: float, Ts: float) -> tuple[float, float]:
    """
    Compute first-order filtered white-noise model matrices (noise_A, noise_B)
    for a given filter time constant and target stationary standard deviation.

    Mirrors MATLAB:
        A = [-1/tau, 1/tau; 0, 0];
        M = expm(A * Ts);
        noise_A = M(1,1); noise_B = M(1,2);
        X = dlyap(noise_A, noise_B*noise_B');
        noise_B = noise_B / sqrt(X) * std;
    """
    A_cont = np.array([[-1.0 / tau, 1.0 / tau],
                        [0.0, 0.0]])
    M = scipy.linalg.expm(A_cont * Ts)
    noise_A = float(M[0, 0])
    noise_B = float(M[0, 1])

    if std == 0.0:
        return noise_A, 0.0

    X = scipy.linalg.solve_discrete_lyapunov(
        np.array([[noise_A]]), np.array([[noise_B ** 2]])
    )
    noise_B = noise_B / float(np.sqrt(X[0, 0])) * std
    return noise_A, noise_B
