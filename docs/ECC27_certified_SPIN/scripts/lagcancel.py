"""Exact record of the lag-cancelled loop  de/dt = -k e(t-1)  (L_d = 1), all k at once."""
import numpy as np

def records(ks, dt=1e-3, tend=80.0):
    ks = np.asarray(ks, float); n = int(round(tend / dt)); nd = int(round(1 / dt))
    e = np.ones((n + 1, len(ks)))
    for i in range(nd + 1, n + 1):              # method of steps, trapezoid on the delayed term
        e[i] = e[i - 1] - ks * dt * 0.5 * (e[i - 1 - nd] + e[i - nd])
    return np.arange(n + 1) * dt, e

def Ms(k):
    x = np.linspace(1e-4, 60, 600000)
    return float(1 / np.min(np.sqrt(x * x - 2 * k * x * np.sin(x) + k * k) / x))
