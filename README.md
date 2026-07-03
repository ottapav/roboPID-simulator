# roboPID-simulator

**RoboPID** is an interactive Dash web app that simulates PID controller
algorithms — design, optimization, and auto-tuning — against a
first-order-plus-dead-time (FOPTD) process model.

## Overview

The app models a process as a cascade of first-order lags with a static gain
and dead time:

```
P(s) = K * exp(-Td*s) / prod(s*tau_i + 1)
```

You set the plant parameters (`tau`, `K`, `Td`) and pick a controller type
(I, PI, or PID), then either:

- **Optimize** — finds gains that minimize a 6-term weighted cost over the
  closed-loop step response (tracking error, overshoot, control effort,
  smoothness), via SciPy SLSQP.
- **Tune** — iteratively nudges the optimized gains up or down based on
  closed-loop robustness "features" (phase-plane/encirclement metrics) and
  path-ratio sluggishness checks, streaming progress live as it runs.

RoboPID is built as an educational algorithm simulator for exploring PID
tuning behavior interactively, making the effect of each gain and each
tuning pass visible in real time.

## Features

- FOPTD plant modeling with an arbitrary number of cascaded time constants
- PID / PI / I gain optimization (SciPy SLSQP)
- Iterative, feature-driven auto-tuning with live progress streaming
- Interactive Dash GUI: plant/controller inputs, Kp/Ki/Kd multiplier
  sliders, live step-response plot, and three feature/phase-plane plots
- Runtime-configurable simulation settings via `robopid.config`

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`: `numpy`, `scipy`, `dash`,
  `dash-bootstrap-components`, `plotly`, `diskcache`, `psutil`, `multiprocess`

## Installation

```
git clone <this-repo-url>
cd roboPID-simulator
pip install -r requirements.txt
```

(A virtual environment is recommended but not required.)

## Usage

```
python app.py [--tau "5,5,5,5"] [--K 1.25] [--Td 8] [--Ts 1] [--ctype PID] [--port 8050] [--debug]
```

Then open `http://localhost:8050` in your browser.

| Flag | Default | Description |
|------|---------|-------------|
| `--tau` | `[5,5,5,5]` | Plant time constants, e.g. `"[5,5,5,5]"` or `"10"` |
| `--K` | `1.25` | Plant static gain |
| `--Td` | `8.0` | Plant dead time |
| `--Ts` | `1.0` | Sampling period |
| `--ctype` | `PID` | Controller type: `I`, `PI`, or `PID` |
| `--port` | `8050` | Server port |
| `--debug` | off | Enable Dash debug mode |

These flags only seed the initial values shown in the GUI — all parameters
can be changed live in the browser afterward.

## Configuration

`robopid.config` holds runtime simulation settings, overriding the built-in
defaults in `core/config.py` (whitespace-separated `key value` pairs, one
per line):

| Key | Default | Meaning |
|-----|---------|---------|
| `simtype` | `0` | `0` = linear simulation, `1` = discrete simulation with actuator saturation + disturbance |
| `minu` / `maxu` | `-1.0` / `1.0` | Actuator output limits (used when `simtype=1`) |
| `dist_tau` | `120.0` | Disturbance model time constant |
| `dist_std` | `0.05` | Disturbance standard deviation |
| `n_iter` | `100` | Number of iterations for the "Tune" search |
| `lipsch_const` | `0.0` | Lipschitz constant used by the tuning search |
| `tune_step` | `0.05` | Step size for each tuning iteration |

## Project Structure

```
roboPID-simulator/
├── app.py              # Dash entry point + CLI argument parsing
├── layout.py            # Page layout (plant/controller cards, plots)
├── callbacks.py          # Dash callbacks: Optimize, background "Tune", plot updates
├── robopid.config        # Runtime simulation settings (see Configuration)
├── requirements.txt
└── core/                  # Pure simulation/control logic, no UI dependencies
    ├── plant.py             # FOPTD plant transfer function + step response
    ├── pid.py                # Closed-loop PID simulation (linear + anti-windup)
    ├── optimizer.py          # SLSQP-based gain optimizer
    ├── tuning.py             # Iterative feature-driven auto-tuning
    ├── features.py           # Phase-plane / encirclement robustness features
    ├── signals.py            # Loop signal assembly, scaling, derivative features
    └── config.py             # Config file reader + disturbance model builder
```

## License

MIT — see [LICENSE](LICENSE).
