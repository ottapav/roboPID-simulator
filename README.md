# roboPID-simulator

**RoboPID** is an interactive Dash web app implementing model-free PID
tuning via encirclements of the step response, the method described in
[*Model-Free PID Tuning via Encirclements of the Step Response*](RoboPID_JPC_paper/main.tex)
(Pachner, Otta, Dostál).

## Overview

The app models a process as a cascade of first-order lags with a static gain
and dead time:

```
P(s) = K * exp(-L*s) / prod(s*tau_i + 1)
```

You set the plant parameters (`tau`, `K`, `L`) and pick a controller type
(I, PI, or PID), then click **Tune**. Each iteration steps the setpoint,
forms three phase portraits of the control error — Γ0, Γ1, Γ2, the paper's
"Pachner plots" — and counts how many times each winds around its settling
point (a settling-anchored window guard keeps the count independent of how
long the simulation runs). A maximum-likelihood stability screen first backs
off every gain when the record is growing rather than decaying; otherwise
the lowest band that loops too much (Γ0 → Ki, Γ1 → Kp, Γ2 → Kd) is cut one
notch and every clean band below it is raised — the paper's single
triangular rule. Progress streams live as it runs, and the resulting gain
trajectory is plotted once it finishes. The **Iter** field defaults to a
controller-structure-appropriate iteration budget — fewer terms to search
means fewer iterations are needed: I uses 50, PI uses 100, and PID uses
200 — and resets to that default whenever you switch controller type, but
you can type a custom count at any time.

RoboPID is built as an educational algorithm simulator for exploring PID
tuning behavior interactively, making the effect of each gain and each
tuning pass visible in real time.

## Features

- FOPTD plant modeling with an arbitrary number of cascaded time constants
- Iterative, encirclement-driven auto-tuning with live progress streaming
- Interactive Dash GUI: plant/controller inputs, Kp/Ki/Kd log-scale gain
  sliders (0.01-10, only the gains relevant to the selected controller
  type are shown), live step-response plot, the three Pachner-plot
  (Γ0/Γ1/Γ2) phase planes, and a gain-history plot of the last Tune run
- Optional first-order filtered white noise on the plant output, toggled
  from the Plant card (off by default), with live σ and filter time
  constant controls
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
python app.py [--tau "5,5,5,5"] [--K 1.25] [--L 8] [--Ts 1] [--ctype PID] [--port 8050] [--debug]
```

Then open `http://localhost:8050` in your browser.

| Flag | Default | Description |
|------|---------|-------------|
| `--tau` | `[5,5,5,5]` | Plant time constants, e.g. `"[5,5,5,5]"` or `"10"` |
| `--K` | `1.25` | Plant static gain |
| `--L` | `8.0` | Plant dead time |
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
| `simtype` | `0` | `0` = linear simulation, `1` = discrete simulation with actuator saturation |
| `minu` / `maxu` | `-1.0` / `1.0` | Actuator output limits (used when `simtype=1`) |
| `lipsch_const` | `0.0` | Lipschitz constant used by the tuning search |

The paper's dimensionless constants — the Γ0/Γ1/Γ2 loop limits, the
truncation radius ε, the settling-band guard δ, and the step size β — are
adjustable live from the Controller card instead, next to the Tune button,
rather than through `robopid.config`. The same card also exposes the gain
boundary [Kmin, Kmax] that bounds the tuning search box, so it can be
widened if a run stalls at its bound (Section 6 of the paper).

The Plant card exposes an "Output noise" checkbox (off by default) that
adds first-order filtered white noise to the plant's `y` output — a target
standard deviation σ (default 1%) and the noise filter's time constant
(default 0.1 × the average plant τ). Both fields are only editable while
the checkbox is checked.

## Project Structure

```
roboPID-simulator/
├── app.py              # Dash entry point + CLI argument parsing
├── layout.py            # Page layout (plant/controller cards, plots)
├── callbacks.py          # Dash callbacks: background "Tune", plot updates
├── robopid.config        # Runtime simulation settings (see Configuration)
├── requirements.txt
└── core/                  # Pure simulation/control logic, no UI dependencies
    ├── plant.py             # FOPTD plant transfer function + step response
    ├── pid.py                # Closed-loop PID simulation (linear + anti-windup)
    ├── tuning.py             # Triangular tuning rule (paper Table 1)
    ├── features.py           # Pachner plots Γ0/Γ1/Γ2 + encirclement counts
    ├── signals.py            # Loop signals, settling guard, stability screen
    └── config.py             # Config file reader + noise model builder
```

## License

MIT — see [LICENSE](LICENSE).
