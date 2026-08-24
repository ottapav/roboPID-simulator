# roboPID-simulator

**RoboPID** is an interactive Dash web app implementing model-free PID
tuning by inspection of the step response, the method described in
[*Model-Free PID Tuning by Step-Response Inspection*](docs/JPC26_basic/main.tex)
(Pachner, Otta, Dostál, Havlena — submitted to the Journal of Process Control).

## Overview

The app models a process as a cascade of first-order lags with a static gain
and dead time:

```
P(s) = K * exp(-L*s) / prod(s*tau_i + 1)
```

and closes the loop around a parallel-form controller

```
C(s) = Kp + Ki/s + Kd*s
```

whose derivative channel is always rolled off as `Kd*s / (1 + (Kd/ν)*s)` with
`ν = 10` (`core.params.DERIV_FILTER_N`). The filter is not optional: an
unfiltered discrete derivative has Nyquist gain `2*Kd/Ts`, which the tuner
would happily drive into an undamped pole. It is also what puts the app on
the "filtered PID" row of the phase table below.

You set the plant parameters (`tau`, `K`, `L`) and pick a controller type
(I, PI, or PID), then click **Tune**. Each iteration steps the setpoint,
forms three phase portraits of the control error — Γ0, Γ1, Γ2, the paper's
"Pachner plots" — and counts how many times each winds around its settling
point (a settling-anchored window guard keeps the count independent of how
long the simulation runs; the **Guard** checkbox turns it off, pinning δ to 0
and falling back to the raw-window count of Definition 1). A maximum-likelihood
stability screen first backs off every gain when the record is growing rather
than decaying — Ki /2, Kp /4, Kd /8, deepening with the band's frequency;
otherwise the lowest band that loops too much (Γ0 → Ki, Γ1 → Kp, Γ2 → Kd) is
cut one notch `γ = 1/(1−β)` and every clean band below it is raised — the
paper's single triangular rule. Progress streams live as it runs, and the
resulting gain trajectory is plotted once it finishes. The **Iter** field
defaults to a controller-structure-appropriate iteration budget — fewer terms
to search means fewer iterations are needed: I uses 50, PI uses 100, and PID
uses 200 — and resets to that default whenever you switch controller type, but
you can type a custom count at any time.

RoboPID is built as an educational algorithm simulator for exploring PID
tuning behavior interactively, making the effect of each gain and each
tuning pass visible in real time.

## Scope: what the method can tune

The tuner searches for a stability boundary — it raises gains until a band
starts to ring, reads which band rang, and backs that band off. Two
requirements follow, and **Tune** checks both before it runs anything
(`core/admissibility.py`).

**The static gain must be positive.** The method assumes a self-regulating,
positive-acting plant, and the search box admits only positive Kp, Ki and Kd,
so a negative `K` would close the loop as positive feedback. Model an
inverse-acting process by flipping the sign of the measurement instead.

**The plant must supply enough phase lag for a boundary to exist at all.** The
test is structural: with plant relative degree `r` and controller asymptotic
phase `φ∞`, the asymptotic loop phase must sit strictly below −180°,

```
−(π/2)·r + φ∞ < −π        ⟺        r > 2 + (2/π)·φ∞
```

which for a delay-free plant is a minimum number of lags:

| controller | φ∞ | required r |
|---|---|---|
| I | −90° | 2 |
| PI | 0⁻ | 3 |
| PID (filtered — the one implemented here) | 0⁺ | 3 |
| PID, unfiltered derivative | +90° | 4 |

Any dead time `L > 0` satisfies every row, because a dead time makes the phase
unbounded below — so this only binds on a delay-free plant. Above −180° there
is no gain anywhere that makes the loop ring: every turn index stays at zero,
the rule reads "all quiet" forever, and no gain box or iteration budget changes
that. Tune refuses these plants and says why.

The test counts rather than measures, deliberately. Sweeping `arg P(jω)` and
looking for a crossing needs a tolerance exactly where the answer is marginal,
and would be fooled by a discretized model, whose ZOH lag manufactures a
crossing the continuous plant does not have. Counting is exact and independent
of the simulation grid.

The price is that it is **conservative in one band** — PI at `r=2`, filtered PID
at `r=2`, unfiltered PID at `r=3` — where the asymptote lands exactly on −180°
and whether the loop truly crosses is settled by `O(1/ω)` terms in `Ti`, `Td`,
the roll-off `ν` and the plant's own time constants. Those are not evaluated, so
the case is refused rather than guessed at. A plant refused there may well be
tunable — `τ=[1,2]`, `L=0` under PID is, and this tuner converges on it — but
the method does not claim it, and the refusal message says so rather than
pretending the plant has no boundary.

Everything else the app can express is admissible by construction — the plant
form above cannot represent an integrating, open-loop-unstable,
non-minimum-phase or resonant process.

The two blocking gates above are the only ones that refuse a click; a rejection
opens a modal ("Plant outside the tunable class") that states the finding, why
it holds, and what to change. Everything else `core/admissibility.py` reports
is advisory, because a run that hits it is still a valid run that has simply
terminated at a bound rather than at a turn-index limit. Before the run:

- **the target may sit outside the gain box** `[Kmin, Kmax]` — the tuner
  estimates where the boundary is from the plant's own ultimate point and says
  which gain cannot reach it, and by how much;
- **a frequency band may have collapsed** — Γ1 reads `[1/Ti, 1/Td]` and Γ2
  reads `[1/Td, ν/Td]`, and if either narrows the corresponding row of the rule
  stops attributing ringing to the gain it names. Under the shipped constants
  neither can fire, so these only appear if the ratios in
  `core/admissibility.py` or `DERIV_FILTER_N` have been edited.

And after it, read off the state the run actually finished in:

- **every multiplier pinned at the ceiling with nothing ringing** — the search
  ran out of box before any band rang. The boundary exists (the plant passed
  the phase gate); it is above `Kmax`;
- **one multiplier pinned at a bound with its own band still ringing** — the
  rule knew what to do and ran out of room to do it.

All four are reported under the cards, on their own line beneath the
input-validation warning, with widening the gain box being the first remedy
for any of them. When one fires, the gains left on the sliders are a box
boundary rather than a converged tuning, and the status line next to **TUNE**
says so.

## Features

- FOPTD plant modeling with an arbitrary number of cascaded time constants
- Iterative, encirclement-driven auto-tuning with live progress streaming
- Interactive Dash GUI: plant/controller inputs, Kp/Ki/Kd log-scale gain
  sliders (0.001-10, only the gains relevant to the selected controller
  type are shown), live step-response plot, the three Pachner-plot
  (Γ0/Γ1/Γ2) phase planes, and a gain-history plot of the last Tune run
- The paper's four battery plants P1–P4 as one-click presets in the Plant
  card; they fill τ/K/L and leave the controller alone
- Live-editable tuning constants next to Tune: the Γ0/Γ1/Γ2 limits, the gain
  box [Kmin, Kmax] (which also sets the slider range), the truncation radius
  ε, the step size, the settling guard δ and its on/off switch, and the
  iteration budget — plus **Reset controller** and **Reset tuner** buttons
- Admissibility gates that check the plant before a run and diagnose it
  afterwards: a blocking finding opens a modal, an advisory one appears
  beneath the cards and in the status line (see *Scope* above)
- Optional first-order filtered white noise on the plant output, toggled
  from the Plant card (off by default), with live σ and filter time
  constant controls
- Runtime-configurable simulation settings via `robopid.config`
- Every simulation runs on a fixed 500 samples, so the horizon `Tsim` is the
  only grid value on offer: it is proposed from the plant (10 × (Σ τ + L)),
  editable in the Plant card header, and resettable with ↺. The sampling period
  `Ts = Tsim / 499` follows from it and is shown read-only beside it

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`: `numpy`, `scipy`, `dash`,
  `dash-bootstrap-components`, `plotly`, `diskcache`, `psutil`, `multiprocess`,
  `gunicorn` (the last only needed for the hosted deployment)

## Installation

```
git clone <this-repo-url>
cd roboPID-simulator
pip install -r requirements.txt
```

(A virtual environment is recommended but not required.)

## Usage

```
python app.py [--tau "5,5,5,5"] [--K 1.25] [--L 8] [--ctype PID] [--port 8050] [--debug]
```

Then open `http://localhost:8050` in your browser.

| Flag | Default | Description |
|------|---------|-------------|
| `--tau` | `[5,5,5,5]` | Plant time constants, e.g. `"[5,5,5,5]"` or `"10"` |
| `--K` | `1.25` | Plant static gain |
| `--L` | `8.0` | Plant dead time |
| `--ctype` | `PID` | Controller type: `I`, `PI`, or `PID` |
| `--port` | `8050` | Server port |
| `--debug` | off | Enable Dash debug mode |

These flags only seed the initial values shown in the GUI — all parameters
can be changed live in the browser afterward. `$PORT`, if set, overrides
`--port`, which is how the hosted deployment picks its socket.

## Deployment

`app.py` exposes the Flask WSGI object as `server`, so any Gunicorn host can
serve it directly. The bundled `Procfile` is what Render runs:

```
web: gunicorn app:server --workers 2 --threads 4
```

Each worker imports the module separately and gets its own background-callback
cache directory under `.cache/` — `app._make_cache` explains why a shared one
would let a recycled worker wipe another's in-flight tuning job.

## Configuration

`robopid.config` holds runtime simulation settings, overriding the built-in
defaults in `core/config.py` (whitespace-separated `key value` pairs, one
per line):

| Key | Built-in default | Shipped in `robopid.config` | Meaning |
|-----|---------|---------|---------|
| `simtype` | `0` | `1` | `0` = linear simulation, `1` = discrete simulation with actuator saturation |
| `minu` / `maxu` | `-1.0` / `1.0` | `-10` / `+10` | Actuator output limits (used when `simtype=1`) |

The file is re-read whenever its mtime changes, so an edit takes effect on the
next slider drag without restarting the app. Deleting the file falls back to
the built-in defaults. (`robopid.config` may still carry a `lipsch_const` line
from an earlier revision. Nothing reads it — it is inert and can be deleted.)

The paper's dimensionless constants — the Γ0/Γ1/Γ2 loop limits, the
truncation radius ε, the settling-band guard δ, and the step size β (the
**Step γ** field; the applied notch is `γ = 1/(1−β)`) — are adjustable live
from the Controller card instead, next to the Tune button, rather than through
`robopid.config`. The same card also exposes the gain boundary [Kmin, Kmax]
that bounds the tuning search box and the slider range, so it can be widened
if a run stalls at its bound (Section 6 of the paper), and the **Guard**
checkbox that pins δ to 0 and reverts the counts to the unguarded raw window.

The Plant card exposes an "Output noise" checkbox (off by default) that
adds first-order filtered white noise to the plant's `y` output — a target
standard deviation σ (default 1%) and the noise filter's time constant
(default 0.1 × the average plant τ). Both fields are only editable while
the checkbox is checked.

## Project Structure

```
roboPID-simulator/
├── app.py                # Dash entry point + CLI argument parsing
├── layout.py             # Page layout (plant/controller cards, plots, modal)
├── callbacks.py          # Dash callbacks: background "Tune", plot updates
├── robopid.config        # Runtime simulation settings (see Configuration)
├── Procfile              # Gunicorn command for the hosted deployment
├── requirements.txt
├── assets/               # Client-side JS (slider tooltips, figure download)
├── docs/                 # Paper sources, posters and figure scripts
│   └── JPC26_basic/        # The Journal of Process Control manuscript
├── core/                 # Pure simulation/control logic, no UI dependencies
│   ├── params.py           # Shared constants + plant-parameter parsing
│   ├── plant.py            # FOPTD plant transfer function + step response
│   ├── pid.py              # Closed-loop PID simulation (linear + anti-windup)
│   ├── tuning.py           # Triangular tuning rule (paper Table 1)
│   ├── features.py         # Pachner plots Γ0/Γ1/Γ2 + encirclement counts
│   ├── signals.py          # Loop signals, settling guard, stability screen
│   ├── admissibility.py    # Pre-run gates + post-run diagnostics (see Scope)
│   └── config.py           # Config file reader + noise model builder
└── tests/                # Regression suite (see Tests)
    ├── goldens/            # Recorded reference outputs (.npz)
    └── generate_goldens.py # Regenerates them; run only on an intended change
```

## Tests

```
pip install pytest
python -m pytest tests/ -q
```

326 tests, about five seconds. They come in two kinds.

The **golden regressions** pin the simulation against recorded outputs for the
four battery plants P1–P4: step responses (`test_pid.py`), loop signals and the
stability screen (`test_signals.py`), encirclement counts and the feature
pipeline (`test_features.py`) and full tuning trajectories (`test_tuning.py`).
Any numeric change shows up as a diff there.

The **unit tests** pin behaviour that has no golden: the admissibility gates
and runtime detectors (`test_admissibility.py` — including that `LAGS_REQUIRED`
still matches the phase formula it was derived from), the GUI's
proposal/override arbitration for the simulation grid (`test_grid.py`, testing
the pure functions pulled out of `callbacks.py` so no browser is needed), and
parameter parsing and the shared defaults (`test_params.py`).

`tests/generate_goldens.py` rewrites the recorded references. Only run it when
a behaviour change is intended, and review the resulting diff — regenerating
goldens to make a failing test pass defeats the point of having them.

Note on tolerances: discrete outcomes (window indices, tuning trajectories,
which rule branch fired) are reproduced exactly. Floating step responses carry
a looser bound, because the closed-loop denominators are high-order and near
the unit circle — P4 is degree 12 — so any change to summation order moves the
last few digits without changing behaviour.

## Citation

If roboPID is useful in your research, please cite the paper it implements:

> D. Pachner, P. Otta, J. Dostál, V. Havlena, "Model-Free PID Tuning by
> Step-Response Inspection," submitted to Journal of Process Control.

```bibtex
@misc{pachner2026robopid,
  author = {Pachner, Daniel and Otta, Pavel and Dostál, Jiří and Havlena, Vladimír},
  title  = {Model-Free PID Tuning by Step-Response Inspection},
  note   = {Submitted to Journal of Process Control},
  year   = {2026},
  url    = {https://github.com/ottapav/roboPID-simulator}
}
```

## License

MIT — see [LICENSE](LICENSE).
