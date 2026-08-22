"""
Limit-cycle figures for reduced-order SPIN tuning (PI, I-only), built as
the natural n-gain generalization of the PID (n=3) triangular rule.

EXTRAPOLATION NOTICE: the PID move set {e, d0, d1, d2} is taken from the
project handoff.  The PI (n=2) and I-only (n=1) move sets below are *my*
extension of that same "staircase" pattern to fewer gains -- not something
verified against the actual SPIN code.  Check against a real 1- or
2-parameter run before using in the paper; if the real rule differs
(e.g. different thresholds or an extra move), only the geometry changes,
not the balance-weight argument.

General n-gain rule:
    e     = (+1, +1, ..., +1)                    (expand)
    d_k   = (+1, ..., +1, -1, 0, ..., 0)          (cut band k, k=0..n-1)
            (first k entries +1, entry k is -1, rest 0)

Balance weights (solved the same way as Proposition 1):
    n=1 (I only):      (w_e, w_d0)                 = (1/2, 1/2)
    n=2 (PI):          (w_e, w_d0, w_d1)            = (1/4, 1/2, 1/4)
    n=3 (PID):         (w_e, w_d0, w_d1, w_d2)      = (1/8, 1/2, 1/4, 1/8)

Each case traces the corners of the n-cube {0,-h}^n in binary-counting
order: a segment (n=1), a square (n=2), a cube (n=3, see plot_cycle.py).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"d0": "#c0392b", "d1": "#e08e0b", "e": "#1b6ca8"}
LABELS = {"d0": r"$d_0$", "d1": r"$d_1$", "e": r"$e$"}


# ---------------------------------------------------------------- PI (n=2)
def orbit_2d(h=1.0):
    moves = {"e": np.array([1.0, 1.0]),
             "d0": np.array([-1.0, 0.0]),
             "d1": np.array([1.0, -1.0])}
    seq = ["d0", "d1", "d0", "e"]  # period 4, weights (1/2, 1/4, 1/4)
    p = np.zeros(2)
    pts = [p.copy()]
    for m in seq:
        p = p + h * moves[m]
        pts.append(p.copy())
    return np.array(pts), seq


def plot_pi(h=1.0):
    pts, seq = orbit_2d(h)
    centre = pts[:-1].mean(axis=0)

    fig, ax = plt.subplots(figsize=(3.0, 3.0))

    # square mesh first
    sq = np.array([[0, 0], [-h, 0], [-h, -h], [0, -h], [0, 0]])
    ax.plot(sq[:, 0], sq[:, 1], color="0.82", lw=0.6)

    # the four moves, bolder, on top
    seen = set()
    for k, m in enumerate(seq):
        seg = pts[k:k + 2]
        ax.plot(seg[:, 0], seg[:, 1], color=COLORS[m], lw=2.4,
                solid_capstyle="round",
                label=LABELS[m] if m not in seen else None)
        seen.add(m)

    ax.scatter(pts[:-1, 0], pts[:-1, 1], s=26, color="k", zorder=4)
    for v in pts[:-1]:
        b = [int(round(-x / h)) for x in v]
        n = b[0] + 2 * b[1]
        ax.annotate(str(n), v, xytext=(6, 5), textcoords="offset points",
                    fontsize=8, color="0.45")

    # small black cross at the centre (same convention as the cube figure)
    arm = 0.14 * h
    ax.plot([centre[0] - arm, centre[0] + arm], [centre[1], centre[1]],
            color="k", lw=1.6)
    ax.plot([centre[0], centre[0]], [centre[1] - arm, centre[1] + arm],
            color="k", lw=1.6)
    ax.annotate(r"$\theta^*$", centre, xytext=(8, -12),
                textcoords="offset points", fontsize=10, color="k")

    m = 0.22 * h
    ax.set_xlim(-h - m, m); ax.set_ylim(-h - m, m)
    ax.set_xlabel(r"$\log K_i$", fontsize=10)
    ax.set_ylabel(r"$\log K_p$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False,
              handlelength=1.3, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle_PI.pdf", bbox_inches="tight")
    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle_PI.png", dpi=260,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- I-only (n=1)
def plot_i(h=1.0):
    # sequence: d0, e, d0, e, ... period 2, weights (1/2, 1/2)
    p0, p1 = 0.0, -h
    centre = (p0 + p1) / 2

    fig, ax = plt.subplots(figsize=(3.4, 1.6))

    # base line (mesh) first
    ax.plot([p1 - 0.25 * h, p0 + 0.25 * h], [0, 0], color="0.82", lw=0.6)

    # the two moves, bolder, on top, drawn as slightly separated arcs
    # so both directions of the same segment are visible
    ax.plot([p0, p1], [0.05, 0.05], color=COLORS["d0"], lw=2.4,
            solid_capstyle="round", label=LABELS["d0"])
    ax.plot([p1, p0], [-0.05, -0.05], color=COLORS["e"], lw=2.4,
            solid_capstyle="round", label=LABELS["e"])

    ax.scatter([p0, p1], [0, 0], s=26, color="k", zorder=4)
    ax.annotate("0", (p0, 0), xytext=(0, 10), textcoords="offset points",
                fontsize=8, color="0.45", ha="center")
    ax.annotate("1", (p1, 0), xytext=(0, 10), textcoords="offset points",
                fontsize=8, color="0.45", ha="center")

    arm = 0.09
    ax.plot([centre - h * arm, centre + h * arm], [0, 0], color="k", lw=1.6)
    ax.annotate(r"$\theta^*$", (centre, 0), xytext=(0, -20),
                textcoords="offset points", fontsize=10, color="k",
                ha="center")

    ax.set_xlim(p1 - 0.3 * h, p0 + 0.3 * h)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel(r"$\log K_i$", fontsize=10)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2,
              fontsize=9, frameon=False, handlelength=1.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle_I.pdf", bbox_inches="tight")
    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle_I.png", dpi=260,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_pi()
    plot_i()
    print("done")
