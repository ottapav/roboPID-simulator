"""
Three-panel limit-cycle hierarchy: I-only (segment), PI (square), PID (cube).

Coordinates are in BAND ORDER, following paper 1:
    K^(0) = K_i,  K^(1) = K_p,  K^(2) = K_d
so log K_i is the least significant bit of the odometer.

Panel captions are placed with fig.text at a single shared baseline, so that
they line up across panels whose axes have different heights (the I-only
panel is short, the PI panel square, the PID panel a 3D box).

EXTRAPOLATION NOTICE: the PID move set is paper 1's Table 2. The PI (n=2)
and I-only (n=1) move sets are the natural staircase reduction and have not
been verified against the implementation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

COLORS = {"d0": "#c0392b", "d1": "#e08e0b", "d2": "#2e8b57", "e": "#1b6ca8"}
CAPTION_Y = 0.035          # shared baseline for all three captions
BOTTOM = 0.17              # space reserved beneath the axes
H = 1.0


def cross(ax, c, arm, lw=1.6, d3=False):
    """Small black cross marking theta*."""
    if d3:
        ax.plot([c[0] - arm, c[0] + arm], [c[1], c[1]], [c[2], c[2]], color="k", lw=lw)
        ax.plot([c[0], c[0]], [c[1] - arm, c[1] + arm], [c[2], c[2]], color="k", lw=lw)
        ax.plot([c[0], c[0]], [c[1], c[1]], [c[2] - arm, c[2] + arm], color="k", lw=lw)
    else:
        ax.plot([c[0] - arm, c[0] + arm], [c[1], c[1]], color="k", lw=lw)
        ax.plot([c[0], c[0]], [c[1] - arm, c[1] + arm], color="k", lw=lw)


def panel_i(ax):
    p0, p1 = 0.0, -H
    c = (p0 + p1) / 2
    ax.plot([p1 - .25 * H, p0 + .25 * H], [0, 0], color="0.82", lw=0.6)
    ax.plot([p0, p1], [.05, .05], color=COLORS["d0"], lw=2.4,
            solid_capstyle="round", label=r"$d_0$")
    ax.plot([p1, p0], [-.05, -.05], color=COLORS["e"], lw=2.4,
            solid_capstyle="round", label=r"$e$")
    ax.scatter([p0, p1], [0, 0], s=26, color="k", zorder=4)
    ax.plot([c - .09 * H, c + .09 * H], [0, 0], color="k", lw=1.6)
    ax.annotate(r"$\theta^*$", (c, 0), xytext=(0, 9),
                textcoords="offset points", fontsize=10, ha="center")
    ax.set_xlim(p1 - .3 * H, p0 + .3 * H)
    ax.set_ylim(-.9, .9)
    ax.set_xlabel(r"$\log K_i$", fontsize=10, labelpad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2,
              fontsize=8.5, frameon=False, handlelength=1.2, columnspacing=1.2)


def panel_pi(ax):
    M = {"e": np.array([1., 1.]), "d0": np.array([-1., 0.]),
         "d1": np.array([1., -1.])}
    seq = ["d0", "d1", "d0", "e"]
    p = np.zeros(2); pts = [p.copy()]
    for m in seq:
        p = p + H * M[m]; pts.append(p.copy())
    pts = np.array(pts); c = pts[:-1].mean(0)

    sq = np.array([[0, 0], [-H, 0], [-H, -H], [0, -H], [0, 0]])
    ax.plot(sq[:, 0], sq[:, 1], color="0.82", lw=0.6)
    seen = set()
    for k, m in enumerate(seq):
        sg = pts[k:k + 2]
        lab = (r"$e$" if m == "e" else r"$d_%s$" % m[1]) if m not in seen else None
        ax.plot(sg[:, 0], sg[:, 1], color=COLORS[m], lw=2.4,
                solid_capstyle="round", label=lab)
        seen.add(m)
    ax.scatter(pts[:-1, 0], pts[:-1, 1], s=26, color="k", zorder=4)
    cross(ax, c, .14 * H)
    ax.annotate(r"$\theta^*$", c, xytext=(8, -13),
                textcoords="offset points", fontsize=10)
    m_ = .22 * H
    ax.set_xlim(-H - m_, m_); ax.set_ylim(-H - m_, m_)
    ax.set_xlabel(r"$\log K_i$", fontsize=10, labelpad=2)
    ax.set_ylabel(r"$\log K_p$", fontsize=10, labelpad=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False,
              handlelength=1.2, labelspacing=0.3)


def panel_pid(ax):
    M = {"e": np.array([1., 1., 1.]), "d0": np.array([-1., 0., 0.]),
         "d1": np.array([1., -1., 0.]), "d2": np.array([1., 1., -1.])}
    seq = ["d0", "d1", "d0", "d2", "d0", "d1", "d0", "e"]
    p = np.zeros(3); pts = [p.copy()]
    for m in seq:
        p = p + H * M[m]; pts.append(p.copy())
    pts = np.array(pts); c = pts[:-1].mean(0)

    cor = np.array([[a, b, d] for a in (0., -H) for b in (0., -H) for d in (0., -H)])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(np.abs(cor[i] - cor[j]) > 1e-9) == 1:
                sg = np.array([cor[i], cor[j]])
                ax.plot(sg[:, 0], sg[:, 1], sg[:, 2], color="0.82", lw=0.6)
    for k, m in enumerate(seq):
        sg = pts[k:k + 2]
        ax.plot(sg[:, 0], sg[:, 1], sg[:, 2], color=COLORS[m], lw=2.2,
                solid_capstyle="round")
    ax.scatter(pts[:-1, 0], pts[:-1, 1], pts[:-1, 2], s=14, color="k")
    cross(ax, c, .18 * H, lw=1.4, d3=True)
    m_ = .16 * H
    ax.set_xlim(-H - m_, m_); ax.set_ylim(-H - m_, m_); ax.set_zlim(-H - m_, m_)
    ax.set_xlabel(r"$\log K_i$", fontsize=8, labelpad=-8)
    ax.set_ylabel(r"$\log K_p$", fontsize=8, labelpad=-8)
    ax.set_zlabel(r"$\log K_d$", fontsize=8, labelpad=-12)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=17, azim=-58)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)


def main():
    fig = plt.figure(figsize=(9.0, 3.3))
    axA = fig.add_subplot(1, 3, 1)
    axB = fig.add_subplot(1, 3, 2)
    axC = fig.add_subplot(1, 3, 3, projection="3d")

    panel_i(axA)
    panel_pi(axB)
    panel_pid(axC)

    fig.subplots_adjust(left=0.03, right=0.96, bottom=BOTTOM, top=0.96,
                        wspace=0.26)

    # captions on one shared baseline, centred under each panel
    for ax, text in ((axA, "(a) I only  —  period 2"),
                     (axB, "(b) PI  —  period 4"),
                     (axC, "(c) PID  —  period 8")):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, CAPTION_Y, text,
                 ha="center", va="bottom", fontsize=10)

    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle_hierarchy.pdf")
    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle_hierarchy.png", dpi=260)


if __name__ == "__main__":
    main()
