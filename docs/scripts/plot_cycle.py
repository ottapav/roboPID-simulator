"""
Figure: the SPIN limit cycle in log-gain space.

The period-8 ruler orbit visits exactly the eight corners of a cube of
side h = log(1+beta), centred on the triple point theta*, in binary
counting order (log Ki = least significant bit (band order K_i, K_p, K_d)).

NOTE: the ordering drawn is the ruler/binary-carry sequence implied by
the measured duty cycle (1/2, 1/4, 1/8, 1/8).  Verify against the
recorded move sequence of a real terminal orbit before publishing.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

MOVES = {
    "e":  np.array([1.0, 1.0, 1.0]),
    "d0": np.array([-1.0, 0.0, 0.0]),
    "d1": np.array([1.0, -1.0, 0.0]),
    "d2": np.array([1.0, 1.0, -1.0]),
}
COLORS = {"d0": "#c0392b", "d1": "#e08e0b", "d2": "#2e8b57", "e": "#1b6ca8"}
LABELS = {"d0": r"$d_0$", "d1": r"$d_1$", "d2": r"$d_2$", "e": r"$e$"}
SEQ = ["d0", "d1", "d0", "d2", "d0", "d1", "d0", "e"]

VIEW = dict(elev=17, azim=-58)


def orbit(h=1.0):
    p = np.zeros(3)
    pts = [p.copy()]
    for m in SEQ:
        p = p + h * MOVES[m]
        pts.append(p.copy())
    return np.array(pts)


def cube_edges(h=1.0):
    corners = np.array([[a, b, c] for a in (0.0, -h)
                        for b in (0.0, -h) for c in (0.0, -h)])
    return [[corners[i], corners[j]]
            for i in range(8) for j in range(i + 1, 8)
            if np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-9) == 1]


def main():
    h = 1.0
    pts = orbit(h)
    centre = pts[:-1].mean(axis=0)

    fig = plt.figure(figsize=(4.4, 3.9))
    ax = fig.add_subplot(111, projection="3d")

    # cube mesh plotted FIRST using individual plot commands
    for edge in cube_edges(h):
        seg = np.array(edge)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="0.82", lw=0.6)

    # the eight moves of one period, plotted SECOND with bolder lines
    seen = set()
    for k, m in enumerate(SEQ):
        seg = pts[k:k + 2]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=COLORS[m],
                lw=2.4, solid_capstyle="round",
                label=LABELS[m] if m not in seen else None)
        seen.add(m)

    # vertices, labelled by odometer reading
    ax.scatter(pts[:-1, 0], pts[:-1, 1], pts[:-1, 2], s=16, color="k",
               depthshade=False, zorder=4)
    for v in pts[:-1]:
        b = [int(round(-x / h)) for x in v]
        ax.text(v[0] + .05 * h, v[1] + .05 * h, v[2] + .07 * h,
                str(b[0] + 2 * b[1] + 4 * b[2]), fontsize=7, color="0.45", zorder=4)

    # triple point at the cube centre: 3D cross (three orthogonal segments)
    cross_arm = 0.18 * h
    ax.plot([centre[0] - cross_arm, centre[0] + cross_arm],
            [centre[1], centre[1]], [centre[2], centre[2]],
            color="k", lw=1.8)
    ax.plot([centre[0], centre[0]],
            [centre[1] - cross_arm, centre[1] + cross_arm],
            [centre[2], centre[2]],
            color="k", lw=1.8)
    ax.plot([centre[0], centre[0]],
            [centre[1], centre[1]],
            [centre[2] - cross_arm, centre[2] + cross_arm],
            color="k", lw=1.8)
    ax.text(centre[0] + .10 * h, centre[1] - .02 * h, centre[2] - .22 * h,
            r"$\theta^*$", fontsize=9, color="k")

    m = 0.16 * h
    ax.set_xlim(-h - m, m); ax.set_ylim(-h - m, m); ax.set_zlim(-h - m, m)
    ax.set_xlabel(r"$\log K_i$", fontsize=9, labelpad=-8)
    ax.set_ylabel(r"$\log K_p$", fontsize=9, labelpad=-8)
    ax.set_zlabel(r"$\log K_d$", fontsize=9, labelpad=-8)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(**VIEW)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

    ax.legend(loc="upper right", bbox_to_anchor=(1.04, 0.96), fontsize=8.5,
              frameon=False, handlelength=1.3, labelspacing=0.35)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle.pdf", bbox_inches="tight")
    fig.savefig("/mnt/user-data/outputs/fig_limit_cycle.png", dpi=260,
                bbox_inches="tight")


if __name__ == "__main__":
    main()
