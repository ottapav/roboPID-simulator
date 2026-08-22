"""
Measured terminal orbit of P1 under the REGULARIZED index.

Replaces the idealized cube figure. The orbit really has period 8 with move
counts (1,4,2,1), as Corollary 2 requires, but its vertices are not the
corners of a cube: three d0 moves occur consecutively, and the eight points
fill a 3 x 2 x 1 box in units of h.

Coordinates are in band order (log K_i, log K_p, log K_d).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import regindex
regindex.patch()
from duty_cycle import run, ROW2MOVE, PLANTS   # noqa: E402

COLORS = {"d0": "#c0392b", "d1": "#e08e0b", "d2": "#2e8b57", "e": "#1b6ca8"}
LABEL = {"d0": r"$d_0$", "d1": r"$d_1$", "d2": r"$d_2$", "e": r"$e$"}
MOVE = {"e": np.array([1., 1., 1.]), "d0": np.array([-1., 0., 0.]),
        "d1": np.array([1., -1., 0.]), "d2": np.array([1., 1., -1.])}


def terminal_orbit(plant="P1", beta=0.10, n_iter=600, pmax=48):
    tau, K, L = PLANTS[plant]
    rows, _ = run(tau, K, L, beta, n_iter=n_iter)
    seq = [ROW2MOVE.get(r, "X") for r in rows if r in ROW2MOVE]
    for p in range(1, pmax + 1):
        n_ok = 0
        for i in range(len(seq) - p - 1, -1, -1):
            if seq[i] == seq[i + p]:
                n_ok += 1
            else:
                break
        if n_ok >= 2 * p:
            return seq[-p:]
    raise RuntimeError("no periodic terminal orbit found")


def main():
    unit = terminal_orbit()
    x = np.zeros(3)
    pts = [x.copy()]
    for m in unit:
        x = x + MOVE[m]
        pts.append(x.copy())
    pts = np.array(pts)
    verts = pts[:-1]
    lo, hi = verts.min(0), verts.max(0)
    c = verts.mean(0)

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")

    # bounding box, drawn faintly
    for a in (0, 1):
        for b in (0, 1):
            ax.plot([lo[0], hi[0]], [[lo[1], hi[1]][a]] * 2,
                    [[lo[2], hi[2]][b]] * 2, color="0.86", lw=0.6)
            ax.plot([[lo[0], hi[0]][a]] * 2, [lo[1], hi[1]],
                    [[lo[2], hi[2]][b]] * 2, color="0.86", lw=0.6)
            ax.plot([[lo[0], hi[0]][a]] * 2, [[lo[1], hi[1]][b]] * 2,
                    [lo[2], hi[2]], color="0.86", lw=0.6)

    seen = set()
    for k, m in enumerate(unit):
        sg = pts[k:k + 2]
        ax.plot(sg[:, 0], sg[:, 1], sg[:, 2], color=COLORS[m], lw=2.4,
                solid_capstyle="round",
                label=LABEL[m] if m not in seen else None)
        seen.add(m)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=26, color="k",
               depthshade=False)
    for i, v in enumerate(verts):
        ax.text(v[0] + .08, v[1] + .05, v[2] + .06, str(i), fontsize=7.5,
                color="0.4")

    arm = 0.28
    for d in range(3):
        a = np.zeros(3); a[d] = arm
        ax.plot(*[[c[i] - a[i], c[i] + a[i]] for i in range(3)],
                color="k", lw=1.5)

    ax.set_xlabel(r"$\log K_i$", fontsize=9, labelpad=-6)
    ax.set_ylabel(r"$\log K_p$", fontsize=9, labelpad=-6)
    ax.set_zlabel(r"$\log K_d$", fontsize=9, labelpad=-12)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticks([])
    ax.set_box_aspect((3, 2, 1.4))
    ax.view_init(elev=22, azim=-66)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, handlelength=1.3,
              labelspacing=0.3, bbox_to_anchor=(-0.02, 0.95))
    ax.set_title("P1 terminal orbit, regularized index  ($\\beta = 0.10$)",
                 fontsize=10, y=0.99)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig_orbit_P1.png", dpi=250)
    fig.savefig("/mnt/user-data/outputs/fig_orbit_P1.pdf")

    print("unit    :", " ".join(unit))
    print("vertices:\n", verts)
    print("half-span/h:", (hi - lo) / 2)


if __name__ == "__main__":
    main()
