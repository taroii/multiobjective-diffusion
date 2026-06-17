"""Publication-quality figures for the multi-objective experiments.

All of these are computed analytically from the moments (Gram matrices Sigma_i,
vectors b_i) -- no diffusion sampling -- so they render instantly.  They share a
clean, conference-ready style (Computer-Modern math, vector PDF).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from socp import g_values, max_g

# A small, colour-blind-safe palette.
C_MG = "#0072B2"     # MG-AMOO (blue)
C_PA = "#D55E00"     # PAMOO (vermillion)
C_STAR = "#CC2222"   # w* / alpha*
C_UNI = "#444444"    # uniform
C_BOUND = "#117733"  # bound / V-shape (green)


def pubstyle():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 12,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.18,
    })


# --- ternary helpers --------------------------------------------------------
# Equilateral-triangle corners for objectives (0, 1, 2).
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def bary_xy(W):
    """Barycentric weights (..., 3) -> 2D coords of an equilateral triangle."""
    W = np.atleast_2d(np.asarray(W, float))
    return W @ _V


def _simplex_mesh(res=120):
    """Dense simplex grid + Delaunay triangulation in 2D for surfaces/contours."""
    pts = []
    for i in range(res + 1):
        for j in range(res + 1 - i):
            k = res - i - j
            pts.append([i / res, j / res, k / res])
    W = np.array(pts)
    XY = bary_xy(W)
    tri = mtri.Triangulation(XY[:, 0], XY[:, 1])
    return W, XY, tri


# --- Figure 1: 3D bound surface over the simplex ----------------------------
def fig_bound_surface_3d(Sigmas, bs, kappa, w_star, names, path, res=110):
    """The convex minimax max_i g_i(w) as a 3D surface over the 2-simplex."""
    W, XY, tri = _simplex_mesh(res)
    Z = np.array([max_g(w, Sigmas, bs, kappa) for w in W])
    zmin, zmax = Z.min(), Z.max()
    floor = zmin - 0.32 * (zmax - zmin)

    fig = plt.figure(figsize=(7.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(XY[:, 0], XY[:, 1], Z, triangles=tri.triangles,
                    cmap="viridis", linewidth=0.0, antialiased=True, alpha=0.95)
    # filled contour projected on the floor for depth
    ax.tricontourf(mtri.Triangulation(XY[:, 0], XY[:, 1], tri.triangles), Z,
                   levels=18, zdir="z", offset=floor, cmap="viridis", alpha=0.65)
    # optimum: stem from floor to the surface + bright star
    xs, ys = bary_xy(w_star)[0]
    zs = max_g(w_star, Sigmas, bs, kappa)
    ax.plot([xs, xs], [ys, ys], [floor, zs], color=C_STAR, lw=1.8, ls="--", zorder=8)
    ax.scatter([xs], [ys], [zs], color=C_STAR, marker="*", s=320, depthshade=False,
               zorder=12, edgecolors="k", linewidths=0.8, label=r"$w^\star$ (SOCP optimum)")
    ax.scatter([xs], [ys], [floor], color=C_STAR, marker="*", s=120, depthshade=False,
               zorder=12, edgecolors="k", linewidths=0.5, alpha=0.8)
    # vertex labels (single objectives)
    for i, nm in enumerate(names):
        vx, vy = _V[i]
        ax.text(vx, vy, max_g(np.eye(len(names))[i], Sigmas, bs, kappa) + 0.05 * (zmax - zmin),
                nm, fontsize=12, ha="center", va="bottom", fontweight="bold")
    ax.set_zlim(floor, zmax * 1.02)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_zlabel(r"$\max_i\, g_i(w)$", labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 0.72), zoom=0.92)
    ax.set_title(r"Convex minimax landscape on the simplex $\Delta^3$", pad=12, y=0.98)
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.93))
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --- Figure 2: simplex contour + algorithm trajectories ---------------------
def fig_simplex_trajectories(Sigmas, bs, kappa, w_star, traj_mg, traj_pa, names, path, res=160):
    W, XY, tri = _simplex_mesh(res)
    Z = np.array([max_g(w, Sigmas, bs, kappa) for w in W])

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    tcf = ax.tricontourf(XY[:, 0], XY[:, 1], Z, levels=24, cmap="viridis", alpha=0.95)
    ax.tricontour(XY[:, 0], XY[:, 1], Z, levels=12, colors="white", linewidths=0.35, alpha=0.5)
    cb = fig.colorbar(tcf, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(r"$\max_i\, g_i(w)$")
    # triangle border
    tri_xy = np.vstack([_V, _V[0]])
    ax.plot(tri_xy[:, 0], tri_xy[:, 1], color="k", lw=1.1)
    # plot the running-AVERAGE iterate (what provably converges to w*), so the
    # path is a clean curve from the uniform start into the optimum.
    def draw(traj, c, lab):
        avg = np.cumsum(traj, axis=0) / np.arange(1, len(traj) + 1)[:, None]
        P = bary_xy(avg)
        ax.plot(P[:, 0], P[:, 1], color=c, lw=2.2, alpha=0.95, label=lab, zorder=4)
        ax.scatter(P[0, 0], P[0, 1], color=c, s=34, zorder=5, edgecolors="k", linewidths=0.5)
    draw(traj_mg, C_MG, r"MG-AMOO $\bar w_K$")
    draw(traj_pa, C_PA, r"PAMOO $\bar w_K$")
    ax.scatter([], [], color="none", label="(uniform start $\\to w^\\star$)")
    sx, sy = bary_xy(w_star)[0]
    ax.scatter([sx], [sy], marker="*", s=320, color=C_STAR, edgecolors="k",
               linewidths=1.0, zorder=6, label=r"$w^\star$")
    for i, nm in enumerate(names):
        off = (0, 8) if i == 2 else (-10 if i == 0 else 10, -16)
        ax.annotate(nm, _V[i], textcoords="offset points", xytext=off,
                    ha="center", fontsize=12, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    ax.margins(0.12)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.0))
    ax.set_title("Both algorithms descend the simplex to $w^\\star$", pad=14)
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --- Figure 2b: realized worst-case TV over the simplex ---------------------
def fig_realized_simplex(W_grid, realized, w_star, names, path, w_hat=None):
    """Scatter heat-map of the REALIZED worst-case TV over a simplex grid, with
    w* (bound-optimal) and the realized grid-minimizer marked."""
    gx, gy = bary_xy(W_grid).T
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sc = ax.scatter(gx, gy, c=realized, s=320, cmap="viridis", edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("realized worst-case TV")
    tri_xy = np.vstack([_V, _V[0]])
    ax.plot(tri_xy[:, 0], tri_xy[:, 1], color="k", lw=1.0, alpha=0.5)
    sx, sy = bary_xy(w_star)[0]
    ax.scatter([sx], [sy], marker="*", s=340, color=C_STAR, edgecolors="k",
               linewidths=1.0, zorder=6, label=r"$w^\star$ (bound-optimal)")
    if w_hat is not None:
        hx, hy = bary_xy(w_hat)[0]
        ax.scatter([hx], [hy], marker="o", s=150, color="#EE99FF", edgecolors="k",
                   linewidths=1.0, zorder=6, label=r"$\hat w$ (realized min)")
    ux, uy = bary_xy(np.ones(3) / 3)[0]
    ax.scatter([ux], [uy], marker="P", s=170, color="white", edgecolors="k",
               linewidths=1.0, zorder=6, label="uniform")
    for i, nm in enumerate(names):
        off = (0, 8) if i == 2 else (-10 if i == 0 else 10, -16)
        ax.annotate(nm, _V[i], textcoords="offset points", xytext=off,
                    ha="center", fontsize=12, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.12)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.0))
    ax.set_title("Realized worst-case TV over the simplex", pad=14)
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --- Figure 3: convergence ---------------------------------------------------
def fig_convergence(hist_mg, hist_pa, val, path):
    K = len(hist_mg)
    Ks = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(Ks, np.maximum(hist_mg - val, 1e-7), color=C_MG, label="MG-AMOO")
    ax.loglog(Ks, np.maximum(hist_pa - val, 1e-7), color=C_PA, label="PAMOO")
    ref = (max(hist_mg[0], hist_pa[0]) - val) / np.sqrt(Ks)
    ax.loglog(Ks, ref, color="k", ls="--", lw=1.2, alpha=0.7, label=r"$\mathcal{O}(1/\sqrt{K})$")
    ax.set_xlabel(r"iteration $K$")
    ax.set_ylabel(r"optimality gap $\Gamma_K=\max_i g_i(\bar w_K)-\max_i g_i(w^\star)$")
    ax.set_title("Convergence to the SOCP optimum at the predicted rate")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --- Figure 4: m=2 bound money figure ---------------------------------------
def fig_m2_bound(sigma12, sigma21, b12, b21, kappa, path, realized=None):
    """The constant-free two-objective bound V-shape vs alpha.

    g_1(a)=(1-a)(sigma12+kappa b12), g_2(a)=a(sigma21+kappa b21); their max is the
    bound, minimized at alpha*.  Optionally overlay a realized worst-case-TV
    curve (dict with 'alphas','worst') on a twin axis to show matched argmins.
    """
    K1 = sigma12 + kappa * b12
    K2 = sigma21 + kappa * b21
    astar = K1 / (K1 + K2)
    a = np.linspace(0, 1, 400)
    g1 = (1 - a) * K1
    g2 = a * K2
    gmax = np.maximum(g1, g2)
    ystar = np.interp(astar, a, gmax)
    yu = np.interp(0.5, a, gmax)

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.plot(a, g1, ls=":", lw=1.3, color="#999999")
    ax.plot(a, g2, ls=":", lw=1.3, color="#999999")
    ax.plot(a, gmax, color=C_BOUND, lw=3.0, label=r"convex bound $\max_i g_i(\alpha)$", zorder=4)
    ax.axvline(astar, color=C_STAR, lw=1.4, alpha=0.8, zorder=2)
    ax.scatter([astar], [ystar], color=C_STAR, s=110, zorder=6,
               edgecolors="k", linewidths=0.7, label=rf"$\alpha^\star={astar:.2f}$ (bound-optimal)")
    ax.scatter([0.5], [yu], color=C_UNI, s=95, marker="s", zorder=6,
               edgecolors="k", linewidths=0.7, label="uniform")
    ax.annotate(r"uniform pays $%.0f\%%$ more" % (100 * (yu / ystar - 1)),
                (0.5, yu), textcoords="offset points", xytext=(-6, 12),
                fontsize=10, color=C_UNI, ha="right",
                arrowprops=dict(arrowstyle="->", color=C_UNI, lw=1.0))
    ax.set_xlabel(r"weight $\alpha$ (on objective 1)")
    ax.set_ylabel(r"constant-free worst-case bound", color=C_BOUND)
    ax.set_ylim(0, gmax.max() * 1.08)
    ax.set_title(r"Two objectives: $\alpha^\star$ minimizes the convex bound")

    if realized is not None:
        axR = ax.twinx()
        ralph = np.asarray(realized["alphas"]); rworst = np.asarray(realized["worst"])
        axR.plot(ralph, rworst, color="#333333", lw=1.7, marker="o", ms=2.5,
                 label="realized worst-case TV", zorder=3)
        rhat = ralph[int(np.argmin(rworst))]
        axR.annotate(rf"realized argmin $\approx {rhat:.2f}$", (rhat, rworst.min()),
                     textcoords="offset points", xytext=(8, -2), fontsize=9, color="#333333")
        axR.set_ylabel("realized worst-case TV", color="#333333")
        axR.spines["top"].set_visible(False)

    # one combined legend, outside on the right
    h, l = ax.get_legend_handles_labels()
    if realized is not None:
        h2, l2 = axR.get_legend_handles_labels(); h += h2; l += l2
    ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.12, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)
