"""Regenerate the publication BOUND figures from saved results (no diffusion).

The bound-based figures depend only on the moments (Gram matrices Sigma_i,
vectors b_i, and the four two-objective scalars), which are stored in
``results_socp.json`` (m=3) and ``results.json`` (m=2).  This script re-solves
the SOCP, re-runs MG-AMOO / PAMOO, and re-renders every bound figure in a few
seconds -- handy for restyling without re-running the expensive diffusion
pipelines.

Run:  python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from socp import solve_socp, mg_amoo, pamoo, max_g
import pubfigs as pf

HERE = Path(__file__).parent
FIGS = HERE / "figs"


def m3_bound_figures(K: int = 4000):
    d = json.loads((HERE / "results_socp.json").read_text())
    Sigmas = [np.array(S) for S in d["Sigmas"]]
    bs = [np.array(b) for b in d["bs"]]
    kappa, names = d["kappa"], d["names"]
    w_star, val = solve_socp(Sigmas, bs, kappa)
    uni = max_g(np.ones(len(names)) / len(names), Sigmas, bs, kappa)
    print(f"m=3: w*={np.round(w_star,3)}  bound(w*)={val:.3f}  bound(uniform)={uni:.3f} "
          f"({100*(uni/val-1):.0f}% higher)")
    _, hist_mg, traj_mg = mg_amoo(Sigmas, bs, kappa, K=K, return_traj=True)
    _, hist_pa, traj_pa = pamoo(Sigmas, bs, kappa, K=K, return_traj=True)
    pf.fig_bound_surface_3d(Sigmas, bs, kappa, w_star, names, str(FIGS / "socp_surface3d"))
    pf.fig_simplex_trajectories(Sigmas, bs, kappa, w_star, traj_mg, traj_pa, names,
                                str(FIGS / "socp_trajectories"))
    pf.fig_convergence(hist_mg, hist_pa, val, str(FIGS / "socp_convergence"))
    rl = d.get("realized", {})
    if "worstTV_grid" in rl:    # regenerate the realized heat-map from saved grid
        pf.fig_realized_simplex(np.array(rl["W_grid"]), np.array(rl["worstTV_grid"]),
                                w_star, names, str(FIGS / "socp_simplex"),
                                w_hat=np.array(rl["grid_argmin"]))


def m2_bound_figure():
    r = json.loads((HERE / "results.json").read_text())
    mom = r["moments"]
    realized = {"alphas": r["sweep"]["alphas"], "worst": r["sweep"]["worstTV"]}
    pf.fig_m2_bound(mom["sigma12"], mom["sigma21"], mom["b12"], mom["b21"], mom["kappa"],
                    str(FIGS / "m2_bound"), realized=realized)


def main():
    pf.pubstyle()
    FIGS.mkdir(exist_ok=True)
    if (HERE / "results_socp.json").exists():
        m3_bound_figures()
    if (HERE / "results.json").exists():
        m2_bound_figure()
    print("regenerated publication bound figures")


if __name__ == "__main__":
    main()
