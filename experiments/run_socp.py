"""General multi-objective experiment (Key Result I), m = 3.

1. Estimate the Gram matrices Sigma_i and Jacobian vectors b_i by Monte Carlo.
2. Solve the reduced constant-free SOCP for the ground-truth weight w*.
3. Run MG-AMOO and PAMOO and show they reach w* at the predicted O(1/sqrt(K))
   rate (convergence figure).
4. Validate on the REALIZED weighted-score chain: a ternary heat-map of the
   realized worst-case TV over the simplex, with w* sitting on the empirical
   minimum and beating uniform / the single-objective vertices.

Run:  python run_socp.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_targets_m
from diffusion import VPSchedule, AnalyticScore
from moments import estimate_gram
from socp import (kappa_of, g_values, max_g, solve_socp, mg_amoo, pamoo)
from sampler import sample_weighted_multi
from tv import TVGrid, worst_case_tv_multi
import pubfigs as pf

FIGS = Path(__file__).parent / "figs"
HP = dict(T=400, n_mc=2500, n_tgrid=60, n_samples=45_000, n_bins=180,
          box=(-3.6, 3.6), grid_res=7, K=4000, seed=0, moments_seed=11)


def bary_to_xy(W):
    """Barycentric (n,3) -> 2D coords for a ternary plot."""
    W = np.asarray(W)
    x = W[:, 1] + 0.5 * W[:, 2]
    y = (np.sqrt(3) / 2) * W[:, 2]
    return x, y


def simplex_grid(res):
    """All (i,j,k)/res compositions on the 3-simplex."""
    pts = []
    for i in range(res + 1):
        for j in range(res + 1 - i):
            k = res - i - j
            pts.append([i / res, j / res, k / res])
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        HP.update(T=300, n_mc=1200, n_tgrid=35, n_samples=18_000, n_bins=120, grid_res=5, K=2000)

    pf.pubstyle()
    FIGS.mkdir(exist_ok=True)
    t0 = time.time()
    m = 3

    targets = make_targets_m(m)
    names = ["grid", "blob", "ring"][:m]
    sched = VPSchedule(T=HP["T"])
    scores = [AnalyticScore(t, sched) for t in targets]

    print("[1/4] estimating Gram matrices Sigma_i, b_i ...")
    Sigmas, bs = estimate_gram(scores, targets, sched,
                               n_mc=HP["n_mc"], n_tgrid=HP["n_tgrid"], seed=HP["moments_seed"])
    kappa = kappa_of(sched)
    for i in range(m):
        print(f"  obj {i} ({names[i]}): diag(Sigma)={np.round(np.diag(Sigmas[i]),2)} b={np.round(bs[i],2)}")

    print("[2/4] solving SOCP for w* ...")
    w_star, val = solve_socp(Sigmas, bs, kappa)
    print(f"  w*={np.round(w_star,4)}  max_i g_i(w*)={val:.4f}  "
          f"uniform max_g={max_g(np.ones(m)/m, Sigmas, bs, kappa):.4f}")

    print("[3/4] MG-AMOO / PAMOO convergence ...")
    wbar_mg, hist_mg, traj_mg = mg_amoo(Sigmas, bs, kappa, K=HP["K"], return_traj=True)
    wbar_pa, hist_pa, traj_pa = pamoo(Sigmas, bs, kappa, K=HP["K"], return_traj=True)
    print(f"  MG-AMOO w_bar={np.round(wbar_mg,4)}  final gap={hist_mg[-1]-val:.5f}")
    print(f"  PAMOO   w_bar={np.round(wbar_pa,4)}  final gap={hist_pa[-1]-val:.5f}")

    pf.fig_bound_surface_3d(Sigmas, bs, kappa, w_star, names, str(FIGS / "socp_surface3d"))
    pf.fig_simplex_trajectories(Sigmas, bs, kappa, w_star, traj_mg, traj_pa, names,
                                str(FIGS / "socp_trajectories"))
    pf.fig_convergence(hist_mg, hist_pa, val, str(FIGS / "socp_convergence"))

    print("[4/4] realized worst-case TV over the simplex ...")
    tvgrid = TVGrid(box=HP["box"], n_bins=HP["n_bins"])
    margs = [t.forward(sched.abar_at(1)) for t in targets]
    W = simplex_grid(HP["grid_res"])
    rng = np.random.default_rng(HP["seed"])
    realized = np.empty(len(W))
    for idx, w in enumerate(W):
        s = sample_weighted_multi(scores, w, sched, n_samples=HP["n_samples"], rng=rng)
        realized[idx], _ = worst_case_tv_multi(tvgrid, margs, s)
    hat = W[int(np.argmin(realized))]
    # realized worst-case TV at the SOCP optimum and the algorithm outputs
    def realized_at(w):
        s = sample_weighted_multi(scores, w, sched, n_samples=HP["n_samples"], rng=rng)
        return worst_case_tv_multi(tvgrid, margs, s)[0]
    r_star = realized_at(w_star); r_mg = realized_at(wbar_mg); r_pa = realized_at(wbar_pa)

    pf.fig_realized_simplex(W, realized, w_star, names, str(FIGS / "socp_simplex"), w_hat=hat)

    res = {
        "hyperparameters": HP, "m": m, "names": names, "kappa": kappa,
        "Sigmas": [S.tolist() for S in Sigmas], "bs": [b.tolist() for b in bs],
        "w_star": w_star.tolist(), "socp_value": val,
        "mg_amoo": {"w_bar": wbar_mg.tolist(), "final_gap": float(hist_mg[-1] - val)},
        "pamoo": {"w_bar": wbar_pa.tolist(), "final_gap": float(hist_pa[-1] - val)},
        "realized": {"grid_argmin": hat.tolist(),
                     "W_grid": W.tolist(), "worstTV_grid": realized.tolist(),
                     "worstTV_at_wstar": float(r_star),
                     "worstTV_at_uniform": float(realized_at(np.ones(m)/m)),
                     "worstTV_at_mg": float(r_mg), "worstTV_at_pamoo": float(r_pa),
                     "worstTV_vertices": [float(realized_at(np.eye(m)[i])) for i in range(m)]},
        "runtime_sec": time.time() - t0,
    }
    (Path(__file__).parent / "results_socp.json").write_text(json.dumps(res, indent=2))
    print(f"\nDone in {res['runtime_sec']:.1f}s.")
    print(f"  w*={np.round(w_star,3)}  grid-argmin={np.round(hat,3)}")
    print(f"  realized worstTV: w*={r_star:.4f} uniform={res['realized']['worstTV_at_uniform']:.4f} "
          f"vertices={[round(v,3) for v in res['realized']['worstTV_vertices']]}")


if __name__ == "__main__":
    main()
