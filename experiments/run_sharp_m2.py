"""Sharper two-objective instance: a hard 16-mode grid vs an easy broad blob.

The two targets differ in DIFFICULTY (sharpness / multimodality), not just
location, so the pairwise score/Jacobian gaps are large and asymmetric.  This
gives a deep worst-case-TV bowl in which alpha* tilts hard toward the difficult
target and uniform pays a real price -- a clear money figure (vs. the near-tie
of the very-similar faces).

Run:  python run_sharp_m2.py [--quick]
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

from targets import make_sharp_pair
from diffusion import VPSchedule, AnalyticScore
from moments import estimate_moments
from sampler import sample_weighted
from tv import TVGrid, worst_case_tv

FIGS = Path(__file__).parent / "figs"
HP = dict(T=400, beta_min=1e-4, beta_max=0.02, n_mc=4000, n_tgrid=80,
          n_samples=120_000, n_bins=200, box=(-3.8, 3.8), seed=0, moments_seed=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        HP.update(T=300, n_mc=1500, n_tgrid=40, n_samples=40_000, n_bins=140)

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    FIGS.mkdir(exist_ok=True)
    t0 = time.time()

    hard, easy = make_sharp_pair()           # objective 1 = hard grid, 2 = easy blob
    sched = VPSchedule(T=HP["T"], beta_min=HP["beta_min"], beta_max=HP["beta_max"])
    sH, sE = AnalyticScore(hard, sched), AnalyticScore(easy, sched)

    print("[1/3] moments ...")
    mom = estimate_moments(sH, sE, hard, easy, sched,
                           n_mc=HP["n_mc"], n_tgrid=HP["n_tgrid"], seed=HP["moments_seed"])
    astar = mom.alpha_star()
    print(f"  sigma12={mom.sigma12:.3f} sigma21={mom.sigma21:.3f} "
          f"b12={mom.b12:.3f} b21={mom.b21:.3f} kappa={mom.kappa:.3f}")
    print(f"  alpha*={astar:.4f} (sigma-only {mom.alpha_sigma_only():.4f}, b-only {mom.alpha_b_only():.4f})")

    grid = TVGrid(box=HP["box"], n_bins=HP["n_bins"])
    mH, mE = hard.forward(sched.abar_at(1)), easy.forward(sched.abar_at(1))
    base = np.linspace(0, 1, 21)
    dense = np.linspace(max(0, astar - 0.1), min(1, astar + 0.1), 9)
    alphas = np.unique(np.round(np.concatenate([base, dense, [0.5]]), 4))

    print(f"[2/3] sweeping {len(alphas)} alphas, {HP['n_samples']} samples ...")
    rng = np.random.default_rng(HP["seed"])
    worst, tvH, tvE = [], [], []
    for a in alphas:
        s = sample_weighted((sH, sE), a, sched, n_samples=HP["n_samples"], rng=rng)
        w, t1, t2 = worst_case_tv(grid, mH, mE, s)
        worst.append(w); tvH.append(t1); tvE.append(t2)
        print(f"  alpha={a:.3f} TV_hard={t1:.4f} TV_easy={t2:.4f} worst={w:.4f}")
    worst, tvH, tvE = map(np.array, (worst, tvH, tvE))
    hat = float(alphas[int(np.argmin(worst))])

    def at(a):
        return float(np.interp(a, alphas, worst))
    baselines = {"single_easy (a=0)": at(0.0), "single_hard (a=1)": at(1.0),
                 "uniform (a=0.5)": at(0.5), "closed-form (a*)": at(astar),
                 "empirical (hat)": float(worst.min())}

    print("[3/3] figures ...")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(alphas, worst, "-o", ms=3, color="#222", label="realized worst-case TV")
    ax.plot(alphas, tvH, "--", lw=1, color="#c44", alpha=0.7, label=r"TV$(q_1^{hard},p_1^w)$")
    ax.plot(alphas, tvE, "--", lw=1, color="#46a", alpha=0.7, label=r"TV$(q_1^{easy},p_1^w)$")
    ax.axvline(astar, color="#2a8", lw=2, label=rf"predicted $\alpha^\star={astar:.3f}$")
    ax.axvline(hat, color="#e80", lw=1.5, ls=":", label=rf"empirical $\hat\alpha={hat:.3f}$")
    for a, name, col, dy in [(0.0, "single (easy)", "#555", 9), (1.0, "single (hard)", "#555", 9),
                             (0.5, "uniform", "#a4a", 9)]:
        ax.plot([a], [at(a)], "s", color=col, ms=8)
        ax.annotate(name, (a, at(a)), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=9, color=col, fontweight="bold")
    ax.set_xlabel(r"weight $\alpha$ (on the hard grid target)")
    ax.set_ylabel("total variation at time 1")
    ax.set_title("Sharper $m{=}2$: $\\alpha^\\star$ beats uniform by a clear margin")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "sharp_money.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "sharp_money.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # hero scatter of the morph (hard grid <-> easy blob)
    panels = sorted({0.0, 0.25, round(astar, 3), 0.75, 1.0})
    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 3.0))
    rng2 = np.random.default_rng(HP["seed"] + 1)
    for ax, a in zip(axes, panels):
        s = sample_weighted((sH, sE), a, sched, n_samples=45_000, rng=rng2)
        ax.scatter(s[:, 0], s[:, 1], s=0.6, alpha=0.25, color="#1f3b6e", lw=0)
        ax.set_xlim(*HP["box"]); ax.set_ylim(*HP["box"]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        tag = "alpha^\\star" if a == round(astar, 3) else f"{a}"
        ax.set_title(rf"$\alpha={tag}$")
    fig.suptitle("easy blob ($\\alpha{=}0$)  $\\to$  hard grid ($\\alpha{=}1$)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "sharp_hero.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "sharp_hero.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    res = {"hyperparameters": HP, "moments": mom.summary(), "alpha_star": astar,
           "hat_alpha": hat, "alpha_abs_error": abs(astar - hat),
           "sweep": {"alphas": alphas.tolist(), "worstTV": worst.tolist(),
                     "TV_hard": tvH.tolist(), "TV_easy": tvE.tolist()},
           "baselines_worstTV": baselines, "runtime_sec": time.time() - t0}
    (Path(__file__).parent / "results_sharp.json").write_text(json.dumps(res, indent=2))
    print(f"\nDone in {res['runtime_sec']:.1f}s. alpha*={astar:.3f} hat={hat:.3f} "
          f"|diff|={abs(astar-hat):.3f}")
    print(f"  worstTV: uniform={baselines['uniform (a=0.5)']:.4f} "
          f"alpha*={baselines['closed-form (a*)']:.4f} "
          f"single_hard={baselines['single_hard (a=1)']:.4f} "
          f"single_easy={baselines['single_easy (a=0)']:.4f}")


if __name__ == "__main__":
    main()
