"""Full analytic-score hero pipeline.

Estimates the four moments and alpha*, sweeps the weight alpha, measures the
realized worst-case TV of the weighted-score chain at each alpha, and produces
the three figures plus a results table.

Run:  python run_analytic.py
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

from targets import make_targets
from diffusion import VPSchedule, AnalyticScore
from moments import estimate_moments
from sampler import sample_weighted
from tv import TVGrid, worst_case_tv

FIGS = Path(__file__).parent / "figs"
RESULTS_JSON = Path(__file__).parent / "results.json"
RESULTS_MD = Path(__file__).parent / "results.md"

# ---- hyperparameters (seeded, logged) -------------------------------------
HP = dict(
    T=400,
    beta_min=1e-4,
    beta_max=0.02,
    n_mc=4000,
    n_tgrid=80,
    n_samples=120_000,
    n_bins=180,
    box=(-3.4, 3.4),
    seed=0,
    moments_seed=11,
)


def alpha_grid(alpha_star: float) -> np.ndarray:
    """Uniform sweep, densified around the predicted minimum."""
    base = np.linspace(0.0, 1.0, 21)                       # step 0.05
    dense = np.linspace(max(0, alpha_star - 0.1),
                        min(1, alpha_star + 0.1), 9)
    pts = np.unique(np.round(np.concatenate([base, dense, [0.5]]), 4))
    return pts


def style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def fig_hero(scores, sched, alpha_star, marg1, marg2, grid, rng, panel_alphas):
    fig, axes = plt.subplots(1, len(panel_alphas), figsize=(3.0 * len(panel_alphas), 3.2))
    for ax, a in zip(axes, panel_alphas):
        s = sample_weighted(scores, a, sched, n_samples=40_000, rng=rng)
        w, _, _ = worst_case_tv(grid, marg1, marg2, s)
        ax.scatter(s[:, 0], s[:, 1], s=0.9, alpha=0.28, color="#1f3b6e", linewidths=0)
        ax.set_xlim(*HP["box"]); ax.set_ylim(*HP["box"])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        tag = "alpha^\\star" if abs(a - alpha_star) < 1e-6 else f"{a:.2f}"
        ax.set_title(rf"$\alpha={tag}$" + "\n" + rf"worstTV$={w:.3f}$", fontsize=10)
    fig.suptitle("Weighted-score samples across $\\alpha$  (left: sad $\\alpha{=}0$, right: happy $\\alpha{=}1$)",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "hero_scatter.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "hero_scatter.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_money(alphas, worst, tv1, tv2, alpha_star, hat_alpha,
              excess_worst=None, hat_alpha_excess=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(alphas, worst, "-o", ms=3, color="#222", label=r"realized worst-case TV")
    ax.plot(alphas, tv1, "--", lw=1, color="#c44", alpha=0.7, label=r"TV$(q_1^{happy},p_1^w)$")
    ax.plot(alphas, tv2, "--", lw=1, color="#46a", alpha=0.7, label=r"TV$(q_1^{sad},p_1^w)$")
    if excess_worst is not None:
        ax.plot(alphas, excess_worst, "-", lw=1.3, color="#777", alpha=0.9,
                label=r"bias-corrected worst-case TV")
    ax.axvline(alpha_star, color="#2a8", lw=2, label=rf"predicted $\alpha^\star={alpha_star:.3f}$")
    ax.axvline(hat_alpha, color="#e80", lw=1.5, ls=":", label=rf"empirical $\hat\alpha={hat_alpha:.3f}$")
    if hat_alpha_excess is not None:
        ax.axvline(hat_alpha_excess, color="#b59", lw=1.5, ls="-.",
                   label=rf"bias-corrected $\hat\alpha={hat_alpha_excess:.3f}$")
    for a, name, col, dy in [(0.0, "single (sad)", "#555", 9), (1.0, "single (happy)", "#555", 9),
                             (0.5, "uniform", "#a4a", -14)]:
        wa = np.interp(a, alphas, worst)
        ax.plot([a], [wa], "s", color=col, ms=8)
        ax.annotate(name, (a, wa), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=9, color=col, fontweight="bold")
    ax.set_xlabel(r"weight $\alpha$ (on the happy face)")
    ax.set_ylabel("total variation at time 1")
    ax.set_title("Worst-case TV bottoms out near the closed-form $\\alpha^\\star$")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "money_worstTV.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "money_worstTV.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_theory(alphas, worst, mom, alpha_star, hat_alpha):
    k = mom.kappa
    g1 = (1 - alphas) * (mom.sigma12 + k * mom.b12)        # decreasing
    g2 = alphas * (mom.sigma21 + k * mom.b21)              # increasing
    gmax = np.maximum(g1, g2)

    fig, axL = plt.subplots(figsize=(6.2, 4.2))
    axL.plot(alphas, g1, "--", color="#c44", lw=1, label=r"$g_{happy}(\alpha)=(1-\alpha)(\sigma_{12}+\kappa b_{12})$")
    axL.plot(alphas, g2, "--", color="#46a", lw=1, label=r"$g_{sad}(\alpha)=\alpha(\sigma_{21}+\kappa b_{21})$")
    axL.plot(alphas, gmax, "-", color="#2a8", lw=2, label=r"predicted V-shape $\max(g_{happy},g_{sad})$")
    axL.axvline(alpha_star, color="#2a8", lw=1, ls="-")
    axL.set_xlabel(r"weight $\alpha$")
    axL.set_ylabel("constant-free bound (a.u.)", color="#2a8")
    axL.tick_params(axis="y", labelcolor="#2a8")

    axR = axL.twinx()
    axR.plot(alphas, worst, "-o", ms=3, color="#222", label="realized worst-case TV")
    axR.axvline(hat_alpha, color="#e80", lw=1.5, ls=":")
    axR.set_ylabel("realized worst-case TV", color="#222")
    axR.spines["top"].set_visible(False)

    lines = axL.get_lines()[:3] + axR.get_lines()[:1]
    axL.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper center")
    axL.set_title(rf"Theory V-shape vs realized TV: argmins $\alpha^\star={alpha_star:.3f}$, $\hat\alpha={hat_alpha:.3f}$")
    fig.tight_layout()
    fig.savefig(FIGS / "theory_match.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "theory_match.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller run for a fast check")
    args = ap.parse_args()
    if args.quick:
        HP.update(T=300, n_mc=1500, n_tgrid=40, n_samples=40_000, n_bins=120)

    style()
    FIGS.mkdir(exist_ok=True)
    t0 = time.time()

    happy, sad = make_targets()
    sched = VPSchedule(T=HP["T"], beta_min=HP["beta_min"], beta_max=HP["beta_max"])
    sA = AnalyticScore(happy, sched)
    sI = AnalyticScore(sad, sched)

    print("[1/3] estimating moments ...")
    mom = estimate_moments(sA, sI, happy, sad, sched,
                           n_mc=HP["n_mc"], n_tgrid=HP["n_tgrid"], seed=HP["moments_seed"])
    alpha_star = mom.alpha_star()
    print(f"  sigma12={mom.sigma12:.4f} sigma21={mom.sigma21:.4f} "
          f"b12={mom.b12:.4f} b21={mom.b21:.4f} kappa={mom.kappa:.4f}")
    print(f"  alpha* = {alpha_star:.4f}  (sigma-only {mom.alpha_sigma_only():.4f}, "
          f"b-only {mom.alpha_b_only():.4f})")

    grid = TVGrid(box=HP["box"], n_bins=HP["n_bins"])
    marg1 = happy.forward(sched.abar_at(1))
    marg2 = sad.forward(sched.abar_at(1))

    alphas = alpha_grid(alpha_star)
    print(f"[2/3] sweeping {len(alphas)} alphas, {HP['n_samples']} samples each ...")
    rng = np.random.default_rng(HP["seed"])
    worst, tv1, tv2 = [], [], []
    for a in alphas:
        s = sample_weighted((sA, sI), a, sched, n_samples=HP["n_samples"], rng=rng)
        w, t1, t2 = worst_case_tv(grid, marg1, marg2, s)
        worst.append(w); tv1.append(t1); tv2.append(t2)
        print(f"  alpha={a:.3f}  TV_happy={t1:.4f}  TV_sad={t2:.4f}  worst={w:.4f}")
    worst = np.array(worst); tv1 = np.array(tv1); tv2 = np.array(tv2)
    hat_alpha = float(alphas[int(np.argmin(worst))])

    # --- estimator-bias correction --------------------------------------
    # The binned-TV estimator has an additive, alpha-independent floor for each
    # objective: even with the exactly-correct score (alpha=1 for A, alpha=0 for
    # I) the sample histogram differs from the sharp analytic target by a
    # histogram-resolution bias.  That floor is a measurement artifact, not part
    # of the theory's TV (whose discretization term Delta_T is alpha-independent
    # and common to both objectives).  Because the two faces have different
    # footprints their floors differ, which biases the *raw* crossing.  We report
    # the floor-corrected ("excess") worst-case TV alongside the raw one.
    floor_A = float(tv1[int(np.argmin(np.abs(alphas - 1.0)))])   # TV_happy at alpha=1
    floor_I = float(tv2[int(np.argmin(np.abs(alphas - 0.0)))])   # TV_sad at alpha=0
    ex1 = np.clip(tv1 - floor_A, 0, None)
    ex2 = np.clip(tv2 - floor_I, 0, None)
    excess_worst = np.maximum(ex1, ex2)
    hat_alpha_excess = float(alphas[int(np.argmin(excess_worst))])

    def at(a):
        return float(np.interp(a, alphas, worst))

    baselines = {
        "single_sad (alpha=0)": at(0.0),
        "single_happy (alpha=1)": at(1.0),
        "uniform (alpha=0.5)": at(0.5),
        "closed-form (alpha*)": at(alpha_star),
        "empirical (hat_alpha)": float(np.min(worst)),
    }

    print("[3/3] figures ...")
    panel_alphas = sorted({0.0, 0.25, round(alpha_star, 4), 0.75, 1.0})
    fig_hero((sA, sI), sched, round(alpha_star, 4), marg1, marg2, grid,
             np.random.default_rng(HP["seed"] + 1), panel_alphas)
    fig_money(alphas, worst, tv1, tv2, alpha_star, hat_alpha,
              excess_worst, hat_alpha_excess)
    fig_theory(alphas, worst, mom, alpha_star, hat_alpha)

    results = {
        "hyperparameters": HP,
        "moments": mom.summary(),
        "alpha_star": alpha_star,
        "hat_alpha": hat_alpha,
        "hat_alpha_excess": hat_alpha_excess,
        "alpha_abs_error": abs(alpha_star - hat_alpha),
        "alpha_abs_error_excess": abs(alpha_star - hat_alpha_excess),
        "estimator_floor": {"floor_happy": floor_A, "floor_sad": floor_I},
        "sweep": {"alphas": alphas.tolist(), "worstTV": worst.tolist(),
                   "TV_happy": tv1.tolist(), "TV_sad": tv2.tolist(),
                   "excess_worstTV": excess_worst.tolist()},
        "baselines_worstTV": baselines,
        "runtime_sec": time.time() - t0,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    write_results_md(results, mom, alpha_star, hat_alpha, baselines,
                     hat_alpha_excess, (floor_A, floor_I))
    print(f"\nDone in {results['runtime_sec']:.1f}s. "
          f"alpha*={alpha_star:.3f}, hat_alpha={hat_alpha:.3f}, "
          f"|diff|={abs(alpha_star-hat_alpha):.3f}")


def write_results_md(results, mom, alpha_star, hat_alpha, baselines,
                     hat_alpha_excess=None, floors=None):
    lines = []
    lines.append("# Analytic-score results\n")
    lines.append("Two related 2D GMM targets: a **happy** and a **sad** smiley face "
                 "(shared outline + eyes, differing expression); analytic true scores "
                 "and Jacobians; deterministic PF-ODE weighted-score chain.\n")
    lines.append("## Moments (time-averaged, under each objective's own marginals)\n")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| $\\sigma_{{12}}$ (score gap under happy) | {mom.sigma12:.4f} |")
    lines.append(f"| $\\sigma_{{21}}$ (score gap under sad) | {mom.sigma21:.4f} |")
    lines.append(f"| $b_{{12}}$ (Jacobian gap under happy) | {mom.b12:.4f} |")
    lines.append(f"| $b_{{21}}$ (Jacobian gap under sad) | {mom.b21:.4f} |")
    lines.append(f"| $\\kappa=\\sqrt{{d/\\log T}}$ | {mom.kappa:.4f} |")
    lines.append("")
    lines.append("## Predicted vs empirical weight\n")
    lines.append("| weight | value |")
    lines.append("|---|---|")
    lines.append(f"| closed-form $\\alpha^\\star$ | **{alpha_star:.4f}** |")
    lines.append(f"| $\\sigma$-only limit ($\\kappa\\to0$) | {mom.alpha_sigma_only():.4f} |")
    lines.append(f"| $b$-only limit ($\\kappa\\to\\infty$) | {mom.alpha_b_only():.4f} |")
    lines.append(f"| empirical oracle $\\hat\\alpha$ (raw TV) | {hat_alpha:.4f} |")
    if hat_alpha_excess is not None:
        lines.append(f"| empirical oracle $\\hat\\alpha$ (bias-corrected) | {hat_alpha_excess:.4f} |")
    lines.append(f"| $|\\alpha^\\star-\\hat\\alpha|$ (raw) | {abs(alpha_star-hat_alpha):.4f} |")
    if hat_alpha_excess is not None:
        lines.append(f"| $|\\alpha^\\star-\\hat\\alpha|$ (bias-corrected) | {abs(alpha_star-hat_alpha_excess):.4f} |")
    lines.append("")
    if floors is not None:
        lines.append(f"Estimator floors (single-objective TV at the matched weight, "
                     f"pure histogram/discretization bias): "
                     f"$\\mathrm{{floor}}_{{happy}}={floors[0]:.4f}$, "
                     f"$\\mathrm{{floor}}_{{sad}}={floors[1]:.4f}$. "
                     f"The bias-corrected oracle subtracts these before taking the max.\n")
    lines.append("## Worst-case TV at the baselines\n")
    lines.append("| weight | worst-case TV |")
    lines.append("|---|---|")
    for k, v in baselines.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    star = baselines["closed-form (alpha*)"]
    uni = baselines["uniform (alpha=0.5)"]
    s0 = baselines["single_sad (alpha=0)"]
    s1 = baselines["single_happy (alpha=1)"]
    lines.append(f"The closed-form weight beats uniform by "
                 f"{uni - star:+.4f} TV and the single-objective baselines by "
                 f"{min(s0, s1) - star:+.4f} (vs the better single objective).\n")
    RESULTS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
