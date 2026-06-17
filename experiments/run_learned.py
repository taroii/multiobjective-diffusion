"""Learned-score robustness check (Sec. 8 of the plan).

Train two tiny per-objective score MLPs by independent single-objective DSM,
re-estimate the moments and alpha* from the *learned* scores (Jacobians via
autograd), rerun the weight sweep, and compare the learned-score empirical
minimizer against the analytic prediction.  A small shift is expected and is
itself informative (it motivates the learned-score concentration item in the
paper's open problems); we quantify it rather than hide it.

Run:  python run_learned.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_targets
from diffusion import VPSchedule
from moments import estimate_moments
from sampler import sample_weighted
from tv import TVGrid, worst_case_tv
from nets import train_score_mlp, LearnedScore

FIGS = Path(__file__).parent / "figs"
RESULTS_JSON = Path(__file__).parent / "results_learned.json"

HP = dict(
    T=400, n_mc=2000, n_tgrid=60, n_samples=60_000, n_bins=180,
    box=(-3.4, 3.4), seed=0, moments_seed=11,
    train_steps=6000, hidden=128, lr=2e-3,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        HP.update(T=300, n_mc=800, n_tgrid=30, n_samples=20_000, n_bins=120,
                  train_steps=1500)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    FIGS.mkdir(exist_ok=True)
    t0 = time.time()

    happy, sad = make_targets()
    sched = VPSchedule(T=HP["T"])

    print("[1/4] training per-objective score MLPs ...")
    print("  happy face:")
    mA = train_score_mlp(happy, sched, steps=HP["train_steps"], hidden=HP["hidden"],
                         lr=HP["lr"], seed=1)
    print("  sad face:")
    mI = train_score_mlp(sad, sched, steps=HP["train_steps"], hidden=HP["hidden"],
                         lr=HP["lr"], seed=2)
    lsA = LearnedScore(mA, sched)
    lsI = LearnedScore(mI, sched)

    print("[2/4] estimating moments from learned scores ...")
    mom = estimate_moments(lsA, lsI, happy, sad, sched,
                           n_mc=HP["n_mc"], n_tgrid=HP["n_tgrid"], seed=HP["moments_seed"])
    alpha_star = mom.alpha_star()
    print(f"  (learned) sigma12={mom.sigma12:.3f} sigma21={mom.sigma21:.3f} "
          f"b12={mom.b12:.3f} b21={mom.b21:.3f}")
    print(f"  (learned) alpha* = {alpha_star:.4f}")

    grid = TVGrid(box=HP["box"], n_bins=HP["n_bins"])
    marg1 = happy.forward(sched.abar_at(1))
    marg2 = sad.forward(sched.abar_at(1))

    print("[3/4] sweeping alpha with the learned-score chain ...")
    base = np.linspace(0, 1, 16)
    dense = np.linspace(max(0, alpha_star - 0.1), min(1, alpha_star + 0.1), 7)
    alphas = np.unique(np.round(np.concatenate([base, dense, [0.5]]), 4))
    rng = np.random.default_rng(HP["seed"])
    worst, tv1, tv2 = [], [], []
    for a in alphas:
        s = sample_weighted((lsA, lsI), a, sched, n_samples=HP["n_samples"],
                            rng=rng, batch_size=20000)
        w, t1, t2 = worst_case_tv(grid, marg1, marg2, s)
        worst.append(w); tv1.append(t1); tv2.append(t2)
        print(f"  alpha={a:.3f}  TV_happy={t1:.4f}  TV_sad={t2:.4f}  worst={w:.4f}")
    worst = np.array(worst); tv1 = np.array(tv1); tv2 = np.array(tv2)
    hat_alpha = float(alphas[int(np.argmin(worst))])

    # load analytic prediction for comparison if available
    analytic = None
    aj = Path(__file__).parent / "results.json"
    if aj.exists():
        analytic = json.loads(aj.read_text())

    print("[4/4] figure + results ...")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(alphas, worst, "-o", ms=3, color="#222", label="learned worst-case TV")
    ax.axvline(alpha_star, color="#2a8", lw=2, label=rf"learned $\alpha^\star={alpha_star:.3f}$")
    ax.axvline(hat_alpha, color="#e80", lw=1.5, ls=":", label=rf"learned $\hat\alpha={hat_alpha:.3f}$")
    if analytic is not None:
        ax.axvline(analytic["alpha_star"], color="#88c", lw=1.5, ls="--",
                   label=rf"analytic $\alpha^\star={analytic['alpha_star']:.3f}$")
    ax.set_xlabel(r"weight $\alpha$ (on the happy face)")
    ax.set_ylabel("total variation at time 1")
    ax.set_title("Learned-score robustness check")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "learned_worstTV.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "learned_worstTV.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    results = {
        "hyperparameters": HP,
        "moments_learned": mom.summary(),
        "alpha_star_learned": alpha_star,
        "hat_alpha_learned": hat_alpha,
        "sweep": {"alphas": alphas.tolist(), "worstTV": worst.tolist(),
                   "TV_happy": tv1.tolist(), "TV_sad": tv2.tolist()},
        "runtime_sec": time.time() - t0,
    }
    if analytic is not None:
        results["analytic_alpha_star"] = analytic["alpha_star"]
        results["shift_vs_analytic"] = alpha_star - analytic["alpha_star"]
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nDone in {results['runtime_sec']:.1f}s. learned alpha*={alpha_star:.3f}, "
          f"learned hat_alpha={hat_alpha:.3f}")
    if analytic is not None:
        print(f"shift vs analytic alpha* ({analytic['alpha_star']:.3f}): "
              f"{alpha_star - analytic['alpha_star']:+.3f}")


if __name__ == "__main__":
    main()
