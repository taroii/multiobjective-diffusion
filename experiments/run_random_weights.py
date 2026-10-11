"""Minimax weights vs. random weights and other weight-selection rules.

For every configuration and every seed we build Gaussian targets and evaluate
the true worst-case total variation F_T(w) = max_i TV(q_1^i, p_1^w) of the
discretized sampler for

  ours      minimax schedule (eq. (minimax)) and minimax constant weight;
  random    w ~ Uniform(Delta^m), either kept fixed for the whole run or redrawn
            independently at every step;
  equal     w = 1/m;
  uniform-average rule   the constant weight minimizing the uniform time average
            of the score error (the comparison rule of the experiments section);
  best constant found    a direct search over constant weights for the smallest
            F_T (an upper bound on the optimum over constant weights).

Nothing is simulated: the sampler moments are propagated exactly and the TVs
are computed by quadrature, so the only randomness is the one controlled by the
seed (the random weights and, for the random-instance configurations, the
targets).

Run:  python run_random_weights.py            (all configurations, 100 seeds)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import functools
import json
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gaussian import GaussianTargets, Schedule, worst_case_tv
from minimax import minimax_constant, minimax_schedule, uniform_average_constant

OUT = Path(__file__).parent / "results"
C0, C1 = 4.0, 8.0          # schedule constants of eq. (schedule)
BEAT_TOL = 1e-4            # a random draw "beats" minimax only beyond quadrature error


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------
def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def paper_instance(T: int = 3000, radius: float = 2.86):
    """The instance of Section (experiments): m = 3, Sigma_0 = diag(30, 1), means
    equidistant (Mahalanobis distance `radius`) from their center in Sigma_0^{-1}.
    Target 1 lies on the high-variance axis, which is the orientation for which the
    uniform-average rule drops it and selects [0, 1/2, 1/2]."""
    S0 = np.diag([30.0, 1.0])
    ang = np.deg2rad([0.0, 120.0, 240.0])
    unit = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    mus = radius * unit * np.sqrt(np.diag(S0))[None]
    return GaussianTargets(mus, np.repeat(S0[None], 3, axis=0), Schedule(T, C0, C1))


def common_instance(rng, m: int, T: int = 1000, spread: float = 1.5):
    """Random targets with a common covariance (the setting of Corollary (gauss)).

    Sigma_0 has condition number log-uniform in [1, 30] and a uniformly random
    orientation; the means are `spread` * N(0, I) in the whitened coordinates.
    """
    kappa = np.exp(rng.uniform(0.0, np.log(30.0)))
    R = _rot(rng.uniform(0.0, np.pi))
    root = R @ np.diag(np.sqrt([kappa, 1.0]))
    mus = spread * rng.normal(size=(m, 2)) @ root.T
    return GaussianTargets(mus, np.repeat((root @ root.T)[None], m, axis=0), Schedule(T, C0, C1))


def hetero_instance(rng, m: int, T: int = 1000, spread: float = 1.5):
    """Random targets with different covariances (outside Corollary (gauss)).

    S_i = s_i R_i diag(sqrt(k_i), 1/sqrt(k_i)) R_i^T with scale s_i log-uniform in
    [1/2, 2], condition number k_i log-uniform in [1, 10], random orientation;
    means are `spread` * N(0, I).
    """
    covs = []
    for _ in range(m):
        k = np.exp(rng.uniform(0.0, np.log(10.0)))
        s = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
        R = _rot(rng.uniform(0.0, np.pi))
        covs.append(s * R @ np.diag([np.sqrt(k), 1.0 / np.sqrt(k)]) @ R.T)
    mus = spread * rng.normal(size=(m, 2))
    return GaussianTargets(mus, np.stack(covs), Schedule(T, C0, C1))


CONFIGS = {
    # name:        (kind,     m, T)
    "paper_m3":    ("paper",  3, 3000),
    "common_m2":   ("common", 2, 1000),
    "common_m3":   ("common", 3, 1000),
    "common_m5":   ("common", 5, 1000),
    "hetero_m2":   ("hetero", 2, 1000),
    "hetero_m3":   ("hetero", 3, 1000),
    "hetero_m5":   ("hetero", 5, 1000),
}


# ---------------------------------------------------------------------------
# one seed
# ---------------------------------------------------------------------------
METHODS = ("minimax_schedule", "minimax_constant", "equal", "uniform_average", "best_constant")
SEARCH_RADII = (0.15, 0.07, 0.03, 0.015, 0.007, 0.003, 0.0015)
SEARCH_BATCH = 64


def best_constant(tg, init_w, init_F, rng):
    """Direct search for the constant weight with the smallest true F_T.

    Starts from the best of the already evaluated weights and repeatedly samples
    a shrinking neighbourhood on the simplex.  The result is the best weight
    found, i.e. an upper bound on min_w F_T(w) over constant weights.
    """
    k = int(np.argmin(init_F))
    w_best, F_best = init_w[k].copy(), float(init_F[k])
    for radius in SEARCH_RADII:
        cand = np.clip(w_best + radius * rng.normal(size=(SEARCH_BATCH, tg.m)), 0.0, None)
        cand = cand[cand.sum(axis=1) > 0]
        cand /= cand.sum(axis=1, keepdims=True)
        Fc, _ = worst_case_tv(tg, cand)
        if Fc.min() < F_best:
            w_best, F_best = cand[np.argmin(Fc)].copy(), float(Fc.min())
    return w_best, F_best


def _solve(tg):
    F = tg.gram_features()
    return (F, minimax_schedule(F, tg.sched.beta)[0], minimax_constant(F, tg.sched.beta)[0],
            uniform_average_constant(F)[0])


@functools.lru_cache(maxsize=2)
def _paper_solution(T):
    tg = paper_instance(T)
    return (tg,) + _solve(tg)


def run_seed(args):
    name, seed, n_const, n_sched = args
    kind, m, T = CONFIGS[name]
    # keyed on the configuration name, so adding or reordering configurations
    # does not change any instance
    streams = np.random.SeedSequence([zlib.crc32(name.encode()), seed]).spawn(4)
    rng_inst, rng_const, rng_sched, rng_search = (np.random.default_rng(x) for x in streams)

    if kind == "paper":
        tg, F, W_ms, w_mc, w_ua = _paper_solution(T)
    else:
        tg = (common_instance if kind == "common" else hetero_instance)(rng_inst, m, T)
        F, W_ms, w_mc, w_ua = _solve(tg)

    ones = np.ones(m)
    w_rand = rng_const.dirichlet(ones, size=n_const)                 # uniform on the simplex
    W_rand = rng_sched.dirichlet(ones, size=(n_sched, T + 1))

    const = np.concatenate([w_mc[None], ones[None] / m, w_ua[None], w_rand])
    sched = np.concatenate([W_ms[None], W_rand])
    F_const, _, e1 = worst_case_tv(tg, const, return_err=True)
    F_sched, _, e2 = worst_case_tv(tg, sched, return_err=True)
    w_best, F_best = best_constant(tg, const, F_const, rng_search)
    Phi_const = tg.V(const, F).max(axis=1)
    Phi_sched = tg.V(sched, F).max(axis=1)

    out = dict(
        seed=seed,
        F_minimax_schedule=F_sched[0], F_minimax_constant=F_const[0], F_equal=F_const[1],
        F_uniform_average=F_const[2], F_best_constant=F_best,
        F_random_constant=F_const[3:], F_random_schedule=F_sched[1:],
        Phi_minimax_schedule=Phi_sched[0], Phi_minimax_constant=Phi_const[0], Phi_equal=Phi_const[1],
        Phi_uniform_average=Phi_const[2], Phi_best_constant=tg.V(w_best[None], F).max(),
        Phi_random_constant=Phi_const[3:], Phi_random_schedule=Phi_sched[1:],
        w_minimax_constant=w_mc, w_uniform_average=w_ua, w_best_constant=w_best,
        quad_err=max(e1, e2),
    )
    if kind in ("paper", "common"):
        # worst-case Mahalanobis distance of the combined mean to the target means
        Pi = np.linalg.inv(tg.covs[0])
        dm = (np.concatenate([const, w_best[None]]) @ tg.mus)[:, None, :] - tg.mus[None]
        dist = np.sqrt(np.einsum("bid,de,bie->bi", dm, Pi, dm)).max(axis=1)
        out.update(dist_minimax_constant=dist[0], dist_equal=dist[1], dist_uniform_average=dist[2],
                   dist_random_constant=dist[3:-1], dist_best_constant=dist[-1])
    return out


# ---------------------------------------------------------------------------
# statistics over seeds
# ---------------------------------------------------------------------------
def _boot_ci(x, rng, n_boot=10_000):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return [float(lo), float(hi)]


def _paired(ours, other, rng):
    """Paired comparison over seeds: `other` - `ours`, positive when ours is better.

    `other` is (n_seeds,) for a deterministic rule, or (n_seeds, R) for random draws,
    in which case the per-seed value is the mean over the draws.
    """
    draws = other if other.ndim == 2 else other[:, None]
    diff = draws.mean(axis=1) - ours
    beats = draws < ours[:, None] - BEAT_TOL
    q25, q50, q75 = np.percentile(diff, [25, 50, 75])
    return dict(
        mean=float(draws.mean()),
        diff_mean=float(diff.mean()), diff_ci95=_boot_ci(diff, rng),
        diff_median=float(q50), diff_iqr=[float(q25), float(q75)],
        frac_seeds_beating_ours=float((diff < -BEAT_TOL).mean()),
        frac_draws_beating_ours=float(beats.mean()),
        frac_seeds_with_any_draw_beating_ours=float(beats.any(axis=1).mean()),
        largest_gain_over_ours=float((ours[:, None] - draws).max()),
    )


def summarize(rows):
    rng = np.random.default_rng(12345)
    col = lambda k: np.array([r[k] for r in rows])
    s = dict(n_seeds=len(rows), quad_err_max=float(col("quad_err").max()))
    for k in METHODS:
        x = col(f"F_{k}")
        s[f"F_{k}"] = dict(mean=float(x.mean()), std=float(x.std(ddof=1)), median=float(np.median(x)))
    others = METHODS + ("random_constant", "random_schedule")
    for ours in ("minimax_schedule", "minimax_constant"):
        s[f"vs_{ours}"] = {k: _paired(col(f"F_{ours}"), col(f"F_{k}"), rng) for k in others if k != ours}
    # how the random draws fare against the uniform-average rule
    s["frac_random_constant_draws_beating_uniform_average"] = float(
        (col("F_random_constant") < col("F_uniform_average")[:, None] - BEAT_TOL).mean())
    for k in METHODS + ("random_constant",):
        s[f"Phi_{k}_mean"] = float(col(f"Phi_{k}").mean())
    if "dist_minimax_constant" in rows[0]:
        d0 = col("dist_best_constant")
        ok = d0 > 1e-9          # guard against a zero optimum (coincident means)
        for k in ("minimax_constant", "equal", "uniform_average"):
            s[f"dist_excess_{k}"] = float((col(f"dist_{k}")[ok] / d0[ok] - 1.0).mean())
        s["dist_excess_random_constant"] = float(
            (col("dist_random_constant")[ok].mean(axis=1) / d0[ok] - 1.0).mean())
    return s


LABELS = {"minimax_schedule": "ours (schedule)", "minimax_constant": "ours (constant)",
          "equal": "equal weights", "uniform_average": "uniform-average rule",
          "best_constant": "best constant found", "random_constant": "random, fixed",
          "random_schedule": "random, per step"}


def to_markdown(summary, meta):
    order = ("random_constant", "random_schedule", "equal", "uniform_average",
             "minimax_constant", "minimax_schedule", "best_constant")
    L = ["# Minimax weights vs. random weights and other rules", "",
         f"{meta['n_seeds']} seeds per configuration; per seed, {meta['n_const']} fixed random weights and "
         f"{meta['n_sched']} per-step random schedules (uniform on the simplex). $F_T$ is the exact worst-case "
         "TV of the discretized sampler. 'paper' is the fixed instance of the experiments section (only the "
         "random weights change with the seed); 'common'/'hetero' redraw the targets (common / different "
         "covariances) for every seed. 'Best constant found' is a direct search over constant weights.", "",
         "## Mean worst-case TV $F_T$ over seeds", "",
         "| config | " + " | ".join(LABELS[k] for k in order) + " |", "|---|" + "---|" * len(order)]
    for name, s in summary.items():
        v = s["vs_minimax_schedule"]
        cells = [f"{s['F_' + k]['mean']:.4f}" if k in METHODS else f"{v[k]['mean']:.4f}" for k in order]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    for ours in ("minimax_schedule", "minimax_constant"):
        L += ["", f"## Paired comparison with {LABELS[ours]}", "",
              "Excess = other − ours, per seed (random: mean over the seed's draws); positive means ours is better. "
              "'Beats ours' counts seeds (random: individual draws) that are better than ours by more than "
              f"{meta['beat_tol']:g}.", "",
              "| config | other | median excess [IQR] | mean excess [95% CI] | beats ours | largest gain over ours |",
              "|---|---|---|---|---|---|"]
        for name, s in summary.items():
            for k, a in s[f"vs_{ours}"].items():
                beat = a["frac_draws_beating_ours"] if k.startswith("random") else a["frac_seeds_beating_ours"]
                L.append(f"| {name} | {LABELS[k]} | {a['diff_median']:+.4f} [{a['diff_iqr'][0]:+.4f}, "
                         f"{a['diff_iqr'][1]:+.4f}] | {a['diff_mean']:+.4f} [{a['diff_ci95'][0]:+.4f}, "
                         f"{a['diff_ci95'][1]:+.4f}] | {100 * beat:.1f}% | {a['largest_gain_over_ours']:+.4f} |")
    L += ["", "## Mahalanobis distance, common-covariance configurations (mean excess over the best constant found)",
          "", "| config | ours (constant) | equal weights | uniform-average rule | random, fixed "
          "| random draws beating the uniform-average rule (in TV) |", "|---|---|---|---|---|---|"]
    for name, s in summary.items():
        if "dist_excess_equal" in s:
            L.append(f"| {name} | {100 * s['dist_excess_minimax_constant']:+.1f}% "
                     f"| {100 * s['dist_excess_equal']:+.1f}% | {100 * s['dist_excess_uniform_average']:+.1f}% "
                     f"| {100 * s['dist_excess_random_constant']:+.1f}% "
                     f"| {100 * s['frac_random_constant_draws_beating_uniform_average']:.1f}% |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n-const", type=int, default=100, help="fixed random weights per seed")
    ap.add_argument("--n-sched", type=int, default=20, help="per-step random schedules per seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--tag", default="random_weights")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    raw, summary = {}, {}
    with Pool(a.workers) as pool:
        for name in a.configs:
            t0 = time.time()
            rows = pool.map(run_seed, [(name, s, a.n_const, a.n_sched) for s in range(a.seeds)])
            raw[name], summary[name] = rows, summarize(rows)
            print(f"{name}: {time.time() - t0:.0f}s  " + "  ".join(
                f"{k}={summary[name]['F_' + k]['mean']:.4f}" for k in METHODS), flush=True)

    meta = dict(n_seeds=a.seeds, n_const=a.n_const, n_sched=a.n_sched, c0=C0, c1=C1,
                configs={k: dict(zip(("kind", "m", "T"), CONFIGS[k])) for k in a.configs},
                beat_tol=BEAT_TOL)
    tolist = lambda o: o.tolist() if isinstance(o, np.ndarray) else (o if isinstance(o, int) else float(o))
    (OUT / f"{a.tag}.json").write_text(json.dumps(
        dict(meta=meta, summary=summary,
             raw={n: [{k: tolist(v) for k, v in r.items()} for r in rows] for n, rows in raw.items()})))
    md = to_markdown(summary, meta)
    (OUT / f"{a.tag}.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
