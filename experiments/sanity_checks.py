"""Checks of the exact Gaussian machinery and the minimax solvers.

Run:  python sanity_checks.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from gaussian import GaussianTargets, Schedule, tv_gauss2d, worst_case_tv
from minimax import (m2_constant, m2_schedule, minimax_constant, minimax_schedule,
                     uniform_average_constant)
from run_random_weights import hetero_instance, paper_instance


def check(name, value, ok):
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}")
    assert ok, name


def main():
    rng = np.random.default_rng(0)

    # 1. TV quadrature vs. the closed form for a common covariance.
    C = np.array([[3.0, 1.0], [1.0, 2.0]])
    deltas = rng.normal(size=(50, 2)) * 2.0
    maha = np.sqrt(np.einsum("bd,de,be->b", deltas, np.linalg.inv(C), deltas))
    exact = 2.0 * norm.cdf(0.5 * maha) - 1.0
    tv = tv_gauss2d(np.zeros((50, 2)), np.repeat(C[None], 50, 0), deltas, np.repeat(C[None], 50, 0))
    check("TV quadrature vs closed form (max abs err)", f"{np.abs(tv - exact).max():.2e}",
          np.abs(tv - exact).max() < 2e-4)

    # 2. TV quadrature vs. Monte Carlo for unequal, badly matched covariances.
    m1, C1 = np.array([0.3, -0.2]), np.array([[20.0, 3.0], [3.0, 1.0]])
    m2, C2 = np.array([1.0, 0.5]), np.array([[0.4, -0.1], [-0.1, 6.0]])
    tv, err = tv_gauss2d(m1[None], C1[None], m2[None], C2[None], return_err=True)
    x = rng.multivariate_normal(m1, C1, size=2_000_000)

    def logpdf(x, m, C):
        d = x - m
        return -0.5 * np.einsum("nd,de,ne->n", d, np.linalg.inv(C), d) - 0.5 * np.linalg.slogdet(C)[1]

    mc = np.mean(np.clip(1.0 - np.exp(logpdf(x, m2, C2) - logpdf(x, m1, C1)), 0.0, None))
    check("TV quadrature vs Monte Carlo", f"{tv[0]:.5f} vs {mc:.5f} (two-sided gap {err[0]:.1e})",
          abs(tv[0] - mc) < 2e-3 and err[0] < 1e-3)

    # 3. Paper instance: closed-form Gram quantities and known TV values.
    tg = paper_instance(T=3000)
    s = tg.sched
    F = tg.gram_features()
    w = rng.dirichlet(np.ones(3), size=4)
    S0 = tg.covs[0]
    MT = 0.5 * sum((s.gamma[t] - s.gamma[t - 1])
                   * np.linalg.matrix_power(np.linalg.inv(S0 + s.gamma[t] * np.eye(2)), 2)
                   for t in range(2, s.T + 1))
    dm = (w @ tg.mus)[:, None, :] - tg.mus[None]
    V_cor = np.einsum("bid,de,bie->bi", dm, MT, dm)        # Corollary (gauss)(b)
    check("V_i(w) vs Corollary (gauss)(b) (max rel err)",
          f"{np.abs(tg.V(w, F) / V_cor - 1).max():.2e}", np.allclose(tg.V(w, F), V_cor, rtol=1e-9))
    check("M_T vs Sigma_0^{-1}/2 (rel Frobenius err)",
          f"{np.linalg.norm(MT - 0.5 * np.linalg.inv(S0)) / np.linalg.norm(0.5 * np.linalg.inv(S0)):.2e}",
          np.linalg.norm(MT - 0.5 * np.linalg.inv(S0)) / np.linalg.norm(0.5 * np.linalg.inv(S0)) < 2e-2)

    Fw, tvs, err = worst_case_tv(tg, np.array([[1 / 3, 1 / 3, 1 / 3], [0, .5, .5], [1., 0, 0]]),
                                 return_err=True)
    r = np.sqrt((tg.mus[0] @ np.linalg.inv(S0) @ tg.mus[0]))
    check("F_T at [1/3,1/3,1/3] vs ideal 2Phi(r/2)-1", f"{Fw[0]:.4f} vs {2 * norm.cdf(r / 2) - 1:.4f}",
          abs(Fw[0] - (2 * norm.cdf(r / 2) - 1)) < 2e-3)
    check("F_T at [0,1/2,1/2] vs ideal 2Phi(3r/4)-1", f"{Fw[1]:.4f} vs {2 * norm.cdf(.75 * r) - 1:.4f}",
          abs(Fw[1] - (2 * norm.cdf(.75 * r) - 1)) < 2e-3)
    check("discretization floor TV(q_1^1, p_1^{e_1})", f"{tvs[2, 0]:.2e}", tvs[2, 0] < 5e-3)
    check("quadrature two-sided gap", f"{err:.1e}", err < 1e-3)

    Wc, vc = minimax_constant(F, s.beta)
    Ws, vs = minimax_schedule(F, s.beta)
    check("minimax constant weight on the paper instance", np.round(Wc, 4),
          np.abs(Wc - 1 / 3).max() < 1e-3)
    check("schedule value == constant value (Corollary (gauss)(d))", f"{vs:.6f} vs {vc:.6f}",
          abs(vs / vc - 1) < 1e-4)
    w_ua, _ = uniform_average_constant(F)
    Mavg = sum(np.linalg.matrix_power(np.linalg.inv(S0 + s.gamma[t] * np.eye(2)), 2) / s.abar[t]
               for t in range(1, s.T + 1)) / s.T
    check("uniform-average rule on the paper instance", f"{np.round(w_ua, 4)}, cond(M_avg) = {np.linalg.cond(Mavg):.0f}",
          np.abs(w_ua - np.array([0, .5, .5])).max() < 1e-3)
    check("solver value == V evaluated at the returned schedule",
          f"{vs:.6f} vs {tg.V(Ws[None], F).max():.6f}", abs(tg.V(Ws[None], F).max() / vs - 1) < 1e-4)

    # 4. m = 2, unequal covariances: cone programs vs. Proposition (m2).
    tg = hetero_instance(np.random.default_rng(1), m=2, T=1000)
    s, F = tg.sched, tg.gram_features()
    W_cf, v_cf = m2_schedule(F, s.beta)
    W_cp, v_cp = minimax_schedule(F, s.beta)
    check("m=2 schedule: SOCP value vs closed form", f"{v_cp:.6f} vs {v_cf:.6f}",
          abs(v_cp / v_cf - 1) < 1e-4)
    Fa, _ = worst_case_tv(tg, np.stack([W_cf, W_cp]))
    check("m=2 schedule: F_T of SOCP vs closed-form schedule", f"{Fa[1]:.5f} vs {Fa[0]:.5f}",
          abs(Fa[1] - Fa[0]) < 1e-3)
    w_cf, vb_cf = m2_constant(F, s.beta)
    w_cp, vb_cp = minimax_constant(F, s.beta)
    check("m=2 constant: SOCP vs eq. (m2)", f"{np.round(w_cp, 5)} vs {np.round(w_cf, 5)}",
          np.abs(w_cp - w_cf).max() < 1e-4 and abs(vb_cp / vb_cf - 1) < 1e-4)
    check("schedule value <= constant value", f"{v_cf:.6f} <= {vb_cf:.6f}", v_cf <= vb_cf * (1 + 1e-9))

    print("all checks passed")


if __name__ == "__main__":
    main()
