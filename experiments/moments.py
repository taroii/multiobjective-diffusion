"""Monte-Carlo estimation of the four distribution-level moments and alpha*.

For a pair of objectives (i, j) the per-pair score and Jacobian gaps are

    sigma_ij^2 = (1/T) sum_t  E_{X ~ q_t^i} || s_t^j(X) - s_t^i(X) ||_2^2 ,
    b_ij       = (1/T) sum_t  E_{X ~ q_t^i} || J_{s_t^j}(X) - J_{s_t^i}(X) ||_2 ,

where ||.|| on the Jacobian difference is the spectral (operator-2) norm.  We
estimate the time average over a sub-grid of noise levels and the inner
expectation by sampling X ~ q_t^i (draw x_0 ~ pdata^i, then forward-noise).

Then the closed-form two-objective weight is

    alpha* = (sigma_12 + kappa b_12) / ((sigma_12 + sigma_21) + kappa (b_12 + b_21)).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from diffusion import AnalyticScore, VPSchedule


def _spectral_norm_batch(M):
    """Spectral (operator-2) norm of each 2x2 matrix in a batch M: (N,2,2) -> (N,).

    The spectral norm is the largest singular value.  We use the SVD rather than
    |eigenvalue| because for *learned* scores the Jacobian need not be symmetric
    (only the analytic Hessian-of-log-density is); the SVD is correct for both.
    """
    return np.linalg.svd(M, compute_uv=False).max(axis=1)


@dataclass
class PairMoment:
    sigma2: float       # sigma_ij^2  (mean of per-t estimates)
    sigma2_se: float    # standard error of sigma_ij^2 across the t-grid
    b: float            # b_ij
    b_se: float

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.sigma2))


def estimate_pair(score_i: AnalyticScore, score_j: AnalyticScore,
                  target_i, schedule: VPSchedule,
                  n_mc: int = 4000, n_tgrid: int = 80,
                  rng: np.random.Generator | None = None) -> PairMoment:
    """Estimate sigma_ij^2 and b_ij (expectations under objective i's marginals)."""
    if rng is None:
        rng = np.random.default_rng(0)
    # Sub-grid of diffusion steps spanning [1, T].
    t_grid = np.unique(np.linspace(1, schedule.T, n_tgrid).astype(int))

    sig_per_t = np.empty(len(t_grid))
    b_per_t = np.empty(len(t_grid))
    sqrt_abar = np.sqrt(schedule.abar)
    sqrt_1m = np.sqrt(1.0 - schedule.abar)

    for idx, t in enumerate(t_grid):
        # X ~ q_t^i : draw x0 ~ pdata^i then forward-noise to level t.
        x0 = target_i.sample(n_mc, rng)
        eps = rng.standard_normal((n_mc, 2))
        x = sqrt_abar[t - 1] * x0 + sqrt_1m[t - 1] * eps

        ds = score_j.score(x, t) - score_i.score(x, t)          # (N, 2)
        sig_per_t[idx] = np.mean(np.sum(ds * ds, axis=1))

        dJ = score_j.jacobian(x, t) - score_i.jacobian(x, t)    # (N, 2, 2)
        b_per_t[idx] = np.mean(_spectral_norm_batch(dJ))

    return PairMoment(
        sigma2=float(np.mean(sig_per_t)),
        sigma2_se=float(np.std(sig_per_t, ddof=1) / np.sqrt(len(t_grid))),
        b=float(np.mean(b_per_t)),
        b_se=float(np.std(b_per_t, ddof=1) / np.sqrt(len(t_grid))),
    )


@dataclass
class Moments:
    sigma12: float
    sigma21: float
    b12: float
    b21: float
    kappa: float
    m12: PairMoment
    m21: PairMoment

    def alpha_star(self) -> float:
        k = self.kappa
        num = self.sigma12 + k * self.b12
        den = (self.sigma12 + self.sigma21) + k * (self.b12 + self.b21)
        return float(num / den)

    def alpha_sigma_only(self) -> float:
        """kappa -> 0 limit: weight by score difficulty only."""
        return float(self.sigma12 / (self.sigma12 + self.sigma21))

    def alpha_b_only(self) -> float:
        """kappa -> infinity limit: weight by Jacobian difficulty only."""
        return float(self.b12 / (self.b12 + self.b21))

    def summary(self) -> dict:
        d = {
            "sigma12": self.sigma12, "sigma21": self.sigma21,
            "b12": self.b12, "b21": self.b21,
            "kappa": self.kappa,
            "alpha_star": self.alpha_star(),
            "alpha_sigma_only": self.alpha_sigma_only(),
            "alpha_b_only": self.alpha_b_only(),
            "m12_se": {"sigma2_se": self.m12.sigma2_se, "b_se": self.m12.b_se},
            "m21_se": {"sigma2_se": self.m21.sigma2_se, "b_se": self.m21.b_se},
        }
        return d


def estimate_moments(score1: AnalyticScore, score2: AnalyticScore,
                     target1, target2, schedule: VPSchedule,
                     n_mc: int = 4000, n_tgrid: int = 80, seed: int = 0) -> Moments:
    """Estimate the full set of two-objective moments and alpha*."""
    rng = np.random.default_rng(seed)
    # sigma_12, b_12 : objective 1's marginals, gap to objective 2's score.
    m12 = estimate_pair(score1, score2, target1, schedule, n_mc, n_tgrid, rng)
    # sigma_21, b_21 : objective 2's marginals, gap to objective 1's score.
    m21 = estimate_pair(score2, score1, target2, schedule, n_mc, n_tgrid, rng)
    return Moments(
        sigma12=m12.sigma, sigma21=m21.sigma,
        b12=m12.b, b21=m21.b,
        kappa=schedule.kappa(),
        m12=m12, m21=m21,
    )
