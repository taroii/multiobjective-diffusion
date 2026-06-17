"""VP / DDPM forward schedule shared across both objectives.

A single schedule object holds the standard DDPM quantities (beta_t, alpha_t,
bar_alpha_t) on a grid t = 1..T, and exposes the analytic true score
s_t^{i,*}(x) = grad log q_t^i(x) and its Jacobian for any GaussianMixture
target -- both closed form because the forward marginal of a GMM is a GMM
(``GaussianMixture.forward``).

The schedule is *shared* across objectives, matching Assumption "shared forward
schedule" in the paper.
"""

from __future__ import annotations

import numpy as np

from targets import GaussianMixture


class VPSchedule:
    """Standard DDPM variance-preserving schedule with linear beta.

    Indexing convention: arrays are length T with index k in [0, T-1] mapping
    to diffusion step t = k + 1.  ``abar[k]`` = bar_alpha_t, ``alpha[k]`` =
    alpha_t = 1 - beta_t.  Step t=1 is the smallest noise level (target time),
    step t=T is the terminal (~N(0, I)).
    """

    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        self.T = int(T)
        self.beta = np.linspace(beta_min, beta_max, self.T)      # beta_t, t=1..T
        self.alpha = 1.0 - self.beta                             # alpha_t
        self.abar = np.cumprod(self.alpha)                       # bar_alpha_t

    def abar_at(self, t: int) -> float:
        """bar_alpha_t for diffusion step t in [1, T]."""
        return float(self.abar[t - 1])

    @property
    def d(self) -> int:
        return 2

    def kappa(self) -> float:
        """kappa = sqrt(d / log T), the dimensional ratio in alpha*.

        Uses the same T as the sampler for internal consistency (Sec. 4 of the
        plan).
        """
        return float(np.sqrt(self.d / np.log(self.T)))


class AnalyticScore:
    """True score / Jacobian of a target at any noise level of a schedule.

    Caches the forward-marginal GMM per diffusion step so repeated calls at the
    same step are cheap.
    """

    def __init__(self, target: GaussianMixture, schedule: VPSchedule):
        self.target = target
        self.schedule = schedule
        self._cache: dict[int, GaussianMixture] = {}

    def marginal(self, t: int) -> GaussianMixture:
        gm = self._cache.get(t)
        if gm is None:
            gm = self.target.forward(self.schedule.abar_at(t))
            self._cache[t] = gm
        return gm

    def score(self, x, t: int):
        return self.marginal(t).score(x)

    def jacobian(self, x, t: int):
        return self.marginal(t).jacobian(x)


def weighted_score(scores, alpha: float, x, t: int):
    """s_w(x, t) = alpha s^1 + (1 - alpha) s^2 for a pair of AnalyticScores."""
    s1, s2 = scores
    return alpha * s1.score(x, t) + (1.0 - alpha) * s2.score(x, t)
