"""Binned total-variation estimation in 2D.

TV is honestly estimable in 2D: bin the bounding box, evaluate the *known*
target marginal q_1^i analytically on the bin centers, histogram the chain's
samples for p_1^w, normalize both to sum to 1 over the box, and report

    TV = 1/2 sum_bins | p_hat - q | .

The target side has no estimation error (q_1^i is a GMM evaluated exactly);
only p_1^w is a histogram, so sample count controls the bias.
"""

from __future__ import annotations

import numpy as np


class TVGrid:
    """Fixed bounding-box grid for TV estimation."""

    def __init__(self, box=(-4.0, 4.0), n_bins: int = 200):
        self.lo, self.hi = box
        self.n_bins = n_bins
        self.edges = np.linspace(self.lo, self.hi, n_bins + 1)
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        gx, gy = np.meshgrid(centers, centers, indexing="ij")
        self.centers = np.stack([gx.ravel(), gy.ravel()], axis=1)   # (n_bins^2, 2)
        self.cell_area = (self.edges[1] - self.edges[0]) ** 2

    def target_pmf(self, marginal):
        """Probability mass per bin for an analytic GMM marginal, summing to 1."""
        dens = marginal.pdf(self.centers)            # density at centers
        mass = dens * self.cell_area
        return mass / mass.sum()

    def sample_pmf(self, samples):
        """Histogram pmf of samples over the box, summing to 1.

        Samples outside the box are dropped; with a box that covers the support
        this loses negligible mass.
        """
        H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1],
                                 bins=[self.edges, self.edges])
        H = H.ravel()
        total = H.sum()
        if total == 0:
            return H
        return H / total

    def tv(self, marginal, samples):
        q = self.target_pmf(marginal)
        p = self.sample_pmf(samples)
        return 0.5 * np.sum(np.abs(p - q))


def worst_case_tv(grid: TVGrid, marg1, marg2, samples):
    """max_i TV(q_1^i, p_1^w) for a single sample set."""
    tv1 = grid.tv(marg1, samples)
    tv2 = grid.tv(marg2, samples)
    return max(tv1, tv2), tv1, tv2


def worst_case_tv_multi(grid: TVGrid, marginals, samples):
    """(max_i TV, [TV_i]) for m target marginals against one sample set.

    ``marginals`` is a list of analytic GMM marginals (one per objective); the
    sample pmf is computed once and reused.
    """
    p = grid.sample_pmf(samples)
    tvs = [0.5 * np.sum(np.abs(p - grid.target_pmf(m))) for m in marginals]
    return max(tvs), tvs
