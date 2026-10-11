"""Exact quantities for Gaussian targets under the DDPM-type sampler of the paper.

Targets are p^i = N(mu_i, S_i) on R^d.  Every true score is affine in x, so a
convex combination of scores is affine too and the sampler (eq. (sampler)) maps
Gaussians to Gaussians: we propagate the mean and covariance through all T
steps exactly instead of simulating particles.  Likewise the Gram matrices
Sigma_i^(t) of score differences (eq. (gram)) are available in closed form.

Indexing follows the paper: time runs over t = 1..T and every per-step array
has length T + 1 with entry 0 unused.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
class Schedule:
    """The learning-rate schedule of eq. (schedule) (Li et al., 2023)."""

    def __init__(self, T: int, c0: float = 4.0, c1: float = 8.0):
        self.T, self.c0, self.c1 = T, c0, c1
        r = c1 * np.log(T) / T
        t = np.arange(T + 1, dtype=np.float64)
        beta = np.empty(T + 1)
        beta[0] = np.nan
        beta[1] = T ** (-c0)
        beta[2:] = r * np.minimum(beta[1] * (1.0 + r) ** t[2:], 1.0)
        self.beta = beta                      # 1 - alpha_t
        self.alpha = 1.0 - beta
        abar = np.empty(T + 1)
        abar[0] = 1.0
        abar[1:] = np.cumprod(self.alpha[1:])
        self.abar = abar
        self.sigma2 = beta / self.alpha       # sampler noise variance sigma_t^2
        self.gamma = (1.0 - abar) / abar      # inverse signal-to-noise ratio


# ---------------------------------------------------------------------------
# targets and their noised scores
# ---------------------------------------------------------------------------
class GaussianTargets:
    """m Gaussian targets N(mu_i, S_i) diffused by a common schedule.

    The score of the noised target j at step t is
        s_t^j(x) = -P[t, j] x + h[t, j],
    with P[t, j] = (abar_t S_j + (1 - abar_t) I)^{-1} and
    h[t, j] = sqrt(abar_t) P[t, j] mu_j.
    """

    def __init__(self, mus, covs, sched: Schedule):
        self.mus = np.asarray(mus, dtype=np.float64)       # (m, d)
        self.covs = np.asarray(covs, dtype=np.float64)     # (m, d, d)
        self.m, self.d = self.mus.shape
        self.sched = sched
        abar = sched.abar[:, None, None, None]
        eye = np.eye(self.d)
        self.Sig = abar * self.covs[None] + (1.0 - abar) * eye   # (T+1, m, d, d)
        self.P = np.linalg.inv(self.Sig)
        self.mean = np.sqrt(sched.abar)[:, None, None] * self.mus[None]  # (T+1, m, d)
        self.h = np.einsum("tjde,tje->tjd", self.P, self.mean)

    # -- Gram matrices of score differences ---------------------------------
    def gram_features(self):
        """Factors of the Gram matrices: Sigma_i^(t) = F[t, i]^T F[t, i].

        Under X ~ q_t^i write X = mean_i + L_i xi with xi ~ N(0, I).  Then
        D_ij(X, t) = s_t^j(X) - s_t^i(X) = c_ij + B_ij L_i xi is affine in xi,
        so E[D_ij . D_ik] = c_ij . c_ik + <B_ij L_i, B_ik L_i>_F.  Column j of
        F[t, i] stacks c_ij and vec(B_ij L_i).  Shape (T+1, m, d + d^2, m).
        """
        T, m, d = self.sched.T, self.m, self.d
        L = np.linalg.cholesky(self.Sig)                              # (T+1, m, d, d)
        B = -(self.P[:, None, :, :, :] - self.P[:, :, None, :, :])    # (T+1, i, j, d, d)
        c = (np.einsum("tijde,tie->tijd", B, self.mean)
             + self.h[:, None, :, :] - self.h[:, :, None, :])         # (T+1, i, j, d)
        BL = np.einsum("tijde,tief->tijdf", B, L)
        F = np.concatenate([c, BL.reshape(T + 1, m, m, d * d)], axis=-1)  # (T+1, i, j, k)
        F = np.swapaxes(F, 2, 3)                                      # (T+1, i, k, j)
        F[0] = 0.0                      # t = 0 is not a step
        return F

    def V(self, W, F=None):
        """V_i(w) = 1/2 sum_{t>=2} (1 - alpha_t) w_t^T Sigma_i^(t) w_t  (eq. (Vi)).

        W is a batch of schedules (B, T+1, m) or of constant weights (B, m).
        Returns (B, m).
        """
        F = self.gram_features() if F is None else F
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 2:
            W = np.broadcast_to(W[:, None, :], (W.shape[0], self.sched.T + 1, self.m))
        z = np.einsum("tikj,btj->btik", F[2:], W[:, 2:])
        return 0.5 * np.einsum("t,btik,btik->bi", self.sched.beta[2:], z, z)

    # -- exact sampler ------------------------------------------------------
    def sample_moments(self, W):
        """Mean and covariance of Y_1 for a batch of weight schedules.

        W: (B, T+1, m) schedules or (B, m) constant weights.
        Returns mean (B, d) and covariance (B, d, d) of p_1^w.
        """
        s, d = self.sched, self.d
        W = np.asarray(W, dtype=np.float64)
        const = W.ndim == 2
        Bn = W.shape[0]
        eye = np.eye(d)
        mean = np.zeros((Bn, d))
        cov = np.broadcast_to(eye, (Bn, d, d)).copy()
        for t in range(s.T, 1, -1):
            w = W if const else W[:, t]
            A = np.einsum("bj,jde->bde", w, self.P[t])
            b = w @ self.h[t]
            K = (eye - s.beta[t] * A) / np.sqrt(s.alpha[t])
            mean = np.einsum("bde,be->bd", K, mean) + s.beta[t] / np.sqrt(s.alpha[t]) * b
            cov = K @ cov @ np.swapaxes(K, 1, 2) + s.sigma2[t] * eye
        return mean, cov

    def target_moments(self):
        """Mean and covariance of q_1^i, the laws the output is compared with."""
        return self.mean[1], self.Sig[1]


# ---------------------------------------------------------------------------
# total variation between Gaussians in d = 2
# ---------------------------------------------------------------------------
_HALF_WIDTH = 8.5
_N_GRID = 341


def _one_sided(m1, C1, m2, C2):
    """int (p - q)_+ for p = N(m1, C1), q = N(m2, C2), on a grid whitened by p."""
    u = np.linspace(-_HALF_WIDTH, _HALF_WIDTH, _N_GRID)
    hstep = u[1] - u[0]
    U = np.stack(np.meshgrid(u, u, indexing="ij"), axis=-1).reshape(-1, 2)   # (G, 2)
    logphi = -0.5 * np.sum(U * U, axis=1) - np.log(2.0 * np.pi)
    L1 = np.linalg.cholesky(C1)                                               # (B, 2, 2)
    X = m1[:, None, :] + np.einsum("bde,ge->bgd", L1, U)                      # (B, G, 2)
    P2 = np.linalg.inv(C2)
    diff = X - m2[:, None, :]
    quad = np.einsum("bgd,bde,bge->bg", diff, P2, diff)
    # density of q pulled back to the whitened coordinates of p
    logq = (-0.5 * quad - np.log(2.0 * np.pi) - 0.5 * np.linalg.slogdet(C2)[1][:, None]
            + np.linalg.slogdet(L1)[1][:, None])
    return np.sum(np.clip(np.exp(logphi)[None] - np.exp(logq), 0.0, None), axis=1) * hstep ** 2


def tv_gauss2d(m1, C1, m2, C2, chunk: int = 64, return_err: bool = False):
    """TV(N(m1, C1), N(m2, C2)) for batches of 2D Gaussians, by quadrature.

    TV = int (p - q)_+ = int (q - p)_+; we evaluate both (each on the grid
    adapted to the Gaussian that dominates the integrand) and average.  The gap
    between the two is only a consistency diagnostic, not an error bound: it
    vanishes by symmetry for equal covariances and misses errors common to both
    sides.  Against a fine reference quadrature the error is below 5e-5 for the
    standard-deviation ratios used in the experiments (up to about 6); it grows
    to about 1e-4 at ratios of 10-20 and the grid is inadequate near 100.
    """
    m1, C1, m2, C2 = (np.asarray(a, dtype=np.float64) for a in (m1, C1, m2, C2))
    n = m1.shape[0]
    a, b = np.empty(n), np.empty(n)
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        a[sl] = _one_sided(m1[sl], C1[sl], m2[sl], C2[sl])
        b[sl] = _one_sided(m2[sl], C2[sl], m1[sl], C1[sl])
    tv = 0.5 * (a + b)
    return (tv, np.abs(a - b)) if return_err else tv


def worst_case_tv(targets: GaussianTargets, W, return_err: bool = False):
    """F_T(w) = max_i TV(q_1^i, p_1^w) for a batch of weights (eq. (FT)).

    Returns F (B,) and the per-target TVs (B, m).
    """
    mean, cov = targets.sample_moments(W)
    tm, tc = targets.target_moments()
    Bn, m = mean.shape[0], targets.m
    out = tv_gauss2d(
        np.repeat(tm[None], Bn, axis=0).reshape(Bn * m, -1),
        np.repeat(tc[None], Bn, axis=0).reshape(Bn * m, 2, 2),
        np.repeat(mean[:, None], m, axis=1).reshape(Bn * m, -1),
        np.repeat(cov[:, None], m, axis=1).reshape(Bn * m, 2, 2),
        return_err=return_err,
    )
    if return_err:
        tv, err = out
        return tv.reshape(Bn, m).max(axis=1), tv.reshape(Bn, m), float(err.max())
    tv = out.reshape(Bn, m)
    return tv.max(axis=1), tv
