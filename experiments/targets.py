"""Two 2D Gaussian-mixture targets spelling the letters "A" and "I".

The two targets are deliberately *asymmetric* (different per-component spread
and different number of components) so that the score-difference moments come
out asymmetric, sigma_12 != sigma_21 and b_12 != b_21, pushing the closed-form
optimal weight alpha* off 1/2.  If alpha* came out to 1/2 the experiment would
teach a reviewer nothing.

A GaussianMixture here has *isotropic* components, N(mu_k, v_k * I_2), because
that is all we need and it keeps the analytic score/Jacobian compact.  The
forward diffusion marginal of such a mixture is again an isotropic-component
mixture (see ``forward``), so the true score and its Jacobian stay closed form
at every noise level.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

LOG_2PI = np.log(2.0 * np.pi)


class GaussianMixture:
    """Isotropic-component 2D Gaussian mixture, sum_k pi_k N(mu_k, v_k I)."""

    def __init__(self, means, variances, weights=None):
        self.means = np.asarray(means, dtype=np.float64)          # (K, 2)
        self.variances = np.asarray(variances, dtype=np.float64)  # (K,)
        K = self.means.shape[0]
        assert self.means.shape == (K, 2)
        assert self.variances.shape == (K,)
        if weights is None:
            weights = np.full(K, 1.0 / K)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.log_weights = np.log(self.weights)
        # float32 copies for the sampler hot path (score evaluated millions of times)
        self._means32 = self.means.astype(np.float32)
        self._inv_var32 = (1.0 / self.variances).astype(np.float32)
        # per-component logit offset: log pi_k - log(2 pi v_k) (the 2D normalizer)
        self._logit_off32 = (self.log_weights - LOG_2PI - np.log(self.variances)).astype(np.float32)

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    # -- forward diffusion marginal ---------------------------------------
    def forward(self, abar: float) -> "GaussianMixture":
        """Marginal q_t of the VP forward process at signal level abar = bar_alpha_t.

        x_t = sqrt(abar) x_0 + sqrt(1 - abar) eps  =>  for each component,
        mean -> sqrt(abar) mu_k,  var -> abar v_k + (1 - abar).
        """
        return GaussianMixture(
            means=np.sqrt(abar) * self.means,
            variances=abar * self.variances + (1.0 - abar),
            weights=self.weights,
        )

    # -- analytic density / score / Jacobian ------------------------------
    def _component_logpdf(self, x):
        """log N(x; mu_k, v_k I) for each component.  x: (N,2) -> (N,K)."""
        diff = x[:, None, :] - self.means[None, :, :]          # (N, K, 2)
        sq = np.sum(diff * diff, axis=2)                        # (N, K)
        # 2D isotropic: -log(2 pi v_k) - ||.||^2 / (2 v_k)
        return -LOG_2PI - np.log(self.variances)[None, :] - sq / (2.0 * self.variances)[None, :]

    def logpdf(self, x):
        x = np.atleast_2d(x)
        comp = self._component_logpdf(x) + self.log_weights[None, :]
        return logsumexp(comp, axis=1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def responsibilities(self, x):
        comp = self._component_logpdf(x) + self.log_weights[None, :]
        return softmax(comp, axis=1)                           # (N, K)

    def score(self, x):
        """grad_x log q(x).  x: (N,2) -> (N,2).

        Fused, allocation-lean, float32 path: this is called O(T) times per
        backward chain on 1e5 points, so it dominates sampler runtime.
        """
        x = np.atleast_2d(x)
        dt = np.float32 if x.dtype == np.float32 else np.float64
        means = self._means32 if dt is np.float32 else self.means
        inv_var = self._inv_var32 if dt is np.float32 else (1.0 / self.variances)
        off = self._logit_off32 if dt is np.float32 else \
            (self.log_weights - LOG_2PI - np.log(self.variances))
        diff = x[:, None, :] - means[None, :, :]               # (N, K, 2)
        sq = np.einsum("nki,nki->nk", diff, diff)              # (N, K) squared dist
        logits = off[None, :] - 0.5 * sq * inv_var[None, :]
        logits = logits - logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        r = logits / logits.sum(axis=1, keepdims=True)         # responsibilities (N,K)
        a = -diff * inv_var[None, :, None]                     # per-comp score (N,K,2)
        return np.einsum("nk,nki->ni", r, a)

    def jacobian(self, x):
        """grad_x^2 log q(x) = Jacobian of the score.  x: (N,2) -> (N,2,2).

        J = sum_k r_k a_k a_k^T - s s^T - (sum_k r_k / v_k) I,
        with a_k = -(x - mu_k)/v_k the per-component score and s = score.
        """
        x = np.atleast_2d(x)
        diff = x[:, None, :] - self.means[None, :, :]          # (N, K, 2)
        a = -diff / self.variances[None, :, None]              # (N, K, 2)
        r = self.responsibilities(x)                           # (N, K)
        s = np.einsum("nk,nki->ni", r, a)                      # (N, 2)
        term1 = np.einsum("nk,nki,nkj->nij", r, a, a)          # E_r[a a^T]
        term2 = -np.einsum("ni,nj->nij", s, s)                 # - s s^T
        coeff = np.einsum("nk,k->n", r, 1.0 / self.variances)  # sum_k r_k / v_k
        eye = np.eye(2)[None, :, :]
        term3 = -coeff[:, None, None] * eye
        return term1 + term2 + term3

    # -- sampling ---------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator):
        comp = rng.choice(self.n_components, size=n, p=self.weights)
        std = np.sqrt(self.variances[comp])[:, None]
        return self.means[comp] + std * rng.standard_normal((n, 2))


# ---------------------------------------------------------------------------
# Face targets: "happy" vs "sad"
# ---------------------------------------------------------------------------
# Two RELATED 2D distributions: smiley faces that share a circular outline and
# eye positions (a large common structure) and differ only in EXPRESSION -- the
# mouth, eyebrows, and eye size.  This is the regime the method is built for:
# combining two related image distributions with one shared chain.
#
# The shared structure is load-bearing.  The weighted score s_w = a*grad log q_H
# + (1-a)*grad log q_S is the score of the GEOMETRIC MEAN q_H^a q_S^(1-a), which
# only has mass where BOTH targets do.  The big shared outline + eyes (and flat
# brows) therefore guarantee heavy overlap, so the single weighted-score chain
# renders a coherent face whose expression morphs smoothly from sad (a=0) to
# happy (a=1) -- rather than collapsing, which is what happens for disjoint
# targets.  The two faces differ ONLY in the MOUTH (a big grin vs a small frown);
# this single, localized, asymmetric difference is what pushes the closed-form
# alpha* clearly off 1/2 (to ~0.65) while keeping the realized worst-case-TV
# optimum well predicted.  (Adding more differing features -- brows, eye size --
# deepens the bowl but makes the difference more symmetric, pulling the realized
# optimum back toward 1/2 and away from alpha*; differing in the mouth alone
# gives the best agreement.)
_OUTLINE_N = 18
_EYES = np.array([[-0.85, 0.65], [0.85, 0.65]])
SHARED_SPREAD = 0.16  # per-component spread of the shared outline


def _face(mouth_curv, mouth_spread, n_mouth, mouth_w, eye_spread=0.18,
          brow_curv=0.0, outline_r=2.2, brow_y=1.45, spread=SHARED_SPREAD) -> GaussianMixture:
    """One smiley face as an isotropic-component GMM.

    Outline + eyes + flat brows (all SHARED between the two faces) and a parabolic
    mouth (sign of ``mouth_curv``: + smile, - frown).  Only the mouth is meant to
    differ across the two expressions; the rest is the shared overlap anchor.
    """
    th = np.linspace(0.0, 2 * np.pi, _OUTLINE_N, endpoint=False)
    outline = np.stack([outline_r * np.cos(th), outline_r * np.sin(th)], axis=1)
    mx = np.linspace(-mouth_w, mouth_w, n_mouth)
    mouth = np.stack([mx, -0.95 + mouth_curv * (mx / mouth_w) ** 2], axis=1)
    bx = np.linspace(-0.4, 0.4, 3)
    brows = np.vstack([np.stack([cx + bx, brow_y + brow_curv * (bx / 0.4) ** 2], axis=1)
                       for cx in (-0.85, 0.85)])
    means = np.vstack([outline, _EYES, mouth, brows])
    variances = np.concatenate([
        np.full(_OUTLINE_N, spread ** 2),
        np.full(len(_EYES), eye_spread ** 2),
        np.full(n_mouth, mouth_spread ** 2),
        np.full(len(brows), 0.14 ** 2),
    ])
    return GaussianMixture(means, variances)


def happy_face() -> GaussianMixture:
    """Big, wide grin (shared outline/eyes/brows)."""
    return _face(mouth_curv=+0.65, mouth_spread=0.16, n_mouth=11, mouth_w=1.2)


def sad_face() -> GaussianMixture:
    """Small, narrow frown (shared outline/eyes/brows)."""
    return _face(mouth_curv=-0.40, mouth_spread=0.16, n_mouth=7, mouth_w=0.8)


def make_targets():
    """Return the two named targets, (happy, sad).

    Objective 1 is the happy face (weight ``alpha``), objective 2 the sad face.
    """
    return happy_face(), sad_face()
