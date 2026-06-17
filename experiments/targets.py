"""Two 2D Gaussian-mixture targets: a "happy" and a "sad" smiley face.

The two faces are RELATED -- they share a circular outline, eye positions, and
brows, and differ only in the mouth (a big grin vs. a small frown).  This is the
regime the method is built for: combining two related image distributions (same
content, different attribute) with a single shared chain.  The mouth asymmetry
makes the score-difference moments asymmetric (sigma_12 != sigma_21,
b_12 != b_21), pushing the closed-form optimal weight alpha* off 1/2; the large
shared structure keeps the weighted-score (geometric-mean) blend non-degenerate
so the chain morphs smoothly between the two expressions.  See ``_face`` below.

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
# eyes (a large common structure) and differ only in EXPRESSION -- the mouth.
# This is the regime the method is built for: combining two related image
# distributions with one shared chain.
#
# The shared structure is load-bearing.  The weighted score s_w = a*grad log q_H
# + (1-a)*grad log q_S is the score of the GEOMETRIC MEAN q_H^a q_S^(1-a), which
# only has mass where BOTH targets do.  The big shared outline + eyes therefore
# guarantee heavy overlap, so the single weighted-score chain renders a coherent
# face whose expression morphs smoothly from sad (a=0) to happy (a=1) -- rather
# than collapsing, which is what happens for disjoint targets.  The two faces
# differ ONLY in the MOUTH (a big grin vs a small frown); this single, localized,
# asymmetric difference is what pushes the closed-form alpha* clearly off 1/2 (to
# ~0.65) while keeping the realized worst-case-TV optimum well predicted.  The
# outline uses many tightly-spaced components so it renders as a smooth ring.
_OUTLINE_N = 44
_EYES = np.array([[-0.85, 0.65], [0.85, 0.65]])
SHARED_SPREAD = 0.15  # per-component spread of the (densely sampled) outline


def _face(mouth_curv, mouth_spread, n_mouth, mouth_w, eye_spread=0.18,
          outline_r=2.2, spread=SHARED_SPREAD) -> GaussianMixture:
    """One smiley face as an isotropic-component GMM.

    A smooth circular outline + two eyes (both SHARED between the two faces) and a
    parabolic mouth (sign of ``mouth_curv``: + smile, - frown).  Only the mouth
    differs across the two expressions; the rest is the shared overlap anchor.
    """
    th = np.linspace(0.0, 2 * np.pi, _OUTLINE_N, endpoint=False)
    outline = np.stack([outline_r * np.cos(th), outline_r * np.sin(th)], axis=1)
    mx = np.linspace(-mouth_w, mouth_w, n_mouth)
    mouth = np.stack([mx, -0.95 + mouth_curv * (mx / mouth_w) ** 2], axis=1)
    means = np.vstack([outline, _EYES, mouth])
    variances = np.concatenate([
        np.full(_OUTLINE_N, spread ** 2),
        np.full(len(_EYES), eye_spread ** 2),
        np.full(n_mouth, mouth_spread ** 2),
    ])
    return GaussianMixture(means, variances)


def happy_face() -> GaussianMixture:
    """Big, wide grin (shared outline/eyes)."""
    return _face(mouth_curv=+0.65, mouth_spread=0.16, n_mouth=11, mouth_w=1.2)


def sad_face() -> GaussianMixture:
    """Small, narrow frown (shared outline/eyes)."""
    return _face(mouth_curv=-0.40, mouth_spread=0.16, n_mouth=7, mouth_w=0.8)


def make_targets():
    """Return the two named targets, (happy, sad).

    Objective 1 is the happy face (weight ``alpha``), objective 2 the sad face.
    """
    return happy_face(), sad_face()


# ---------------------------------------------------------------------------
# Abstract targets for the quantitative experiments
# ---------------------------------------------------------------------------
# These differ in DIFFICULTY (sharpness / multimodality), not just location, so
# the pairwise score/Jacobian gaps are large and asymmetric -- giving a deep
# worst-case-TV bowl where uniform visibly loses, and (for m>=3) a non-trivial
# interior SOCP optimum.  All are centred on the same region so the
# weighted-score (geometric-mean) blend stays non-degenerate.
def gaussian_grid(n_side: int = 4, extent: float = 2.2, spread: float = 0.13) -> GaussianMixture:
    """n_side x n_side grid of sharp Gaussians -- a hard, multimodal target."""
    xs = np.linspace(-extent, extent, n_side)
    means = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
    return GaussianMixture(means, np.full(len(means), spread ** 2))


def broad_blob(spread: float = 1.1, center=(0.0, 0.0)) -> GaussianMixture:
    """A single broad Gaussian -- an easy, smooth, low-curvature target."""
    return GaussianMixture(np.array([center], float), np.array([spread ** 2]))


def ring(r: float = 2.0, n: int = 16, spread: float = 0.16, center=(0.0, 0.0)) -> GaussianMixture:
    """A ring of Gaussians of radius r."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    means = np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th)], axis=1)
    return GaussianMixture(means, np.full(n, spread ** 2))


def make_sharp_pair():
    """Sharper m=2 instance: a SHARP vs a BROAD 16-mode grid at the SAME
    locations.  The two targets share support and mode locations and differ
    ONLY in difficulty (per-mode sharpness): objective 1 (weight alpha) is the
    sharp, high-curvature grid; objective 2 the broad, smooth one.  Sharing the
    support avoids the moment artifact a broad blob's wide tails would create
    (they sample far regions where the sharp score is erratic), so the
    closed-form alpha* stays calibrated while the bowl is deep and uniform loses.
    """
    xs = np.linspace(-2.0, 2.0, 4)
    means = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
    sharp = GaussianMixture(means, np.full(len(means), 0.12 ** 2))
    broad = GaussianMixture(means, np.full(len(means), 0.42 ** 2))
    return sharp, broad


def make_targets_m(m: int = 3):
    """A set of m related-but-different-difficulty 2D targets for the SOCP /
    algorithms experiment (Key Result I).  Centred on the same region; each is
    "hard" in a different way, so the SOCP optimum w* is interior."""
    pool = [
        gaussian_grid(4, extent=2.2, spread=0.13),   # sharp multimodal grid
        broad_blob(spread=1.2),                       # easy broad blob
        ring(r=2.1, n=16, spread=0.15),               # ring
        gaussian_grid(2, extent=1.4, spread=0.30),    # 4 medium blobs
    ]
    assert 2 <= m <= len(pool)
    return pool[:m]
