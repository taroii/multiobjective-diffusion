"""Deterministic probability-flow-ODE sampler with a weighted score.

Implements the deterministic backward update of Li et al. (2023), the sampler
the paper invokes:

    Y_T ~ N(0, I_2),
    Y_{t-1} = (1 / sqrt(alpha_t)) ( Y_t + ((1 - alpha_t) / 2) s_w(Y_t, t) ),

integrated backward t : T -> 1, plugging in the weighted score
s_w = alpha s^1 + (1 - alpha) s^2.  The chain returns the time-1 samples Y_1,
which are compared against the time-1 forward marginals q_1^i.
"""

from __future__ import annotations

import numpy as np

from diffusion import AnalyticScore, VPSchedule


def sample_weighted(scores, alpha: float, schedule: VPSchedule,
                    n_samples: int, rng: np.random.Generator,
                    batch_size: int = 50000):
    """Draw n_samples from the weighted-score backward chain at time 1.

    ``scores`` is the pair (AnalyticScore for obj 1, AnalyticScore for obj 2).
    Runs in batches to keep memory bounded for n_samples >= 1e5.
    """
    s1, s2 = scores
    inv_sqrt_alpha = (1.0 / np.sqrt(schedule.alpha)).astype(np.float32)
    half_one_minus = (0.5 * (1.0 - schedule.alpha)).astype(np.float32)
    a1 = np.float32(alpha)
    a2 = np.float32(1.0 - alpha)

    out = np.empty((n_samples, 2), dtype=np.float32)
    done = 0
    while done < n_samples:
        nb = min(batch_size, n_samples - done)
        y = rng.standard_normal((nb, 2)).astype(np.float32)    # Y_T ~ N(0, I)
        # backward t = T, T-1, ..., 2  (index k = t-1 from T-1 down to 1)
        for t in range(schedule.T, 1, -1):
            k = t - 1
            sw = a1 * s1.score(y, t) + a2 * s2.score(y, t)
            y = inv_sqrt_alpha[k] * (y + half_one_minus[k] * sw)
        out[done:done + nb] = y
        done += nb
    return out.astype(np.float64)


def sample_weighted_multi(scores, weights, schedule: VPSchedule,
                          n_samples: int, rng: np.random.Generator,
                          batch_size: int = 50000):
    """Draw n_samples from the weighted-score chain for m objectives.

    ``scores`` is a list of ``m`` AnalyticScore (or LearnedScore) objects and
    ``weights`` a length-``m`` simplex vector; the shared score is
    s_w = sum_i w_i s^i.  Generalizes ``sample_weighted`` (the m=2 case).
    """
    w = np.asarray(weights, dtype=np.float32)
    inv_sqrt_alpha = (1.0 / np.sqrt(schedule.alpha)).astype(np.float32)
    half_one_minus = (0.5 * (1.0 - schedule.alpha)).astype(np.float32)

    out = np.empty((n_samples, 2), dtype=np.float32)
    done = 0
    while done < n_samples:
        nb = min(batch_size, n_samples - done)
        y = rng.standard_normal((nb, 2)).astype(np.float32)
        for t in range(schedule.T, 1, -1):
            k = t - 1
            sw = w[0] * scores[0].score(y, t)
            for i in range(1, len(scores)):
                if w[i] != 0.0:
                    sw = sw + w[i] * scores[i].score(y, t)
            y = inv_sqrt_alpha[k] * (y + half_one_minus[k] * sw)
        out[done:done + nb] = y
        done += nb
    return out.astype(np.float64)
