"""Tiny per-objective score networks + denoising score matching (DSM).

For the learned-score robustness check we train one small MLP per letter by
independent single-objective DSM on that letter's samples.  Each network
predicts the noise eps_theta(x, t); the implied score is

    s_theta(x, t) = - eps_theta(x, t) / sqrt(1 - bar_alpha_t).

The ``LearnedScore`` wrapper exposes the same ``.score(x, t)`` / ``.jacobian(x, t)``
numpy interface as the analytic ``AnalyticScore``, so ``moments.py`` and
``sampler.py`` are reused verbatim -- the only thing that changes is where the
scores come from.  Jacobians are obtained by autograd.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from diffusion import VPSchedule


def fourier_time_embed(t_norm, n_freqs: int = 8):
    """Sinusoidal features of normalized time t/T in [0,1].  t_norm: (N,1)."""
    freqs = 2.0 ** torch.arange(n_freqs, device=t_norm.device) * np.pi
    ang = t_norm * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)   # (N, 2*n_freqs)


class ScoreMLP(nn.Module):
    """Small MLP eps_theta(x, t): R^2 x [0,1] -> R^2."""

    def __init__(self, hidden: int = 128, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 2 + 2 * n_freqs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t_norm):
        emb = fourier_time_embed(t_norm, self.n_freqs)
        return self.net(torch.cat([x, emb], dim=1))


def train_score_mlp(target, schedule: VPSchedule, *, steps: int = 6000,
                    batch: int = 2048, lr: float = 2e-3, hidden: int = 128,
                    seed: int = 0, device: str = "cpu", verbose: bool = True):
    """Train eps_theta on one target by DSM over the full schedule."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ScoreMLP(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched_steps = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    abar = torch.tensor(schedule.abar, dtype=torch.float32, device=device)
    T = schedule.T

    model.train()
    for it in range(steps):
        x0 = torch.tensor(target.sample(batch, rng), dtype=torch.float32, device=device)
        t_idx = torch.randint(0, T, (batch,), device=device)          # 0..T-1  (step t=idx+1)
        ab = abar[t_idx][:, None]
        eps = torch.randn(batch, 2, device=device)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
        t_norm = ((t_idx + 1).float() / T)[:, None]
        eps_hat = model(xt, t_norm)
        loss = ((eps_hat - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched_steps.step()
        if verbose and (it % 1000 == 0 or it == steps - 1):
            print(f"    [train] step {it:5d}  DSM loss {loss.item():.4f}")
    model.eval()
    return model


class LearnedScore:
    """Analytic-compatible score/Jacobian backed by a trained eps network."""

    def __init__(self, model: ScoreMLP, schedule: VPSchedule, device: str = "cpu"):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device
        self._sqrt_1m = np.sqrt(1.0 - schedule.abar)

    def _score_torch(self, x_t, t: int):
        """x_t: torch (N,2) requires_grad as needed -> score torch (N,2)."""
        N = x_t.shape[0]
        t_norm = torch.full((N, 1), t / self.schedule.T, device=self.device)
        eps_hat = self.model(x_t, t_norm)
        return -eps_hat / float(self._sqrt_1m[t - 1])

    @torch.no_grad()
    def score(self, x, t: int):
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32, device=self.device)
        return self._score_torch(xt, t).cpu().numpy()

    def jacobian(self, x, t: int):
        """Per-sample Jacobian d(score)/dx via autograd.  (N,2,2)."""
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32, device=self.device)

        def single(xrow):
            return self._score_torch(xrow[None, :], t)[0]

        jac = torch.vmap(torch.func.jacrev(single))(xt)        # (N, 2, 2)
        return jac.detach().cpu().numpy()
