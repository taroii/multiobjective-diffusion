"""Minimax weights: argmin_w max_i V_i(w)  (eq. (minimax)).

`minimax_schedule` solves the second-order cone program of
Proposition (compute)(a) over full schedules w = (w_2, ..., w_T);
`minimax_constant` solves the m-variable version for constant weights.  For
m = 2, `m2_schedule` and `m2_constant` implement the explicit solutions of
Proposition (m2) and are used to cross-check the cone program.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq


def _scaled_features(F, beta):
    """sqrt((1 - alpha_t) / 2) F[t, i] for t = 2..T, so V_i(w) = sum_t |Fs[t, i] w_t|^2."""
    return np.sqrt(0.5 * beta[2:])[:, None, None, None] * F[2:]      # (T-1, m, k, m)


def _solve_schedule(Fs, per_step: bool, solver: str):
    """One cone-program formulation of eq. (minimax); returns W (T-1, m) or None."""
    n_t, m, k, _ = Fs.shape
    W = cp.Variable((n_t, m), nonneg=True)
    tau = cp.Variable()
    w = cp.vec(W, order="C")                 # w[t * m + j] = W[t, j]
    rows = (np.arange(n_t * k)[:, None] + np.zeros(m, dtype=int)).ravel()
    cols = (np.repeat(np.arange(n_t) * m, k)[:, None] + np.arange(m)).ravel()
    cons = [cp.sum(W, axis=1) == 1]
    for i in range(m):
        A = sp.csr_matrix((Fs[:, i].reshape(-1), (rows, cols)), shape=(n_t * k, n_t * m))
        if per_step:    # V_i <= tau with one small cone per squared entry
            cons.append(cp.sum(cp.square(A @ w)) <= tau)
        else:           # sqrt(V_i) <= tau as a single large cone (Proposition (compute)(a))
            cons.append(cp.norm(A @ w, 2) <= tau)
    prob = cp.Problem(cp.Minimize(tau), cons)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve(solver=solver)
    except cp.error.SolverError:
        return None, "failed"
    if W.value is None:
        return None, prob.status
    Wopt = np.clip(W.value, 0.0, None)
    return Wopt / Wopt.sum(axis=1, keepdims=True), prob.status


def minimax_schedule(F, beta, solver: str = "CLARABEL"):
    """Minimax schedule over (Delta^m)^{T-1}.

    F: Gram factors from GaussianTargets.gram_features(), shape (T+1, m, k, m).
    Returns W of shape (T+1, m) (rows 0 and 1 are unused and set to uniform) and
    max_i V_i(W), re-evaluated at the returned schedule.

    The two formulations are mathematically equivalent; the per-step one is
    tried first because the interior-point solver converges on it more reliably,
    and the single-cone one is the fallback when it does not report "optimal".
    """
    Fs = _scaled_features(F, beta)
    # The objective is homogeneous, so rescale the data to O(1) for the solver.
    scale2 = np.einsum("tikj,tikj->i", Fs, Fs).max()
    Fs_n = Fs / np.sqrt(scale2)

    def value(W):
        z = np.einsum("tikj,tj->tik", Fs, W)
        return np.einsum("tik,tik->i", z, z).max()

    best = None
    for per_step in (True, False):
        W, status = _solve_schedule(Fs_n, per_step, solver)
        if W is not None and (best is None or value(W) < value(best)):
            best = W
        if status == "optimal":
            break
    if best is None:
        raise RuntimeError("cone program failed in both formulations")
    out = np.full((Fs.shape[0] + 2, Fs.shape[1]), 1.0 / Fs.shape[1])
    out[2:] = best
    return out, value(best)


def _constant_from_gram(G, solver: str):
    """argmin_{w in Delta^m} max_i w^T G_i w for PSD matrices G_i, shape (m, m, m)."""
    scale = np.sqrt(np.trace(G, axis1=1, axis2=2).max())
    m = G.shape[0]
    w = cp.Variable(m, nonneg=True)
    tau = cp.Variable()
    cons = [cp.sum(w) == 1]
    for i in range(m):
        ev, U = np.linalg.eigh(G[i] / scale ** 2)
        R = np.sqrt(np.clip(ev, 0.0, None))[:, None] * U.T            # G_i = R^T R
        cons.append(cp.norm(R @ w, 2) <= tau)
    prob = cp.Problem(cp.Minimize(tau), cons)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=solver)
    if w.value is None:
        raise RuntimeError(f"cone program failed: {prob.status}")
    wopt = np.clip(w.value, 0.0, None)
    wopt /= wopt.sum()
    return wopt, np.einsum("j,ijl,l->i", wopt, G, wopt).max()


def minimax_constant(F, beta, solver: str = "CLARABEL"):
    """Minimax constant weight: argmin_{w in Delta^m} max_i w^T G_i w."""
    Fs = _scaled_features(F, beta)
    return _constant_from_gram(np.einsum("tikj,tikl->ijl", Fs, Fs), solver)


def uniform_average_constant(F, solver: str = "CLARABEL"):
    """Constant weight minimizing the uniform time average of the score error,
    max_i (1/T) sum_{t=1}^T w^T Sigma_i^(t) w  (Assumption 1 of Li et al., 2023).
    This is the comparison rule of Section (experiments), not our method."""
    T = F.shape[0] - 1
    return _constant_from_gram(np.einsum("tikj,tikl->ijl", F[1:], F[1:]) / T, solver)


# ---------------------------------------------------------------------------
# m = 2: explicit solutions (Proposition (m2)), used as a cross-check
# ---------------------------------------------------------------------------
def _m2_ab(F, beta):
    Fs = _scaled_features(F, beta)
    a = np.einsum("tk,tk->t", Fs[:, 0, :, 1], Fs[:, 0, :, 1])   # 1/2 (1-alpha_t) sigma_{12,t}^2
    b = np.einsum("tk,tk->t", Fs[:, 1, :, 0], Fs[:, 1, :, 0])   # 1/2 (1-alpha_t) sigma_{21,t}^2
    return a, b


def m2_schedule(F, beta):
    """theta_t* = theta_t(lambda*), with lambda* found by scalar root-finding."""
    a, b = _m2_ab(F, beta)

    def theta(lam):
        return lam * a / (lam * a + (1.0 - lam) * b)

    def gap(lam):
        th = theta(lam)
        return np.sum(a * (1.0 - th) ** 2) - np.sum(b * th ** 2)

    lam = brentq(gap, 1e-12, 1.0 - 1e-12, xtol=1e-14)
    th = theta(lam)
    W = np.full((len(a) + 2, 2), 0.5)
    W[2:, 0], W[2:, 1] = th, 1.0 - th
    return W, max(np.sum(a * (1.0 - th) ** 2), np.sum(b * th ** 2))


def m2_constant(F, beta):
    """theta-bar of eq. (m2)."""
    a, b = _m2_ab(F, beta)
    k12, k21 = np.sqrt(a.sum()), np.sqrt(b.sum())
    th = k12 / (k12 + k21)
    return np.array([th, 1.0 - th]), (k12 * k21 / (k12 + k21)) ** 2
