"""Constant-free objective, SOCP solver, and the simplex MG-AMOO / PAMOO
algorithms for the general multi-objective case (paper Key Result I).

The per-objective constant-free objective is

    g_i(w) = sqrt(w^T Sigma_i w) + kappa * b_i^T w ,   kappa = sqrt(d / log T),

each minimized (= 0) at w = e_i.  The minimax  min_{w in simplex} max_i g_i(w)
is the reduced SOCP of Corollary "C1-free reduced SOCP"; its minimizer w* is what
MG-AMOO and PAMOO are proved to reach at an m-independent rate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls  # noqa: F401  (minimize used)

EPS = 1e-12


def kappa_of(schedule):
    return float(np.sqrt(schedule.d / np.log(schedule.T)))


def g_values(w, Sigmas, bs, kappa):
    """Vector [g_i(w)] for all objectives."""
    w = np.asarray(w, float)
    out = np.empty(len(Sigmas))
    for i, (S, b) in enumerate(zip(Sigmas, bs)):
        out[i] = np.sqrt(max(w @ S @ w, 0.0)) + kappa * (b @ w)
    return out


def g_grad(i, w, Sigmas, bs, kappa, tangent=False):
    """Gradient (subgradient at the singularity) of g_i at w.

    With ``tangent=True`` the gradient is projected onto the simplex tangent
    space {sum = 0} (subtract its mean).  This is the component that survives the
    projected step Pi(w - eta * grad); using it keeps PAMOO's combined direction
    J*lambda from collapsing onto the all-ones normal of the simplex.
    """
    S, b = Sigmas[i], bs[i]
    q = w @ S @ w
    quad = (S @ w) / np.sqrt(q) if q > EPS else np.zeros_like(w)
    g = quad + kappa * b
    return g - g.mean() if tangent else g


def max_g(w, Sigmas, bs, kappa):
    return float(np.max(g_values(w, Sigmas, bs, kappa)))


def project_simplex(v):
    """Euclidean projection of v onto the probability simplex."""
    v = np.asarray(v, float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_socp(Sigmas, bs, kappa):
    """Solve the reduced SOCP  min_{w in simplex, tau} tau  s.t. g_i(w) <= tau.

    Convex, so SLSQP from a few starts finds the global minimizer w*.  Returns
    (w_star, value) with value = max_i g_i(w_star).
    """
    m = len(Sigmas)

    def obj(z):
        return z[-1]

    def obj_grad(z):
        g = np.zeros(m + 1); g[-1] = 1.0; return g

    cons = [{"type": "eq", "fun": lambda z: np.sum(z[:m]) - 1.0,
             "jac": lambda z: np.concatenate([np.ones(m), [0.0]])}]
    for i in range(m):
        cons.append({
            "type": "ineq",
            "fun": (lambda z, i=i: z[-1] - (np.sqrt(max(z[:m] @ Sigmas[i] @ z[:m], 0.0))
                                            + kappa * (bs[i] @ z[:m]))),
        })
    bounds = [(0.0, 1.0)] * m + [(0.0, None)]

    best = None
    starts = [np.ones(m) / m] + [np.eye(m)[i] for i in range(m)]
    for w0 in starts:
        z0 = np.concatenate([w0, [max_g(w0, Sigmas, bs, kappa)]])
        res = minimize(obj, z0, jac=obj_grad, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
        w = project_simplex(res.x[:m])
        val = max_g(w, Sigmas, bs, kappa)
        if best is None or val < best[1]:
            best = (w, val)
    return best


def mg_amoo(Sigmas, bs, kappa, K=2000, eta=None, return_traj=False):
    """MG-AMOO on the simplex: descend on the currently worst objective.

    Uses a decaying step eta_k = eta0 / sqrt(k+1) (the schedule that gives the
    O(1/sqrt(K)) average-iterate rate).  Returns (w_bar, hist[, traj]) where
    hist[k] = max_i g_i(running-average iterate) and traj are the iterates.
    """
    m = len(Sigmas)
    eta0 = eta if eta is not None else np.sqrt(2.0)   # ~ diam(simplex)
    w = np.ones(m) / m
    acc = np.zeros(m)
    hist = np.empty(K)
    traj = np.empty((K, m))
    for k in range(K):
        gi = g_values(w, Sigmas, bs, kappa)
        I = int(np.argmax(gi))
        d = g_grad(I, w, Sigmas, bs, kappa, tangent=True)
        nd = np.linalg.norm(d)
        if nd > 1e-12:
            d = d / nd                                # normalized subgradient
        w = project_simplex(w - (eta0 / np.sqrt(k + 1)) * d)
        acc += w
        traj[k] = w
        hist[k] = max_g(acc / (k + 1), Sigmas, bs, kappa)
    wbar = acc / K
    return (wbar, hist, traj) if return_traj else (wbar, hist)


def _pamoo_lambda(Delta, J):
    """argmax_{lambda in simplex}  2 lambda^T Delta - lambda^T (J^T J) lambda.

    The per-step meta-weight QP.  We optimize over the simplex (lambda >= 0,
    sum = 1) rather than the nonnegative orthant: with the simplex tangent
    gradients in J, J^T J is rank-deficient, so the orthant problem is unbounded;
    the compact simplex domain keeps it well-posed and yields a convex
    combination of the per-objective descent directions.
    """
    m = Delta.shape[0]
    Q = J.T @ J
    cons = [{"type": "eq", "fun": lambda l: np.sum(l) - 1.0,
             "jac": lambda l: np.ones(m)}]
    res = minimize(lambda l: -(2 * l @ Delta - l @ Q @ l),
                   np.ones(m) / m, jac=lambda l: -(2 * Delta - 2 * Q @ l),
                   bounds=[(0.0, 1.0)] * m, constraints=cons,
                   method="SLSQP", options={"maxiter": 200, "ftol": 1e-10})
    return project_simplex(res.x)


def pamoo(Sigmas, bs, kappa, K=2000, eta=None, return_traj=False):
    """PAMOO on the simplex: per-step QP for the meta-weights, then descend.

    Decaying step eta_k = eta0 / sqrt(k+1).  Returns (w_bar, hist[, traj]).
    """
    m = len(Sigmas)
    eta0 = eta if eta is not None else np.sqrt(2.0)
    w = np.ones(m) / m
    acc = np.zeros(m)
    hist = np.empty(K)
    traj = np.empty((K, m))
    for k in range(K):
        Delta = g_values(w, Sigmas, bs, kappa)
        J = np.stack([g_grad(i, w, Sigmas, bs, kappa, tangent=True) for i in range(m)], axis=1)
        lam = _pamoo_lambda(Delta, J)
        d = J @ lam
        nd = np.linalg.norm(d)
        if nd > 1e-12:
            d = d / nd
        w = project_simplex(w - (eta0 / np.sqrt(k + 1)) * d)
        acc += w
        traj[k] = w
        hist[k] = max_g(acc / (k + 1), Sigmas, bs, kappa)
    wbar = acc / K
    return (wbar, hist, traj) if return_traj else (wbar, hist)
