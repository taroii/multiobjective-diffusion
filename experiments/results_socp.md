# Key Result I: the SOCP and the simplex algorithms (m = 3)

**Setup.** Three 2D targets of deliberately different *difficulty*: a sharp
16-mode Gaussian grid, a broad single Gaussian, and a ring. We estimate the
per-objective Gram matrices $\Sigma_i\in\mathbb{R}^{3\times3}$ and Jacobian
vectors $b_i\in\mathbb{R}^3$ by Monte Carlo (under each objective's own forward
marginals), form the constant-free objectives
$g_i(w)=\sqrt{w^\top\Sigma_i w}+\kappa\,b_i^\top w$ with
$\kappa=\sqrt{d/\log T}$, and solve the reduced second-order cone program
$\min_{w\in\Delta^3}\max_i g_i(w)$.

## The convex bound: $w^\star$ vs. uniform

| weight | bound $\max_i g_i(w)$ |
|---|---|
| **SOCP optimum $w^\star=(0.16,\,0.63,\,0.20)$** | **5.57** |
| uniform $(\tfrac13,\tfrac13,\tfrac13)$ | 9.04 |

$w^\star$ is **62 % below uniform in the bound** — the theory's actual object, in
which the optimal weight clearly beats uniform. (The minimizer is interior, as
expected when the three objectives have genuinely different difficulty.)

## The algorithms reach $w^\star$ at the predicted rate

| algorithm | $\bar w_K$ | final gap $\Gamma_K$ |
|---|---|---|
| PAMOO  | $(0.16,\,0.63,\,0.20)$ | $\approx 2.6\times10^{-3}$ |
| MG-AMOO | $(0.16,\,0.65,\,0.19)$ | $\approx 7.5\times10^{-2}$ |

Both simplex algorithms (instantiations of MG-AMOO / PAMOO from Kretzu et al.
2025) converge to the SOCP optimum: PAMOO at the smooth $\mathcal{O}(1/K)$ rate,
MG-AMOO tracking the $\mathcal{O}(1/\sqrt K)$ reference. See
`figs/socp_convergence`, `figs/socp_trajectories`, and the 3D landscape
`figs/socp_surface3d`.

## Realized worst-case TV (the honest sampling-side check)

Running the weighted-score chain at each weight and binning the time-1 TV gives
the realized worst-case TV over the simplex (`figs/socp_simplex`):

- $w^\star$ **beats every single-objective vertex** clearly (realized worst-case
  TV $\approx 0.76$ at $w^\star$ vs. $\approx 0.83$ at each vertex).
- Uniform is **near-optimal in realized TV** ($\approx 0.70$): the weighted score
  is the score of the geometric mean $q_1^{w_1}q_2^{w_2}q_3^{w_3}$, which at the
  centroid mismatches all three objectives about equally, so the realized
  worst-case TV is nearly flat there.

This is the loose-constant slack the bound carries: the bound-optimal $w^\star$
clearly beats uniform *in the bound*, while in *realized* TV uniform stays
near-optimal. We report the gap rather than hide it. The clean, instance-robust
validations of Key Result I are the SOCP characterization and the
$m$-independent convergence of the algorithms to $w^\star$.

*(All numbers in `results_socp.json`; figures regenerable from it via
`make_figures.py`.)*
