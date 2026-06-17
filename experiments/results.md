# Analytic-score results

Two related 2D GMM targets: a **happy** and a **sad** smiley face (shared outline + eyes, differing expression); analytic true scores and Jacobians; deterministic PF-ODE weighted-score chain.

## Moments (time-averaged, under each objective's own marginals)

| quantity | value |
|---|---|
| $\sigma_{12}$ (score gap under happy) | 2.7689 |
| $\sigma_{21}$ (score gap under sad) | 1.4895 |
| $b_{12}$ (Jacobian gap under happy) | 2.4752 |
| $b_{21}$ (Jacobian gap under sad) | 1.4734 |
| $\kappa=\sqrt{d/\log T}$ | 0.5778 |

## Predicted vs empirical weight

| weight | value |
|---|---|
| closed-form $\alpha^\star$ | **0.6421** |
| $\sigma$-only limit ($\kappa\to0$) | 0.6502 |
| $b$-only limit ($\kappa\to\infty$) | 0.6268 |
| empirical oracle $\hat\alpha$ (raw TV) | 0.5921 |
| empirical oracle $\hat\alpha$ (bias-corrected) | 0.5921 |
| $|\alpha^\star-\hat\alpha|$ (raw) | 0.0500 |
| $|\alpha^\star-\hat\alpha|$ (bias-corrected) | 0.0500 |

Estimator floors (single-objective TV at the matched weight, pure histogram/discretization bias): $\mathrm{floor}_{happy}=0.1148$, $\mathrm{floor}_{sad}=0.1135$. The bias-corrected oracle subtracts these before taking the max.

## Worst-case TV at the baselines

| weight | worst-case TV |
|---|---|
| single_sad (alpha=0) | 0.2521 |
| single_happy (alpha=1) | 0.2512 |
| uniform (alpha=0.5) | 0.2028 |
| closed-form (alpha*) | 0.1981 |
| empirical (hat_alpha) | 0.1900 |

The closed-form weight beats uniform by +0.0046 TV and the single-objective baselines by +0.0531 (vs the better single objective).
