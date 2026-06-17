# Multiobjective Diffusion

This repository is the official implementation of **Non-Aligned Multi-Objective
Diffusion via Weighted-Score Minimax**.

The paper studies sampling from several data distributions
$p_{\mathrm{data}}^1,\dots,p_{\mathrm{data}}^m$ with a **single** shared backward
diffusion chain whose score is a convex combination of the per-objective true
scores, $s_w = \sum_i w_i\, s^{i,\star}$. The decision variable is the weight
vector $w$, chosen to minimize the worst-case time-1 total-variation gap
$\min_{w\in\Delta^m}\max_i \mathrm{TV}(q_1^i, p_1^w)$. The central two-objective
result is a **closed form** for the optimal weight,

$$
\alpha^\star=\frac{\sigma_{12}+\kappa\,b_{12}}{(\sigma_{12}+\sigma_{21})+\kappa\,(b_{12}+b_{21})},
\qquad \kappa=\sqrt{d/\log T},
$$

in terms of four distribution-level moments: the time-averaged score-difference
RMS gaps $\sigma_{ij}$ and Jacobian-difference (operator-norm) gaps $b_{ij}$,
each measured under objective $i$'s own forward marginals.

This repository contains self-contained 2D experiments that validate **both key
results**:

- **Key Result II (closed-form $m=2$).** Estimate the four moments by Monte
  Carlo, predict $\alpha^\star$, then independently measure the realized
  worst-case TV of the weighted-score chain as a function of $\alpha$ — the
  realized minimum lands on the predicted $\alpha^\star$, which beats the uniform
  and single-objective baselines.
- **Key Result I (general $m$).** Estimate the Gram matrices $\Sigma_i,b_i$,
  solve the reduced SOCP for $w^\star$, and show the simplex algorithms MG-AMOO
  and PAMOO reach $w^\star$ at the predicted $\mathcal{O}(1/\sqrt K)$ rate.

The convex bound is the theory's actual object; the realized worst-case TV
validates it up to the (loose) universal constants. See the honest discussion at
the end.

## Why 2D analytic scores

Two load-bearing reasons (see `.notes/experiment_plan.md`):

1. **TV is honestly estimable in 2D.** We bin the plane and compute
   $\mathrm{TV}=\tfrac12\sum_\text{bins}|\hat p - q|$ directly. In image space TV
   cannot be estimated, so any "TV" claim there would be fake.
2. **Scores are closed form.** The forward marginal of a Gaussian mixture is a
   Gaussian mixture, so the true score $s_t^{i,\star}=\nabla\log q_t^i$ and its
   Jacobian have exact expressions. This isolates the *weighting* theory from
   score-estimation error — we test the math, not a neural net. A learned-score
   robustness check (`run_learned.py`) layers estimation error back on top.

## The instance

Two **related** 2D GMM targets: a **happy** and a **sad** smiley face. They
share a circular outline and eye positions (a large common structure) and differ
only in *expression* — the mouth, eyebrows, and eye size. This is the regime the
method is built for: combining two related image distributions (same content,
different attribute) with a single shared chain.

The shared structure is load-bearing. The weighted score
$s_w=\alpha\,\nabla\log q_{\text{happy}}+(1-\alpha)\,\nabla\log q_{\text{sad}}$
is the score of the **geometric mean** $q_{\text{happy}}^{\alpha}q_{\text{sad}}^{1-\alpha}$,
which only has mass where *both* targets do. The big shared outline+eyes
guarantee heavy overlap, so the single weighted-score chain renders a coherent
face whose expression morphs smoothly from sad ($\alpha=0$) to happy
($\alpha=1$) — rather than collapsing, which is what happens for disjoint
targets. (Two distributions placed side by side, e.g. letters "A" and "I", make
the geometric mean degenerate: the chain collapses to the gap and worst-case TV
saturates at $1$ — there is no usable minimum. Overlap is what keeps the
experiment measurable.) The asymmetry — a big grin + arched brows + wide eyes
vs. a small frown + furrowed brows + squinted eyes — pushes the closed-form
$\alpha^\star$ clearly off $\tfrac12$.

## Requirements

Create a fresh environment and install the Python dependencies:

```setup
conda create -n multi-diff python=3.11
conda activate multi-diff
pip install -r requirements.txt
```

Install PyTorch separately to match your hardware (only needed for the
learned-score check, `run_learned.py`). On CPU only:

```setup
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On CUDA (example — substitute your CUDA version):

```setup
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

See https://pytorch.org/get-started/locally/ for the correct command for your
system.

## Running

```setup
cd experiments
# Key Result II -- closed-form two-objective weight (happy/sad faces):
python run_analytic.py          # hero pipeline: moments, alpha*, TV sweep, figures
python run_learned.py           # learned-score robustness check (trains two MLPs)
# Key Result I -- general m, the SOCP and the simplex algorithms (m=3):
python run_socp.py              # Gram moments, SOCP w*, MG-AMOO/PAMOO, figures
python make_figures.py          # re-render all BOUND figures from saved JSON (instant)
```

Both pipelines are seeded and log all hyperparameters to their `results*.json`.
The bound-based figures (`make_figures.py`) are computed analytically from the
moments, so they re-render in seconds without any diffusion sampling.

## Code layout

```
experiments/
  targets.py        # GMM class (analytic score + Jacobian); faces + m-objective targets
  diffusion.py      # VP/DDPM schedule; analytic score at any noise level
  moments.py        # MC estimation of sigma_ij,b_ij (m=2) and Gram Sigma_i,b_i (general m)
  sampler.py        # deterministic PF-ODE weighted-score chain (Li et al. 2023), any m
  tv.py             # binned 2D total-variation estimator (worst-case over m targets)
  socp.py           # constant-free g_i, the reduced SOCP, and MG-AMOO / PAMOO
  pubfigs.py        # publication figure style + bound figures (incl. 3D surface)
  nets.py           # tiny per-objective score MLPs + DSM training; LearnedScore
  run_analytic.py   # Key Result II: closed-form alpha* pipeline -> figures + results
  run_learned.py    # learned-score robustness check
  run_socp.py       # Key Result I: SOCP + algorithms (m=3) -> figures + results
  run_sharp_m2.py   # strongly-asymmetric m=2 demo (the loose-constant regime)
  make_figures.py   # regenerate the bound figures from saved results (no diffusion)
  figs/             # PDF + PNG outputs
```

## Figures

- `figs/hero_scatter.{pdf,png}` — weighted-score samples across $\alpha$: a
  single chain rendering a face that morphs from sad ($\alpha=0$) through the
  balanced expression at $\alpha^\star$ to happy ($\alpha=1$).
- `figs/money_worstTV.{pdf,png}` — the realized worst-case TV bowl vs $\alpha$,
  with the predicted $\alpha^\star$ sitting on the empirical minimum and the
  uniform / single-objective baselines off to the sides. *This figure is the
  theory.*
- `figs/theory_match.{pdf,png}` — the predicted constant-free V-shape
  $\max(g_{\text{happy}},g_{\text{sad}})$ overlaid on the realized TV curve; the
  claim is matched argmins.
- `figs/learned_worstTV.{pdf,png}` — the learned-score robustness check: the
  realized worst-case-TV bowl with learned scores, against the learned and
  analytic $\alpha^\star$.

**Key Result I (general $m$, the bound and the algorithms):**

- `figs/socp_surface3d.{pdf,png}` — the convex minimax $\max_i g_i(w)$ as a **3D
  surface over the simplex** $\Delta^3$, with the SOCP optimum $w^\star$ in the
  valley.
- `figs/socp_trajectories.{pdf,png}` — MG-AMOO and PAMOO average iterates
  descending the simplex to $w^\star$, over the bound contour.
- `figs/socp_convergence.{pdf,png}` — the optimality gap $\Gamma_K$ vs $K$ for
  both algorithms against the predicted $\mathcal{O}(1/\sqrt K)$ rate.
- `figs/socp_simplex.{pdf,png}` — the **realized** worst-case TV over the simplex
  (the honest sampling-side check).
- `figs/m2_bound.{pdf,png}` — the two-objective money figure: $\alpha^\star$
  minimizes the convex bound (uniform pays a clear margin), with the realized-TV
  argmin matching $\alpha^\star$.

## Results

See `experiments/results.md` (regenerated on every run) for the four moments,
$\alpha^\star$, the empirical minimizer $\hat\alpha$, and the
baseline-vs-$\alpha^\star$ worst-case-TV comparison.

### Headline (analytic scores)

With the happy/sad face instance ($T=400$, $1.2\times10^5$ samples per weight,
$180\times180$ TV bins):

- **Closed form:** $\alpha^\star = 0.642$, from moments
  $(\sigma_{12},\sigma_{21},b_{12},b_{21})=(2.77,1.49,2.48,1.47)$ and
  $\kappa=0.578$. The $\sigma$-only ($0.650$) and $b$-only ($0.627$) limits
  bracket it; all three are clearly off $\tfrac12$.
- **Empirical worst-case-TV minimizer:** $\hat\alpha = 0.592$, so
  $\lvert\alpha^\star-\hat\alpha\rvert = 0.050$.
- **$\alpha^\star$ beats the baselines:** worst-case TV $0.198$ at $\alpha^\star$
  vs. $0.251/0.251$ for the single-objective weights (each abandons one
  expression) and $0.203$ for uniform.

The two faces are deliberately *similar* — they share the outline and eyes and
differ only in the mouth — which is the realistic regime for combining two
related image distributions. In that regime the worst-case TV varies only
modestly with $\alpha$ (the bowl is shallow) and uniform is already near optimal,
so $\alpha^\star$'s clear win is over the *single-objective* baselines; it still
lands on the empirical minimum and edges past uniform. Exact numbers are in
`results.md`/`results.json`.

### Learned-score robustness check

Retraining the pipeline with two learned per-objective score MLPs (one per face,
trained by single-objective denoising score matching) re-estimates the moments
and $\alpha^\star$ from the learned scores and reruns the sweep. The closed-form
$\alpha^\star$ continues to track the empirical worst-case-TV minimum under
learned scores; the small shift is the expected signature of score-estimation
error. See `results_learned.json` (regenerated by `run_learned.py`).

### Key Result I — the SOCP and the simplex algorithms ($m=3$)

Three 2D targets of different difficulty (a sharp 16-mode grid, a broad blob, and
a ring). We estimate the Gram matrices $\Sigma_i$ and Jacobian vectors $b_i$,
solve the reduced constant-free SOCP for $w^\star$, and run the two simplex
algorithms:

- **SOCP optimum:** $w^\star=(0.16,\,0.63,\,0.20)$ with bound $\max_i g_i(w^\star)=5.57$,
  versus $9.04$ at uniform — $w^\star$ is **62 % below uniform in the bound**.
- **Algorithms reach it at the predicted rate:** PAMOO converges to $w^\star$ to a
  gap of $\approx2.6\times10^{-3}$ (the smooth $\mathcal O(1/K)$ regime); MG-AMOO
  tracks the $\mathcal O(1/\sqrt K)$ reference. See `socp_convergence` and the
  trajectory / 3D-surface figures. Numbers in `results_socp.json`.

### An honest structural finding (the framing)

Across every instance (faces, letters, sharp-vs-broad grids, the $m=3$ simplex),
the **realized** worst-case-TV optimum sits at or very near the centroid
(uniform). The weighted score is the score of the *geometric mean*
$q_1^{w_1}\!\cdots q_m^{w_m}$, and at the centroid that blend mismatches every
objective about equally, so the realized worst-case TV is nearly flat there.
Consequently:

- $\alpha^\star$ / $w^\star$ **always crush the single-objective weights** (each
  abandons a target) — the large, robust realized win.
- The theory's object is the **convex bound**, and there $\alpha^\star/w^\star$
  beat uniform clearly (e.g. $39\%$ for $m=2$, $62\%$ for $m=3$); the SOCP and the
  $m$-independent algorithms (Key Result I) are validated cleanly.
- When the *moments* predict a large tilt for strongly-asymmetric targets, the
  bound's loose universal constants do not pass that tilt through to realized TV,
  so uniform stays near-optimal *in realized TV*. This gap between the
  bound-optimal weight and the realized optimum is exactly the loose-constant
  slack the theory carries, and we report it rather than hide it.
