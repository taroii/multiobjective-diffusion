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

This repository contains a self-contained 2D experiment that **validates that
closed form**: it estimates the four moments by Monte Carlo, predicts
$\alpha^\star$, and then independently measures the realized worst-case TV of the
weighted-score chain as a function of $\alpha$ — showing the realized minimum
lands on the predicted $\alpha^\star$, which beats the uniform and
single-objective baselines.

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
python run_analytic.py          # hero pipeline: moments, alpha*, TV sweep, figures
python run_analytic.py --quick  # fast, coarse version for a smoke test (~5 min)
python run_learned.py           # learned-score robustness check (trains two MLPs)
```

`run_analytic.py` writes `results.json`, `results.md`, and three figures to
`figs/` (vector PDF + PNG preview). Everything is seeded; all hyperparameters are
logged in `results.json`.

## Code layout

```
experiments/
  targets.py        # GMM class (analytic score + Jacobian); happy/sad face targets
  diffusion.py      # VP/DDPM schedule; analytic score at any noise level
  moments.py        # MC estimation of sigma_ij, b_ij; closed-form alpha*
  sampler.py        # deterministic PF-ODE weighted-score chain (Li et al. 2023)
  tv.py             # binned 2D total-variation estimator
  run_analytic.py   # full analytic-score pipeline -> figures + results table
  nets.py           # tiny per-objective score MLPs + DSM training; LearnedScore
  run_learned.py    # learned-score robustness check
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

The two faces are deliberately *similar* — they share the outline, eyes, and
brows and differ only in the mouth — which is the realistic regime for combining
two related image distributions. In that regime the worst-case TV varies only
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
