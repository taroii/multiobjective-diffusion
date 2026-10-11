"""Tables and figure for run_random_weights.py.

Writes to results/:
  random_weights_table.tex / .md      main table: mean worst-case TV per rule
  random_weights_paper_instance.tex   the fixed instance of the experiments section
  random_weights_paired.tex           appendix table: paired excess over our weights
  random_weights_appendix.pdf / .png  appendix figure: the same, one dot per instance

Run:  python plot_random_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "results"
BLUE, GREY = "#2a78d6", "#8a8984"
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f1f5fb"

SETTINGS = [("common_m2", "Common", 2), ("common_m3", "Common", 3), ("common_m5", "Common", 5),
            ("hetero_m2", "Different", 2), ("hetero_m3", "Different", 3), ("hetero_m5", "Different", 5)]
# columns of the main table; the last one is a reference, not a rule
COLUMNS = [("random_constant", "Random"), ("equal", "Equal"), ("uniform_average", "Unif.-avg."),
           ("minimax_constant", "Ours (const.)"), ("minimax_schedule", "Ours"),
           ("best_constant", "Best found")]
OTHERS = [("random_constant", "Random weights (fixed)"),
          ("random_schedule", "Random weights (redrawn every step)"),
          ("equal", "Equal weights"),
          ("uniform_average", "Uniform-average rule"),
          ("best_constant", "Best constant weight found")]


def _bold(vals, fmt, wrap):
    best = min(round(v, 3) for v in vals)
    return [wrap(fmt(v)) if round(v, 3) == best else fmt(v) for v in vals]


def mean_F(s, key):
    return s[f"F_{key}"]["mean"] if f"F_{key}" in s else s["vs_minimax_schedule"][key]["mean"]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def main_table(meta, summary, tag):
    f3 = lambda v: f"{v:.3f}"
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{lc ccc cc c}", r"\toprule",
           r" & & \multicolumn{3}{c}{Baselines} & \multicolumn{2}{c}{Minimax weights} & Reference \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
           r"Covariances & $m$ & Random & Equal & Unif.-avg.\ & Constant & Schedule & Best found \\",
           r"\midrule"]
    md = ["| Covariances | m | Random | Equal | Unif.-avg. rule | Ours, constant | Ours, schedule "
          "| *Best constant found* |", "|---|---|---|---|---|---|---|---|"]
    for name, cov, m in SETTINGS:
        if name not in summary:
            continue
        s = summary[name]
        vals = [mean_F(s, k) for k, _ in COLUMNS]
        rules, ref = vals[:-1], vals[-1]
        if name == "hetero_m2":
            tex.append(r"\midrule")
        tex.append(f"{cov} & {m} & " + " & ".join(_bold(rules, f3, lambda x: rf"\textbf{{{x}}}"))
                   + rf" & \textit{{{ref:.3f}}} \\")
        md.append(f"| {cov} | {m} | " + " | ".join(_bold(rules, f3, lambda x: f"**{x}**"))
                  + f" | *{ref:.3f}* |")
    T = meta["configs"]["common_m3"]["T"]
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Worst-case total variation $F_T(\bw)=\max_i\TV(q_1^i,p_1^{\bw})$ of the sampler "
            r"(lower is better), averaged over "
            f"{meta['n_seeds']} random instances of $m$ Gaussian targets in $d=2$ with $T={T}$; "
            r"the best rule in each row is in bold. "
            r"\emph{Random}: a weight drawn uniformly from the simplex and kept fixed (mean over "
            f"{meta['n_const']} draws per instance). "
            r"\emph{Equal}: $w=1/m$. "
            r"\emph{Unif.-avg.}: the constant weight minimizing the uniform time average of the score error. "
            r"\emph{Minimax}: our weights~\eqref{eq:minimax}, restricted to constant weights or as a full schedule. "
            r"\emph{Best found}: the constant weight with the smallest $F_T$ found by direct search, "
            r"shown as a reference; a schedule can fall below it. "
            r"All values are computed exactly (no sampling error); paired differences are in "
            r"Table~\ref{tab:random-paired}.}",
            r"\label{tab:random}", r"\end{table}"]
    (OUT / f"{tag}_table.tex").write_text("\n".join(tex) + "\n")
    (OUT / f"{tag}_table.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


def paper_instance_table(meta, data, tag):
    if "paper_m3" not in data["summary"]:
        return ""
    s, row = data["summary"]["paper_m3"], data["raw"]["paper_m3"][0]
    best = s["F_best_constant"]["mean"]
    wfmt = lambda w: "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
    lines = [("Best constant found", wfmt(row["w_best_constant"]), best, 0.0),
             (r"\textbf{Minimax weights \eqref{eq:minimax}}", wfmt(row["w_minimax_constant"]),
              s["F_minimax_schedule"]["mean"], s["dist_excess_minimax_constant"]),
             ("Equal weights", wfmt([1 / 3] * 3), s["F_equal"]["mean"], s["dist_excess_equal"]),
             ("Uniform time average", wfmt(row["w_uniform_average"]), s["F_uniform_average"]["mean"],
              s["dist_excess_uniform_average"]),
             ("Uniform random $w$ (mean)", "---", s["vs_minimax_schedule"]["random_constant"]["mean"],
              s["dist_excess_random_constant"])]
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Rule & Selected weights & Worst-case TV & Excess Mahalanobis distance \\", r"\midrule"]
    md = ["| Rule | Selected weights | Worst-case TV | Excess Mahalanobis distance |", "|---|---|---|---|"]
    for label, w, F, d in lines:
        tex.append(rf"{label} & ${w}$ & ${F:.3f}$ & ${100 * d:+.1f}\%$ \\" if w != "---"
                   else rf"{label} & --- & ${F:.3f}$ & ${100 * d:+.1f}\%$ \\")
        clean = label.replace(r"\textbf{", "").replace(r" \eqref{eq:minimax}}", "").replace("$", "")
        md.append(f"| {clean} | {w} | {F:.3f} | {100 * d:+.1f}% |")
    share = 100 * s["frac_random_constant_draws_beating_uniform_average"]
    T = meta["configs"]["paper_m3"]["T"]
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{The fixed instance: $m=3$, $d=2$, $\Sze=\mathrm{diag}(30,1)$, means at "
            r"Mahalanobis distance $2.86$ from their center and $120^\circ$ apart in the whitened "
            rf"coordinates, target~1 on the high-variance axis; $T={T}$, $c_0={meta['c0']:g}$, $c_1={meta['c1']:g}$. "
            r"The last column is the worst-case distance $\max_i\|\mu_w-\mu_i\|_{\Sze^{-1}}$ relative to the best "
            rf"constant weight found. {share:.0f}\% of uniformly random weights have a smaller worst-case TV than "
            r"the uniform-time-average rule; none has a smaller one than the minimax weights.}",
            r"\label{tab:metric}", r"\end{table}"]
    (OUT / f"{tag}_paper_instance.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md) + f"\n\n{share:.1f}% of random draws beat the uniform-average rule."


def paired_table(meta, summary, tag):
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{lcl r@{\;}l r}", r"\toprule",
           r"Covariances & $m$ & Compared with & \multicolumn{2}{c}{Excess worst-case TV, median [IQR]} "
           r"& Beats ours \\", r"\midrule"]
    for name, cov, m in SETTINGS:
        if name not in summary:
            continue
        for j, (k, label) in enumerate(OTHERS):
            a = summary[name]["vs_minimax_schedule"][k]
            beat = a["frac_draws_beating_ours"] if k.startswith("random") else a["frac_seeds_beating_ours"]
            lead = f"{cov} & {m}" if j == 0 else " & "
            tex.append(rf"{lead} & {label} & ${a['diff_median']:+.3f}$ & "
                       rf"$[{a['diff_iqr'][0]:+.3f},\,{a['diff_iqr'][1]:+.3f}]$ & ${100 * beat:.0f}\%$ \\")
        tex.append(r"\addlinespace")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}",
            r"\caption{Paired comparison with the minimax schedule over "
            f"{meta['n_seeds']} random instances: worst-case TV of the other rule minus ours on the same "
            r"instance (positive: ours is better). For the random rules the per-instance value is the mean over "
            r"the draws, and \emph{Beats ours} is the share of individual draws with a smaller worst-case TV "
            r"than ours; for the other rules it is the share of instances.}",
            r"\label{tab:random-paired}", r"\end{table}"]
    (OUT / f"{tag}_paired.tex").write_text("\n".join(tex) + "\n")


# ---------------------------------------------------------------------------
# appendix figure: paired excess over our weights, one dot per instance
# ---------------------------------------------------------------------------
def figure(data, tag):
    raw, tol = data["raw"], data["meta"]["beat_tol"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), sharex=True, sharey=True)
    lo, hi = -0.16, 0.36
    rng = np.random.default_rng(0)
    ypos = np.arange(len(OTHERS))[::-1]
    for ax, (name, cov, m) in zip(axes.ravel(), SETTINGS):
        rows = raw[name]
        ours = np.array([r["F_minimax_schedule"] for r in rows])
        ax.axvspan(0, hi, color=SHADE, lw=0)
        ax.axvline(0, color=INK, lw=1.0)
        for y, (k, _) in zip(ypos, OTHERS):
            other = np.array([r[f"F_{k}"] for r in rows])
            draws = other if other.ndim == 2 else other[:, None]
            diff = draws.mean(axis=1) - ours
            color = GREY if k == "best_constant" else BLUE
            ax.scatter(np.clip(diff, lo, hi), y + rng.uniform(-0.22, 0.22, len(diff)), s=11,
                       color=color, alpha=0.45, linewidth=0, zorder=3)
            q25, q50, q75 = np.percentile(diff, [25, 50, 75])
            ax.plot([q25, q75], [y, y], color=INK, lw=2.2, solid_capstyle="butt", zorder=4)
            ax.plot([q50], [y], marker="o", ms=5.5, mfc="white", mec=INK, mew=1.4, zorder=5)
            beat = 100 * np.mean(draws < ours[:, None] - tol)
            ax.text(hi - 0.005, y + 0.3, f"beats ours: {beat:.0f}%", ha="right", va="center",
                    fontsize=8, color=MUTED, zorder=6)
        ax.set_title(f"{cov} covariance{'s' if cov == 'Different' else ''}, {m} targets",
                     fontsize=10.5, loc="left", pad=6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, len(OTHERS) - 0.4)
        ax.set_yticks(ypos, [label for _, label in OTHERS])
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Worst-case TV minus ours   (→ ours better)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{tag}_appendix.{ext}", dpi=200, bbox_inches="tight")


def main(tag="random_weights"):
    data = json.loads((OUT / f"{tag}.json").read_text())
    meta, summary = data["meta"], data["summary"]
    print(main_table(meta, summary, tag), "\n")
    print(paper_instance_table(meta, data, tag))
    paired_table(meta, summary, tag)
    figure(data, tag)


if __name__ == "__main__":
    main()
