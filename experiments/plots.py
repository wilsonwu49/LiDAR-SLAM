"""Figures for Part F. Reads experiments/results.npz, writes docs/figures/.

    python -m experiments.plots
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "#4a6fa5"      # fixed radius
UNC = "#b5543a"       # uncertainty aware
GREY = "#5b6472"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GREY, "axes.labelcolor": "#1d2129",
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def load(path="experiments/results.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def label_for(v):
    return "none" if v < 0 else f"{v:g} m"


def paired(d, metric):
    """(conditions, baseline_matrix, uncertainty_matrix) aligned by seed."""
    conds = sorted(set(d["rib_spacing"]), key=lambda v: (v < 0, -v))
    b, u = [], []
    for c in conds:
        mb = (d["rib_spacing"] == c) & (d["use_cov"] == 0)
        mu = (d["rib_spacing"] == c) & (d["use_cov"] == 1)
        ob = np.argsort(d["seed"][mb])
        ou = np.argsort(d["seed"][mu])
        b.append(d[metric][mb][ob])
        u.append(d[metric][mu][ou])
    return conds, b, u


def grouped(ax, conds, b, u, ylabel, title, scale=1.0, pct=False):
    x = np.arange(len(conds))
    w = 0.34
    for i, (bb, uu) in enumerate(zip(b, u)):
        for off, vals, col in ((-w / 2, bb, BASE), (w / 2, uu, UNC)):
            v = vals * scale
            ax.bar(i + off, v.mean(), w * 0.9, color=col, alpha=0.75,
                   yerr=v.std() if len(v) > 1 else None,
                   error_kw=dict(ecolor=GREY, lw=1, capsize=3))
            ax.scatter(np.full(len(v), i + off) + np.random.uniform(-0.05, 0.05, len(v)),
                       v, s=9, color="white", edgecolor=col, linewidth=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(c) for c in conds])
    ax.set_xlabel("feature spacing in the corridor")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")


def main():
    d = load()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    # 1. the safety metric
    conds, b, u = paired(d, "min_clearance")
    grouped(axes[0, 0], conds, b, u, "minimum true clearance (mm)",
            "1 · Safety — closest the robot actually came", scale=1000)

    # 2. the cost
    conds, b, u = paired(d, "total_intervention")
    grouped(axes[0, 1], conds, b, u, "total intervention  ∫‖u* − u_nom‖",
            "2 · Cost of safety — how much the filter interfered")

    # 3. the mechanism
    conds, b, u = paired(d, "margin_max")
    grouped(axes[1, 0], conds, b, u, "max margin k·σ_los (mm)",
            "3 · Mechanism — the margin the covariance produced", scale=1000)

    # 4. how often it engaged
    conds, b, u = paired(d, "active_frac")
    grouped(axes[1, 1], conds, b, u, "control steps with the filter active",
            "4 · Engagement — fraction of steps the QP bound", scale=100, pct=True)

    h = [plt.Rectangle((0, 0), 1, 1, color=BASE, alpha=0.75),
         plt.Rectangle((0, 0), 1, 1, color=UNC, alpha=0.75)]
    fig.legend(h, ["fixed radius (baseline)", "uncertainty-aware"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    n = len(d["seed"]) // (2 * len(conds))
    fig.suptitle(f"Part F — paired trials, n = {n} seeds per condition",
                 y=0.965, fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "svg"):
        fig.savefig(f"docs/figures/partf_summary.{ext}", dpi=160)
    print("wrote docs/figures/partf_summary.png / .svg")

    # ---- paired-difference figure: the statistically correct view ----
    fig2, ax = plt.subplots(figsize=(6.4, 4.0))
    conds, b, u = paired(d, "min_clearance")
    for i, (bb, uu) in enumerate(zip(b, u)):
        diff = (uu - bb) * 1000
        ax.scatter(np.full(len(diff), i) + np.random.uniform(-0.06, 0.06, len(diff)),
                   diff, s=22, color="white", edgecolor=UNC, linewidth=1.1, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [diff.mean()] * 2, color=UNC, lw=2.2)
    ax.axhline(0, color=GREY, lw=1, ls="--")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([label_for(c) for c in conds])
    ax.set_xlabel("feature spacing in the corridor")
    ax.set_ylabel("clearance gained (mm)")
    ax.set_title("Paired difference — uncertainty-aware minus baseline, same seed",
                 loc="left", fontweight="bold", fontsize=9.5)
    ax.text(0.99, 0.03, "above zero = safer", transform=ax.transAxes,
            ha="right", color=GREY, fontsize=8)
    fig2.tight_layout()
    for ext in ("png", "svg"):
        fig2.savefig(f"docs/figures/partf_paired.{ext}", dpi=160)
    print("wrote docs/figures/partf_paired.png / .svg")


if __name__ == "__main__":
    main()
