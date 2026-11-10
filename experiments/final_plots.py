"""Writeup figures: the mechanism (sigma over a run) and the result (clearance vs success).

    python -m experiments.final_plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lidar_nav.navigate as nav
import sim.scenarios as sc
from lidar_nav.control.controller import ControlParams
from sim.scenarios import DT
from sim.sensors import SimSensors
from experiments.two_pinch import (scene, load_prior_map, START, GOAL, Z_MAX,
                                   X_EARLY, X_LATE)

FIXED = "#4a6fa5"
ADAPT = "#b5543a"
GREY = "#5b6472"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GREY})


def sigma_trace(seed=1):
    seg = scene(0.80, 0.75)
    params = ControlParams()
    grid = sc.make_grid(width=12.0, height=10.0)
    grid.z_max = Z_MAX
    load_prior_map(seg, grid)
    grid.integrate_scan = lambda *a, **kw: None
    ekf = sc.make_ekf(START)
    sens = SimSensors(seg, START, params, dt=DT, z_max=Z_MAX, sigma_range=0.02, seed=seed)
    r = nav.navigate(sens, grid, ekf, np.array(GOAL), params, dt=DT,
                      max_cycles=2500, use_covariance=False)
    x = np.array([i["pose"][0] for i in r["log"]])
    P = np.array([i["P3"] for i in r["log"]])
    return x, np.sqrt(P[:, 0, 0]) * 1000, np.sqrt(P[:, 1, 1]) * 1000


def mechanism_figure():
    x, sx, sy = sigma_trace()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(x, sx, color=FIXED, lw=1.8, label="forward (σx)")
    ax.plot(x, sy, color=ADAPT, lw=1.8, label="sideways (σy)")
    for xp, name in ((X_EARLY, "early gate"), (X_LATE, "late gate")):
        ax.axvline(xp, color=GREY, ls="--", lw=1)
        ax.text(xp, ax.get_ylim()[1] * 0.95, f" {name}", color=GREY, fontsize=8, va="top")
    ax.set_xlabel("distance along the route (m)")
    ax.set_ylabel("pose uncertainty (mm)")
    ax.set_title("Pose uncertainty along the route",
                 loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.88))
    fig.tight_layout()
    for ext in ("png", "svg"):
        os.makedirs("docs/figures", exist_ok=True)
        fig.savefig(f"docs/figures/mechanism_sigma.{ext}", dpi=160)
    print("wrote docs/figures/mechanism_sigma.png / .svg")


def result_figure():
    d = np.load("experiments/results/final_results.npz")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for R, k in sorted(set(zip(d["R"], d["k"])), key=lambda c: (c[1] > 0, c[0], c[1])):
        sel = (d["R"] == R) & (d["k"] == k)
        late = d["clear_late"][sel] * 1000
        late = late[~np.isnan(late)]
        succ = d["reached"][sel]
        p = succ.mean()
        hp = 1.96 * np.sqrt(p * (1 - p) / len(succ))
        hl = 1.96 * late.std(ddof=1) / np.sqrt(len(late)) if len(late) > 1 else 0
        col = FIXED if k == 0 else ADAPT
        tag = f"R={R:.2f}" if k == 0 else f"k={k:.1f}"
        ax.errorbar(p * 100, late.mean(), xerr=hp * 100, yerr=hl, fmt="o",
                    color=col, ms=7, capsize=3, lw=1.2)
        ax.annotate(tag, (p * 100, late.mean()), textcoords="offset points",
                    xytext=(7, 5), fontsize=8, color=col)
    ax.set_xlabel("task success (%)")
    ax.set_ylabel("clearance at the late gate (mm)")
    ax.set_title("Fixed buffer vs adaptive margin (95% CI)", loc="left",
                 fontsize=9.5, fontweight="bold")
    ax.plot([], [], "o", color=FIXED, label="fixed buffer")
    ax.plot([], [], "o", color=ADAPT, label="adaptive k·σ")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        os.makedirs("docs/figures", exist_ok=True)
        fig.savefig(f"docs/figures/result_fixed_vs_adaptive.{ext}", dpi=160)
    print("wrote docs/figures/result_fixed_vs_adaptive.png / .svg")


if __name__ == "__main__":
    mechanism_figure()
    try:
        result_figure()
    except FileNotFoundError:
        print("run `python -m experiments.final_run` first for the result figure")
