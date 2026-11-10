"""Iso-safety test: does the adaptive margin beat a retuned fixed radius?

The critique this answers: "you could just make R bigger."

A fixed radius must be sized for the worst case; an adaptive one sizes for the
case you are in. That only matters if uncertainty varies during a run. So:
sweep a FIXED radius and sweep the adaptive multiplier k, and compare the two
curves in (minimum clearance, task success) space. If, at equal clearance, the
adaptive filter completes more tasks, the margin earns its place. If the curves
lie on top of each other, it does not, and the honest move is to drop it.

Note on fairness: raising the fixed R past the planner's inflation would
normally trip ControlParams' invariant. The adaptive margin exceeds it too
whenever sigma is large, so both conditions are allowed to here -- the point is
to isolate the CBF, with A* held identical across every run.

    python -m experiments.iso_safety
"""

import time

import numpy as np

import main
import sim.scenarios as sc
from movement.controller import ControlParams
from sim.scenarios import clearances, pose_errors, DT
from sim.sensors import SimSensors
from experiments.prior_map_scene import scene, load_prior_map, START, GOAL, Z_MAX

GAP = 0.85
SEEDS = range(1, 7)
FIXED_R = [0.25, 0.30, 0.35, 0.40, 0.45]      # baseline, retuned
ADAPTIVE_K = [0.5, 1.0, 1.5, 2.0]             # the proposed filter


def trial(R, k, seed, gap=GAP):
    params = ControlParams()
    params.R = R              # set after construction: see the fairness note above
    params.k = k
    grid = sc.make_grid()
    grid.z_max = Z_MAX
    seg = scene(gap)
    load_prior_map(seg, grid)
    grid.integrate_scan = lambda *a, **kw: None
    ekf = sc.make_ekf(START)
    sens = SimSensors(seg, START, params, dt=DT, z_max=Z_MAX,
                      sigma_range=0.02, seed=seed)
    r = main.navigate(sens, grid, ekf, np.array(GOAL), params, dt=DT,
                      max_cycles=3000, use_covariance=k > 0)
    if not r["log"]:
        return dict(R=R, k=k, seed=seed, min_clearance=np.nan, reached=0,
                    cycles=0, outcome=r["outcome"], intervention=np.nan)
    c = clearances({"sens": sens, "params": params})
    return dict(
        R=R, k=k, seed=seed,
        min_clearance=float(c.min()),
        reached=int(np.linalg.norm(sens.pose_true[:2] - np.array(GOAL)) < 0.30),
        cycles=int(r["cycles"]),
        outcome=r["outcome"],
        intervention=float(np.sum([i["deviation"] for i in r["log"]])),
    )


def main_():
    rows = []
    configs = ([("fixed", R, 0.0) for R in FIXED_R] +
               [("adaptive", 0.25, k) for k in ADAPTIVE_K])
    t0 = time.perf_counter()
    for kind, R, k in configs:
        for seed in SEEDS:
            rows.append(dict(kind=kind, **trial(R, k, seed)))
            print(f"  {kind:>8} R={R:.2f} k={k:.1f} seed={seed} "
                  f"clear={rows[-1]['min_clearance']*1000:7.1f}mm "
                  f"reached={rows[-1]['reached']} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    print()
    print(f"{'condition':>22} {'clearance mean':>15} {'success':>9} {'cycles':>8}")
    summary = []
    for kind, R, k in configs:
        sel = [r for r in rows if r["kind"] == kind and r["R"] == R and r["k"] == k]
        cl = np.nanmean([r["min_clearance"] for r in sel]) * 1000
        su = np.mean([r["reached"] for r in sel])
        cy = np.mean([r["cycles"] for r in sel])
        tag = f"fixed R={R:.2f}" if kind == "fixed" else f"adaptive k={k:.1f}"
        print(f"{tag:>22} {cl:14.1f}mm {su*100:8.0f}% {cy:8.0f}")
        summary.append((tag, kind, cl, su))

    print()
    print("iso-safety verdict -- for each adaptive point, the fixed point with the")
    print("closest clearance, and which completes more tasks:")
    for tag, kind, cl, su in summary:
        if kind != "adaptive":
            continue
        f = min((s for s in summary if s[1] == "fixed"), key=lambda s: abs(s[2] - cl))
        verdict = ("adaptive WINS" if su > f[3] + 1e-9 else
                   "fixed wins" if f[3] > su + 1e-9 else "tie")
        print(f"  {tag:>16} ({cl:6.1f}mm, {su*100:3.0f}%)  vs  "
              f"{f[0]:>14} ({f[2]:6.1f}mm, {f[3]*100:3.0f}%)   -> {verdict}")

    np.savez("experiments/iso_safety.npz",
             **{key: np.array([r[key] for r in rows])
                for key in rows[0] if key not in ("kind", "outcome")},
             kind=np.array([r["kind"] for r in rows]),
             outcome=np.array([r["outcome"] for r in rows]))
    print("\nwrote experiments/iso_safety.npz")


if __name__ == "__main__":
    main_()
