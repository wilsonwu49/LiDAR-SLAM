"""Part F: paired trials, fixed-radius CBF vs uncertainty-aware CBF.

One variable: whether the barrier radius is inflated by the EKF covariance.
Everything else -- map, seed, path, params, sensor noise -- is held identical,
and each seed is run through BOTH conditions so the comparison is paired.

The independent variable across blocks is how much localisation information the
environment provides (rib spacing). That is the quantity the margin is supposed
to respond to.

    python -m experiments.run_partf            # full run, writes results.npz
    python -m experiments.run_partf --smoke    # 2 seeds, 2 conditions
"""

import sys
import time

import numpy as np

from movement.controller import ControlParams
from sim.scenarios import corridor, run, clearances, pose_errors

RIB_SPACINGS = [None, 4.0, 2.0, 1.0, 0.5]
N_SEEDS = 20
LENGTH, WIDTH = 7.0, 1.4
START = [2.0, 5.0, 0.0]         # well clear of the end wall, so the clearance
GOAL = [7.4, 5.0]               # metric measures the robot, not the start pose
Z_MAX = 2.5                     # short range, so the corridor ends are not visible

# A pillar partway down narrows the gap to 0.8 m. Without it the corridor is so
# wide the CBF never engages and both conditions are trivially identical -- the
# experiment needs a place where getting the pose wrong actually costs you.
PILLAR = np.array([[4.4, 4.30, 4.6, 4.30],
                   [4.6, 4.30, 4.6, 4.90],
                   [4.6, 4.90, 4.4, 4.90],
                   [4.4, 4.90, 4.4, 4.30]])


def scene(rib_spacing):
    return np.vstack([corridor(length=LENGTH, width=WIDTH,
                               rib_spacing=rib_spacing), PILLAR])


def one_trial(rib_spacing, use_cov, seed):
    """One run. Returns the four F.3 metrics plus diagnostics."""
    seg = scene(rib_spacing)
    params = ControlParams()
    r = run(seg, START, GOAL, use_covariance=use_cov, seed=seed,
            max_cycles=2500, z_max=Z_MAX)
    c = clearances(r)
    e = pose_errors(r)
    dev = np.array([i["deviation"] for i in r["log"]])
    marg = np.array([i["R_eff"] for i in r["log"]]) - params.R
    trail = np.array(r["sens"].trail)
    path_len = float(np.linalg.norm(np.diff(trail[:, :2], axis=0), axis=1).sum())

    return dict(
        rib_spacing=-1.0 if rib_spacing is None else rib_spacing,
        use_cov=int(use_cov),
        seed=seed,
        # 1. safety
        min_clearance=float(c.min()),
        # 2. violations
        violation_frac=float(np.mean(c < 0.0)),
        # 3. cost of safety
        path_length=path_len,
        cycles=int(r["cycles"]),
        total_intervention=float(dev.sum()),
        active_frac=float(np.mean([i["active"] for i in r["log"]])),
        # 4. task success -- against GROUND TRUTH, not the robot's belief
        truly_reached=int(r["truly_reached"]),
        believed_reached=int(r["believed_reached"]),
        goal_error=float(r["goal_error"]),
        stalls=int(r["stalls"]),
        # diagnostics
        margin_max=float(marg.max()),
        margin_mean=float(marg.mean()),
        pose_err_mean=float(np.linalg.norm(e[:, :2], axis=1).mean()),
        replans=int(r["replans"]),
    )


def main():
    smoke = "--smoke" in sys.argv
    spacings = RIB_SPACINGS[:2] if smoke else RIB_SPACINGS
    seeds = range(1, 3) if smoke else range(1, N_SEEDS + 1)

    rows = []
    total = len(spacings) * len(seeds) * 2
    t0 = time.perf_counter()
    n = 0
    for sp in spacings:
        for seed in seeds:
            for use_cov in (False, True):          # paired: same seed, both filters
                rows.append(one_trial(sp, use_cov, seed))
                n += 1
                el = time.perf_counter() - t0
                print(f"  [{n:3d}/{total}] ribs={str(sp):>5} cov={int(use_cov)} "
                      f"seed={seed:2d}  clear={rows[-1]['min_clearance']*1000:6.1f}mm "
                      f"reached={rows[-1]['truly_reached']}  "
                      f"({el:5.1f}s, eta {el/n*(total-n):5.1f}s)", flush=True)

    keys = list(rows[0].keys())
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez("experiments/results.npz", **data)
    print(f"\nwrote experiments/results.npz  ({len(rows)} trials, "
          f"{time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
