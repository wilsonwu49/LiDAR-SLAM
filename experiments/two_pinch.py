"""Two pinch points, one early and one late: the case a fixed radius cannot cover.

Lateral pose uncertainty grows with distance travelled (heading drift: nothing
measures absolute heading, so a small angular error becomes a large sideways
error the further you go). Measured on this stack: sigma_y goes from 10 mm to
119 mm over a 7 m run, a 12x spread, while sigma_x stays flat at ~10 mm.

So a route with a tight gap EARLY and another LATE asks one number to do two
jobs:

    R sized for the late gap   -> too fat for the early one
    R sized for the early gap  -> too thin to be safe at the late one

An adaptive margin R + k*sigma_los has the one property that matters here: it
is small when the robot has just localised and large once it has dead-reckoned
for metres.

The map is given a priori and never updated. With SLAM the map is built from the
same drifting pose, so map and robot are wrong together and pose error does not
produce collisions -- that was measured too.
"""

import numpy as np

import main
import sim.scenarios as sc
from movement.controller import ControlParams
from sim.scenarios import DT
from sim.sensors import SimSensors

Y_C, WIDTH, Z_MAX = 5.0, 1.6, 2.5
X0, X1 = 1.0, 11.0
X_EARLY, X_LATE = 2.6, 9.4
START = [1.4, Y_C, 0.0]
GOAL = [10.5, Y_C]


def jamb(x, gap):
    """A pinch point at x: two wall stubs leaving `gap` of opening on the centreline."""
    return [[x, Y_C - WIDTH / 2, x, Y_C - gap / 2],
            [x, Y_C + gap / 2, x, Y_C + WIDTH / 2]]


def scene(gap_early, gap_late):
    yb, yt = Y_C - WIDTH / 2, Y_C + WIDTH / 2
    seg = [[X0, yb, X1, yb], [X0, yt, X1, yt], [X0, yb, X0, yt]]
    seg += jamb(X_EARLY, gap_early)
    seg += jamb(X_LATE, gap_late)
    return np.array(seg, dtype=float)


def load_prior_map(segments, grid, l=5.0):
    for x0, y0, x1, y1 in segments:
        n = int(np.hypot(x1 - x0, y1 - y0) / (grid.s * 0.5)) + 2
        for t in np.linspace(0, 1, n):
            cx, cy = grid.world_to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            if grid.in_bounds(cx, cy):
                grid.logodds[cy, cx] = l


def _clearance_to(segments, p):
    a = segments[:, :2]
    e = segments[:, 2:] - a
    L2 = np.einsum("ij,ij->i", e, e)
    t = np.clip(np.einsum("ij,ij->i", p - a, e) / np.maximum(L2, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * e), axis=1).min()


def trial(R, k, seed, gap_early=0.80, gap_late=0.75, max_cycles=4000):
    seg = scene(gap_early, gap_late)
    params = ControlParams()
    params.R = R          # set post-construction: both conditions are allowed to
    params.k = k          # exceed the planner's inflation, see iso_safety.py
    grid = sc.make_grid(width=12.0, height=10.0)
    grid.z_max = Z_MAX
    load_prior_map(seg, grid)
    grid.integrate_scan = lambda *a, **kw: None
    ekf = sc.make_ekf(START)
    sens = SimSensors(seg, START, params, dt=DT, z_max=Z_MAX,
                      sigma_range=0.02, seed=seed)
    r = main.navigate(sens, grid, ekf, np.array(GOAL), params, dt=DT,
                      max_cycles=max_cycles, use_covariance=k > 0)

    trail = np.array(sens.trail)
    rob = params.r_robot
    # clearance measured separately in each pinch's neighbourhood -- the whole
    # point is that the two gates are passed under different uncertainty
    out = dict(R=R, k=k, seed=seed, outcome=r["outcome"],
               reached=int(np.linalg.norm(sens.pose_true[:2] - np.array(GOAL)) < 0.35),
               cycles=int(r["cycles"]))
    for tag, xp in (("early", X_EARLY), ("late", X_LATE)):
        near = trail[np.abs(trail[:, 0] - xp) < 0.45]
        out[f"clear_{tag}"] = (float(min(_clearance_to(seg, p[:2]) for p in near)) - rob
                               if len(near) else np.nan)
        out[f"passed_{tag}"] = int(len(near) > 0 and trail[:, 0].max() > xp + 0.3)
    if r["log"]:
        m = np.array([i["R_eff"] for i in r["log"]]) - R
        x = np.array([i["pose"][0] for i in r["log"]])
        for tag, xp in (("early", X_EARLY), ("late", X_LATE)):
            sel = np.abs(x - xp) < 0.45
            out[f"margin_{tag}"] = float(m[sel].mean()) if sel.any() else np.nan
    return out


CONFIGS = [("fixed", 0.25, 0.0), ("fixed", 0.30, 0.0), ("fixed", 0.35, 0.0),
           ("fixed", 0.40, 0.0), ("fixed", 0.45, 0.0),
           ("adaptive", 0.25, 0.5), ("adaptive", 0.25, 1.0),
           ("adaptive", 0.25, 1.5)]
SEEDS = range(1, 7)


def main_():
    import time
    rows, t0 = [], time.perf_counter()
    for kind, R, k in CONFIGS:
        for seed in SEEDS:
            t = trial(R, k, seed, max_cycles=3000)
            t["kind"] = kind
            rows.append(t)
            print(f"  {kind:>8} R={R:.2f} k={k:.1f} s={seed} "
                  f"early={t['clear_early']*1000:6.1f} late={t['clear_late']*1000:6.1f} "
                  f"ok={t['reached']} ({time.perf_counter()-t0:.0f}s)", flush=True)

    print()
    print(f"{'config':>16} {'early clear':>13} {'late clear':>13} {'margin e/l':>16} {'success':>9}")
    for kind, R, k in CONFIGS:
        s = [r for r in rows if r["R"] == R and r["k"] == k]
        ce = np.nanmean([r["clear_early"] for r in s]) * 1000
        cl = np.nanmean([r["clear_late"] for r in s]) * 1000
        me = np.nanmean([r.get("margin_early", np.nan) for r in s]) * 1000
        ml = np.nanmean([r.get("margin_late", np.nan) for r in s]) * 1000
        su = np.mean([r["reached"] for r in s]) * 100
        tag = f"fixed R={R:.2f}" if kind == "fixed" else f"adaptive k={k:.1f}"
        print(f"{tag:>16} {ce:12.1f}mm {cl:12.1f}mm {me:6.0f} /{ml:6.0f}mm {su:8.0f}%")
    np.savez("experiments/two_pinch.npz",
             **{c: np.array([r.get(c, np.nan) for r in rows])
                for c in ("R", "k", "seed", "clear_early", "clear_late",
                          "margin_early", "margin_late", "reached", "cycles")})
    print("\nwrote experiments/two_pinch.npz")


if __name__ == "__main__":
    main_()
