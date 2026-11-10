"""Navigation against a GIVEN map -- the regime where pose uncertainty bites.

With SLAM, the map is built from the same drifting pose, so map and robot are
wrong together and local obstacle avoidance stays correct. Against a prior map
the drift is a real misalignment: the robot genuinely is closer to the doorway
jamb than it believes. That is the only condition under which inflating the
barrier radius by the pose covariance can help.
"""
import numpy as np
import main
from movement.controller import ControlParams
import sim.scenarios as sc
from sim.sensors import SimSensors
from sim.scenarios import corridor, clearances, pose_errors, DT

LEN, WIDTH, Z_MAX, X_DOOR = 8.0, 1.6, 2.5, 8.6
Y_C = 5.0
START = [1.6, Y_C, 0.0]
GOAL = [9.3, Y_C]


def scene(gap):
    """Featureless corridor ending in a doorway of the given width."""
    walls = corridor(length=LEN, width=WIDTH, y_centre=Y_C, x0=1.0)[:2]
    return np.vstack([walls,
                      [[X_DOOR, Y_C - WIDTH / 2, X_DOOR, Y_C - gap / 2]],
                      [[X_DOOR, Y_C + gap / 2, X_DOOR, Y_C + WIDTH / 2]],
                      [[1.0, Y_C - WIDTH / 2, 1.0, Y_C + WIDTH / 2]]])


def load_prior_map(segments, grid, l=5.0):
    """Rasterise the true walls at full confidence. The robot is GIVEN this."""
    for x0, y0, x1, y1 in segments:
        n = int(np.hypot(x1 - x0, y1 - y0) / (grid.s * 0.5)) + 2
        for t in np.linspace(0, 1, n):
            cx, cy = grid.world_to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            if grid.in_bounds(cx, cy):
                grid.logodds[cy, cx] = l


def trial(gap, use_cov, seed, max_cycles=3000):
    seg = scene(gap)
    params = ControlParams()
    grid = sc.make_grid()
    grid.z_max = Z_MAX
    load_prior_map(seg, grid)
    grid.integrate_scan = lambda *a, **k: None       # given map, never updated
    ekf = sc.make_ekf(START)
    sens = SimSensors(seg, START, params, dt=DT, z_max=Z_MAX,
                      sigma_range=0.02, seed=seed)
    r = main.navigate(sens, grid, ekf, np.array(GOAL), params, dt=DT,
                      max_cycles=max_cycles, use_covariance=use_cov)
    r.update(sens=sens, params=params, goal=np.array(GOAL))
    if not r["log"]:
        return None
    c = clearances(r)
    e = pose_errors(r)
    trail = np.array(sens.trail)
    return dict(
        gap=gap, use_cov=int(use_cov), seed=seed,
        min_clearance=float(c.min()),
        violation_frac=float(np.mean(c < 0)),
        max_pose_err=float(np.linalg.norm(e[:, :2], axis=1).max()),
        truly_reached=int(np.linalg.norm(sens.pose_true[:2] - np.array(GOAL)) < 0.30),
        outcome=r["outcome"],
        margin_max=float((np.array([i["R_eff"] for i in r["log"]]) - params.R).max()),
        total_intervention=float(np.sum([i["deviation"] for i in r["log"]])),
        path_length=float(np.linalg.norm(np.diff(trail[:, :2], axis=0), axis=1).sum()),
        cycles=int(r["cycles"]),
    )
