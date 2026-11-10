import numpy as np

import lidar_nav.navigate as nav
import sim.scenarios as sc
from lidar_nav.control.controller import ControlParams
from sim.sensors import SimSensors
from sim.worlds import WORLDS
from tests.testkit import Suite

T = Suite("Full system: drive to the goal in four worlds")


def drive(w):
    params = ControlParams()
    grid = sc.make_grid(*w["grid_wh"])
    grid.z_max = w["z_max"]
    ekf = sc.make_ekf(w["start"])
    sens = SimSensors(w["segments"], w["start"], params, dt=sc.DT, z_max=w["z_max"],
                      seed=1, movers=w["movers"])
    rec = dict(clear=[], plan_len=[], err=[])

    def hook(cycle, pose, P3, path, grid, info, lethal, replanned, sensed):
        rec["clear"].append(sens.true_clearance(params.r_robot))
        rec["err"].append(np.linalg.norm(sens.pose_true[:2] - pose[:2]))
        if replanned and path is not None:
            rec["plan_len"].append(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    r = nav.navigate(sens, grid, ekf, np.array(w["goal"]), params, dt=sc.DT,
                      max_cycles=w["max_cycles"], on_cycle=hook)
    off = np.linalg.norm(sens.pose_true[:2] - np.array(w["goal"]))
    return r, off, np.array(rec["clear"]), np.array(rec["plan_len"]), np.array(rec["err"]), sens


for w in (WORLDS[name]() for name in ("doorway", "clutter", "crowd", "maze")):
    r, off, clear, plan_len, err, sens = drive(w)
    T.section(f"{w['name']}: {len(w['movers'])} moving obstacles, sensor range {w['z_max']:.0f} m")
    T.check("actually reaches the goal (true position, not belief)", off < 0.3,
            f"{r['outcome']} in {r['cycles'] * sc.DT:.1f} s, {off * 100:.0f} cm from the goal")
    T.check("never touches anything", clear.min() > 0, f"closest {clear.min() * 1000:.0f} mm")
    dist = np.linalg.norm(np.diff(np.array(sens.trail)[:, :2], axis=0), axis=1).sum()
    T.check("position drift stays under 5% of distance driven", err.max() < 0.05 * dist,
            f"worst {err.max() * 1000:.0f} mm over {dist:.1f} m ({100 * err.max() / dist:.1f}%)")
    if "dead_end" in w:
        d = w["dead_end"]
        tr = np.array(sens.trail)
        inside = (tr[:, 0] > d["x0"] + 0.5) & (tr[:, 0] < d["x1"]) & (tr[:, 1] > d["y0"]) & (tr[:, 1] < d["y1"])
        jumps = np.diff(plan_len)
        T.check("drives into the dead-end corridor first", inside.any(),
                f"reaches x = {tr[inside, 0].max():.1f} m of {d['x1']}" if inside.any() else "")
        T.check("discovers the dead end and replans a longer route", jumps.max() > 3.0,
                f"planned route jumps by {jumps.max():.1f} m")

T.done()
