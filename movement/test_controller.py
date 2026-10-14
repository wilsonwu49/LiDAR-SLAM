"""End-to-end: occupancy grid + A* path + EKF pose -> a motor command.

Closed-loop, kinematic integration. No EKF in the loop -- the pose is integrated
exactly and P is injected, so this isolates the controller from estimation error.
"""

import numpy as np

from occupancy.occupancy_grid import OccupancyGridMap
from movement.controller import ControlParams, control_step


def make_grid(s=0.05):
    g = OccupancyGridMap(10.0, 10.0, s, 0.0, 0.0, l_occ=0.85, l_free=-0.40,
                         l_min=-5.0, l_max=5.0, z_max=12.0)
    return g


def occupy_segment(g, x0, y0, x1, y1, n=400):
    for t in np.linspace(0.0, 1.0, n):
        cx, cy = g.world_to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        if g.in_bounds(cx, cy):
            g.logodds[cy, cx] = 5.0


def simulate(grid, path, pose0, params, P=None, dt=0.02, max_steps=3000):
    pose = np.array(pose0, dtype=float)
    trail, dev, hmin, nobs, fails = [pose[:2].copy()], [], [], [], 0
    for n in range(max_steps):
        u, info = control_step(pose, P, path, grid, params)
        dev.append(info["deviation"])
        hmin.append(info["h_min"])
        nobs.append(info["n_obstacles"])
        if info["status"] != "QP solved":
            fails += 1
        if u[0] == 0.0 and u[1] == 0.0:
            break
        pose[0] += u[0] * np.cos(pose[2]) * dt
        pose[1] += u[0] * np.sin(pose[2]) * dt
        pose[2] += u[1] * dt
        trail.append(pose[:2].copy())
    return dict(trail=np.array(trail), pose=pose, steps=n,
                dev=np.array(dev), hmin=np.array(hmin),
                nobs=np.array(nobs), fails=fails)


def clearance(trail, grid, r_phys):
    """True min clearance over the run, from the grid's occupied cells."""
    rows, cols = np.nonzero(grid.logodds >= 0.85)
    if rows.size == 0:
        return np.inf
    obs = np.column_stack([grid.origin_x + (cols + 0.5) * grid.s,
                           grid.origin_y + (rows + 0.5) * grid.s])
    d = np.linalg.norm(trail[:, None, :] - obs[None, :, :], axis=2)
    return d.min() - r_phys


def xtrack(trail, path):
    d = np.linalg.norm(trail[:, None, :] - path[None, :, :], axis=2)
    return d.min(axis=1).max()


ok = True


def check(name, cond):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    ok &= bool(cond)


params = ControlParams()
print()
print(f"ControlParams: inflation {params.r_robot + params.r_safety:.2f} m, "
      f"CBF reach {params.R + params.ell:.2f} m, d_gate {params.d_gate:.2f} m")

# ------------------------------------------------------------------ 1. clear run
print()
print("1. empty map, straight path -- the filter must never intervene")
g = make_grid()
path = np.column_stack([np.arange(1.0, 6.0, 0.05), np.full(100, 5.0)])
res = simulate(g, path, [1.0, 5.0, 0.0], params)
print(f"        {res['steps']} steps, final {res['pose'][:2].round(3)}, "
      f"max deviation {res['dev'].max():.2e}")
check("reached the goal", np.linalg.norm(res["pose"][:2] - path[-1]) < 0.10)
check("zero intervention", res["dev"].max() < 1e-6)
check("no solver failures", res["fails"] == 0)

# ------------------------------------------------- 2. wall across the path
print()
print("2. wall across the path at x = 4.0 -- the filter must stop the robot")
g = make_grid()
occupy_segment(g, 4.0, 3.0, 4.0, 7.0)
res = simulate(g, path, [1.0, 5.0, 0.0], params)
d_stop = 4.0 - res["pose"][0]
print(f"        stopped {d_stop:.3f} m short of the wall after {res['steps']} steps")
print(f"        min h (believed) {res['hmin'].min():+.4f}, "
      f"obstacles seen up to {res['nobs'].max()}")
check("did not reach the goal", np.linalg.norm(res["pose"][:2] - path[-1]) > 1.0)
check("stopped outside the CBF reach", d_stop > params.R)
check("barrier never went meaningfully negative", res["hmin"].min() > -0.01)
check("the filter did intervene", res["dev"].max() > 0.01)

# --------------------------------------------- 3. corridor, path down the middle
print()
print("3. 1.2 m corridor -- must pass through without stalling")
g = make_grid()
occupy_segment(g, 1.0, 5.6, 6.0, 5.6)
occupy_segment(g, 1.0, 4.4, 6.0, 4.4)
res = simulate(g, path, [1.0, 5.0, 0.0], params)
print(f"        {res['steps']} steps, final {res['pose'][:2].round(3)}, "
      f"max x-track {xtrack(res['trail'], path)*1000:.1f} mm, "
      f"obstacles/step {res['nobs'].mean():.1f}")
check("traversed the corridor", res["pose"][0] > 5.5)
check("stayed on the path", xtrack(res["trail"], path) < 0.05)
check("no solver failures", res["fails"] == 0)
check("true clearance stayed positive",
      clearance(res["trail"], g, params.r_robot) > 0.0)

# ------------------------------------- 4. uncertainty inflates the stand-off
print()
print("4. same wall, with pose covariance -- must stop EARLIER")
g = make_grid()
occupy_segment(g, 4.0, 3.0, 4.0, 7.0)
base = simulate(g, path, [1.0, 5.0, 0.0], params)
for sigma in [0.02, 0.05, 0.10]:
    P = np.diag([sigma**2, sigma**2, (0.5 * sigma)**2])
    r = simulate(g, path, [1.0, 5.0, 0.0], params, P=P)
    d = 4.0 - r["pose"][0]
    print(f"        sigma = {sigma:.2f} m  ->  stopped {d:.3f} m short "
          f"(+{d - (4.0 - base['pose'][0]):.3f} m vs P = None; "
          f"2*sigma = {2*sigma:.3f})")
    check(f"sigma = {sigma:.2f} stops no later than the fixed radius",
          d >= 4.0 - base["pose"][0] - 1e-6)

P_big = np.diag([0.10**2, 0.10**2, 0.05**2])
r_big = simulate(g, path, [1.0, 5.0, 0.0], params, P=P_big)
check("large covariance stops strictly earlier",
      (4.0 - r_big["pose"][0]) > (4.0 - base["pose"][0]) + 0.01)

# --------------------------------------------------- 5. P = 0 regression
print()
print("5. P = 0 must reproduce P = None")
r0 = simulate(g, path, [1.0, 5.0, 0.0], params, P=np.zeros((3, 3)))
check("same stopping point", np.allclose(r0["pose"], base["pose"], atol=1e-9))

print()
print("ALL CHECKS", "PASSED" if ok else "FAILED")
print()
