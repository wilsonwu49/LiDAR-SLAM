"""End-to-end simulation: all five modules driving a robot to a goal."""

import time

import numpy as np

from main import navigate
from movement.controller import ControlParams
from occupancy.occupancy_grid import OccupancyGridMap
from sim.sensors import SimSensors
from state_estimation.ekf import ESEKF
from state_estimation.state import NominalState

DT = 0.02


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def make_ekf(pose0, sigma_encoder=0.01):
    x0 = NominalState(p=np.array([pose0[0], pose0[1], 0.0]),
                      v=np.zeros(3),
                      q=yaw_quat(pose0[2]),
                      a_b=np.zeros(3),
                      w_b=np.zeros(3))
    P0 = np.diag([1e-4] * 3 + [1e-4] * 3 + [1e-4] * 3 + [1e-3] * 3 + [1e-4] * 3)
    Q_params = (0.05, 0.005, 1e-4, 1e-5)          # sigma_a, sigma_g, sigma_ba, sigma_bg
    R_enc = np.eye(3) * sigma_encoder ** 2
    R_icp = np.diag([1e-3, 1e-3, 1e-4])
    return ESEKF(x0, P0, Q_params, R_enc, R_icp)


def make_grid():
    return OccupancyGridMap(10.0, 10.0, 0.05, 0.0, 0.0,
                            l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0,
                            z_max=12.0, r_min=0.15, clearing_dist=8.0)


# a 8x8 room with an interior wall at x = 5 and a 1.2 m doorway at y = 4..5.2
ROOM = np.array([
    [1.0, 1.0, 9.0, 1.0],
    [9.0, 1.0, 9.0, 9.0],
    [9.0, 9.0, 1.0, 9.0],
    [1.0, 9.0, 1.0, 1.0],
    [5.0, 1.0, 5.0, 4.0],
    [5.0, 5.2, 5.0, 9.0],
])

OPEN_ROOM = ROOM[:4]


def run(segments, pose0, goal, use_covariance=True, slip=0.0, max_cycles=2500,
        sigma_range=0.02):
    params = ControlParams()
    grid = make_grid()
    ekf = make_ekf(pose0)
    sens = SimSensors(segments, pose0, params, dt=DT, slip_frac=slip,
                      sigma_range=sigma_range, seed=1)
    t0 = time.perf_counter()
    res = navigate(sens, grid, ekf, np.array(goal), params, dt=DT,
                   max_cycles=max_cycles, use_covariance=use_covariance)
    res["wall_s"] = time.perf_counter() - t0
    res["sens"] = sens
    res["grid"] = grid
    res["params"] = params
    res["true_final"] = sens.pose_true.copy()
    res["est_final"] = res["log"][-1]["pose"] if res["log"] else None
    return res


def clearances(res):
    """Min true clearance over the whole run, replayed from the recorded trail."""
    sens = res["sens"]
    seg = sens.segments
    trail = np.array(sens.trail)
    a = seg[:, :2]
    e = seg[:, 2:] - a
    L2 = np.einsum("ij,ij->i", e, e)
    d = []
    for p in trail:
        t = np.clip(np.einsum("ij,ij->i", p[:2] - a, e) / np.maximum(L2, 1e-12), 0, 1)
        d.append(np.linalg.norm(p[:2] - (a + t[:, None] * e), axis=1).min())
    return np.array(d) - res["params"].r_robot


def report(name, res):
    c = clearances(res)
    err = np.linalg.norm(res["true_final"][:2] - res["est_final"][:2])
    dev = np.array([i["deviation"] for i in res["log"]])
    act = np.mean([i["active"] for i in res["log"]])
    nobs = np.mean([i["n_obstacles"] for i in res["log"]])
    icp_rej = sum(1 for i in res["log"] if i["gates"][1] is False)
    print(f"  {name}")
    print(f"    outcome {res['outcome']:8s}  cycles {res['cycles']:5d}  "
          f"replans {res['replans']:3d}  stalls {res['stalls']}  "
          f"wall {res['wall_s']:5.1f} s")
    print(f"    true final {res['true_final'][:2].round(3)}   "
          f"est error {err*1000:6.1f} mm   min clearance {c.min()*1000:7.1f} mm")
    print(f"    filter active {act*100:5.1f}% of steps, mean deviation "
          f"{dev.mean():.4f}, obstacles/step {nobs:4.1f}, icp rejected {icp_rej}")
    return c


if __name__ == "__main__":
    ok = True
    print()
    print("A. open room, diagonal traverse")
    rA = run(OPEN_ROOM, [2.0, 2.0, 0.0], [8.0, 8.0])
    cA = report("open room", rA)
    ok &= rA["outcome"] == "reached" and cA.min() > 0

    print()
    print("B. interior wall with a 1.2 m doorway -- must route through it")
    rB = run(ROOM, [2.0, 5.0, 0.0], [8.0, 5.0])
    cB = report("doorway", rB)
    ok &= rB["outcome"] == "reached" and cB.min() > 0

    print()
    print("C. same, fixed-radius baseline (Part F control condition)")
    rC = run(ROOM, [2.0, 5.0, 0.0], [8.0, 5.0], use_covariance=False)
    cC = report("doorway, P = None", rC)

    print()
    print("D. 15% wheel slip -- the ICP gate should notice")
    rD = run(ROOM, [2.0, 5.0, 0.0], [8.0, 5.0], slip=0.15)
    cD = report("doorway, 15% slip", rD)

    print()
    print("comparison (B vs C): uncertainty-aware vs fixed radius")
    print(f"    min clearance   {cB.min()*1000:7.1f} mm   vs {cC.min()*1000:7.1f} mm")
    print(f"    cycles to goal  {rB['cycles']:7d}      vs {rC['cycles']:7d}")

    print()
    print("END TO END", "PASSED" if ok else "FAILED")
    print()
