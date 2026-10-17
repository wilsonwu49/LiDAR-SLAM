"""Worlds and a run harness. Importable -- nothing executes at import time.

The harness reports BOTH outcomes: what the robot believed, and what was true.
The gap between them is a result, not a bug.
"""

import time

import numpy as np

from main import navigate
from movement.controller import ControlParams
from occupancy.occupancy_grid import OccupancyGridMap
from sim.sensors import SimSensors
from state_estimation.ekf import ESEKF
from state_estimation.state import NominalState

DT = 0.02


# ----------------------------------------------------------------------- worlds

OPEN_ROOM = np.array([
    [1.0, 1.0, 9.0, 1.0],
    [9.0, 1.0, 9.0, 9.0],
    [9.0, 9.0, 1.0, 9.0],
    [1.0, 9.0, 1.0, 1.0],
])

ROOM = np.vstack([OPEN_ROOM,
                  [[5.0, 1.0, 5.0, 4.0]],
                  [[5.0, 5.2, 5.0, 9.0]]])


def corridor(length=7.0, width=1.2, rib_spacing=None, rib_depth=0.10,
             y_centre=5.0, x0=1.0):
    """
    A straight corridor, optionally with ribs that give ICP x-observability.

    length       corridor length [m]
    width        wall-to-wall [m]
    rib_spacing  metres between ribs; None for a featureless corridor
    rib_depth    how far each rib protrudes into the corridor [m]
    y_centre     corridor centreline [m]
    x0           corridor start [m]

    returns (M, 4) segments

    A featureless corridor is unobservable ALONG its axis: every scan looks the
    same wherever you are, so ICP's covariance blows up in x and the EKF has
    nothing to arrest dead-reckoning drift. Ribs restore that observability.
    rib_spacing is the independent variable for Part F.
    """
    yb, yt = y_centre - width / 2, y_centre + width / 2
    x1 = x0 + length
    seg = [[x0, yb, x1, yb],            # bottom wall
           [x0, yt, x1, yt],            # top wall
           [x0, yb, x0, yt],            # closed start
           [x1, yb, x1, yt]]            # closed end
    if rib_spacing is not None:
        for x in np.arange(x0 + rib_spacing, x1 - 0.3, rib_spacing):
            seg.append([x, yb, x, yb + rib_depth])          # rib up from floor
            seg.append([x, yb + rib_depth, x + 0.05, yb + rib_depth])
            seg.append([x, yt, x, yt - rib_depth])          # rib down from ceiling
            seg.append([x, yt - rib_depth, x + 0.05, yt - rib_depth])
    return np.array(seg, dtype=float)


# -------------------------------------------------------------------- the stack

def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def make_ekf(pose0, sigma_a=0.05, sigma_g=0.005, sigma_ba=1e-4, sigma_bg=1e-5,
             r_enc=0.01):
    """
    Build the filter. sigma_* are Q; r_enc is the encoder measurement sigma.

    Q absorbs what the motion model cannot represent (slip, actuation error).
    r_enc should reflect the real encoder noise -- raise it only to express
    genuine distrust of odometry, not to hide model error.
    """
    x0 = NominalState(p=np.array([pose0[0], pose0[1], 0.0]),
                      v=np.zeros(3), q=yaw_quat(pose0[2]),
                      a_b=np.zeros(3), w_b=np.zeros(3))
    P0 = np.diag([1e-4] * 9 + [1e-3] * 3 + [1e-4] * 3)
    return ESEKF(x0, P0, (sigma_a, sigma_g, sigma_ba, sigma_bg),
                 np.eye(3) * r_enc ** 2, np.diag([1e-3, 1e-3, 1e-4]))


def make_grid(width=10.0, height=10.0, s=0.05):
    return OccupancyGridMap(width, height, s, 0.0, 0.0,
                            l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0,
                            z_max=12.0, r_min=0.15, clearing_dist=8.0)


def run(segments, pose0, goal, use_covariance=True, slip=0.0, sigma_range=0.02,
        max_cycles=2500, seed=1, params=None, ekf_kwargs=None, grid_kwargs=None):
    """
    One trial. Returns a dict with both the believed and the true outcome.
    """
    params = params or ControlParams()
    grid = make_grid(**(grid_kwargs or {}))
    ekf = make_ekf(pose0, **(ekf_kwargs or {}))
    sens = SimSensors(segments, pose0, params, dt=DT, slip_frac=slip,
                      sigma_range=sigma_range, seed=seed)

    t0 = time.perf_counter()
    res = navigate(sens, grid, ekf, np.array(goal, dtype=float), params, dt=DT,
                   max_cycles=max_cycles, use_covariance=use_covariance)
    res["wall_s"] = time.perf_counter() - t0

    res.update(sens=sens, grid=grid, params=params, ekf=ekf,
               true_final=sens.pose_true.copy(),
               est_final=res["log"][-1]["pose"] if res["log"] else None,
               goal=np.array(goal, dtype=float))

    # the robot's own verdict vs the truth -- F.3 metric 4 is the second one
    res["believed_reached"] = res["outcome"] == "reached"
    res["truly_reached"] = bool(
        np.linalg.norm(sens.pose_true[:2] - res["goal"]) < 0.25)
    res["goal_error"] = float(np.linalg.norm(sens.pose_true[:2] - res["goal"]))
    return res


# ------------------------------------------------------------------- ground truth

def clearances(res):
    """Min true clearance at every step of the run (F.3 metrics 1 and 2)."""
    seg = res["sens"].segments
    trail = np.array(res["sens"].trail)
    a, e = seg[:, :2], seg[:, 2:] - seg[:, :2]
    L2 = np.einsum("ij,ij->i", e, e)
    out = []
    for p in trail[:, :2]:
        t = np.clip(np.einsum("ij,ij->i", p - a, e) / np.maximum(L2, 1e-12), 0, 1)
        out.append(np.linalg.norm(p - (a + t[:, None] * e), axis=1).min())
    return np.array(out) - res["params"].r_robot


def pose_errors(res):
    """True-minus-estimated pose error per logged cycle, (N, 3) with wrapped yaw."""
    trail = np.array(res["sens"].trail)
    est = np.array([i["pose"] for i in res["log"]])
    n = min(len(trail), len(est))
    e = trail[:n] - est[:n]
    e[:, 2] = np.arctan2(np.sin(e[:, 2]), np.cos(e[:, 2]))
    return e


def nees(res):
    """
    Normalized estimation error squared, per cycle.

    e^T P^-1 e, which for a CONSISTENT filter averages to the state dimension (3).
    Much larger means the covariance is too small -- the filter is overconfident
    and anything downstream that trusts P (the CBF margin) is being lied to.
    """
    e = pose_errors(res)
    out = []
    for ei, info in zip(e, res["log"][:len(e)]):
        P = info.get("P3")
        if P is None:
            continue
        out.append(float(ei @ np.linalg.solve(P, ei)))
    return np.array(out)
