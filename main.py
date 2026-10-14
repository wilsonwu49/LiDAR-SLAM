"""The navigation loop: the supervisor that ties all five modules together.

Two clocks. The control step runs every cycle (~50 Hz) and only brakes or nudges.
A* runs rarely (~2 Hz, or on demand) and is the only thing that can find a detour.
"""

import numpy as np

from astar.astar import astar
from astar.costmap import build_costmap
from astar.path import prune_line_of_sight, to_reference_path
from movement.controller import control_step

STALL_SPEED = 0.02        # m/s; below this the robot is not making progress
STALL_STEPS = 25          # consecutive stalled steps before declaring blocked
REPLAN_PERIOD = 25        # cycles between routine replans (25 @ 50 Hz = 0.5 s)


def path_is_blocked(path, grid, lethal):
    """
    True if any waypoint on the current path now sits in a lethal cell.

    path    (K, 2) reference path in the map frame
    grid    OccupancyGridMap, for world_to_cell
    lethal  (H, W) bool costmap from build_costmap
    """
    # 1. for each waypoint, grid.world_to_cell -> (cx, cy)
    # 2. skip out-of-bounds cells (grid.in_bounds)
    # 3. return True on the first lethal[cy, cx]
    for waypoint in path:
        cx, cy = grid.world_to_cell(waypoint[0], waypoint[1])
        if not grid.in_bounds(cx, cy):
            continue
        if lethal[cy, cx]:
            return True
    return False


def plan(grid, pose, goal_xy, params, occ_threshold=0.85):
    """
    Replan from the current pose to the goal on the latest map.

    grid           OccupancyGridMap
    pose           (3,) [x, y, theta]
    goal_xy        (2,) goal in the map frame
    params         ControlParams, for r_robot and r_safety
    occ_threshold  log-odds at or above which a cell counts as occupied

    returns
      path    (K, 2) reference path in the map frame, or None if no route exists
      lethal  (H, W) bool costmap, reused by path_is_blocked next cycle
    """
    # 1. occupancy = (grid.logodds >= occ_threshold).astype(np.int8)
    # 2. lethal = build_costmap(occupancy, grid.s, params.r_robot, params.r_safety)
    # 3. start and goal cells via grid.world_to_cell, remembering astar wants (r, c)
    # 4. cells = astar(start, goal, lethal, connectivity=8, max_expansions=...)
    #    return None if it returns None
    # 5. pruned = prune_line_of_sight(cells, lethal)
    # 6. return to_reference_path(pruned, grid.s, grid.origin_x, grid.origin_y, ...)

    occupancy = (grid.logodds >= occ_threshold).astype(np.int8)
    lethal = build_costmap(occupancy, grid.s, params.r_robot, params.r_safety)
    start_cx, start_cy = grid.world_to_cell(pose[0], pose[1])
    goal_cx, goal_cy = grid.world_to_cell(goal_xy[0], goal_xy[1])
    cells = astar((start_cy, start_cx), (goal_cy, goal_cx), lethal, connectivity=8, max_expansions=20000)
    if cells is None:
        return None, lethal
    pruned = prune_line_of_sight(cells, lethal)
    return to_reference_path(pruned, (grid.origin_x, grid.origin_y), grid.s, resample_ds=grid.s), lethal

def navigate(sensors, grid, ekf, goal_xy, params,
             goal_tol=0.10, max_cycles=10000, use_covariance=True):
    """
    Run until the goal is reached, the route is gone, or the cycle budget runs out.

    sensors         object with read() -> (imu, encoder, scan) and send(u)
    grid            OccupancyGridMap, updated in place each cycle
    ekf             the Module 1 filter; exposes .x (NominalState) and .P (15x15)
    goal_xy         (2,) goal in the map frame
    params          ControlParams
    goal_tol        distance at which the goal counts as reached [m]
    max_cycles      safety stop on runaway loops
    use_covariance  False runs the fixed-radius baseline (Part F's control)

    returns  dict(outcome, cycles, log, replans, stalls)
      outcome  "reached" | "blocked" | "timeout"
      log      per-cycle list of info dicts, for Part F
    """
    path = None
    lethal = None
    stalled = 0
    replans = 0
    stall_events = 0
    log = []

    for cycle in range(max_cycles):

        # 1. SENSE
        imu, encoder, scan = sensors.read()

        # 2. ESTIMATE -- before mapping, or the scan is integrated at a stale pose
        #    ekf.predict(a_m, w_m, dt, g)
        #    enc_ok = ekf.update_encoder(v_body_meas)
        #    icp_ok = ekf.update_icp(rel_pose_meas, R_icp, x_prev)
        #    the bool is the chi-squared gate: a rejected ICP update is the
        #    wheel-slip signal (encoders and scan matching disagree) -- log it
        p_xy, yaw, P3 = ekf.pose_with_covariance()
        pose = np.array([p_xy[0], p_xy[1], yaw])
        gates = ...          # (enc_ok, icp_ok)

        # 3. MAP -- integrate this scan at the pose just estimated
        grid.integrate_scan(pose, scan_angles, scan_ranges)

        # 4. REPLAN?  build the costmap once, use it for both the check and A*
        need = (path is None
                or cycle % REPLAN_PERIOD == 0
                or stalled >= STALL_STEPS)
        if not need and path_is_blocked(path, grid, lethal):
            need = True
        if need:
            if stalled >= STALL_STEPS:
                stall_events += 1          # instrumentation: F.3 metric 4
                stalled = 0
            path, lethal = plan(grid, pose, goal_xy, params)
            replans += 1
            if path is None:
                return dict(outcome="blocked", cycles=cycle, log=log,
                            replans=replans, stalls=stall_events)

        # 5. CONTROL
        u, info = control_step(pose, P3 if use_covariance else None,
                               path, grid, params)

        # 6. ACT
        sensors.send(u)

        # 7. BOOKKEEP
        info["cycle"] = cycle
        info["pose"] = pose.copy()
        info["gates"] = gates
        log.append(info)

        stalled = stalled + 1 if abs(u[0]) < STALL_SPEED else 0

        if np.linalg.norm(pose[:2] - goal_xy) < goal_tol:
            sensors.send(np.zeros(2))
            return dict(outcome="reached", cycles=cycle, log=log,
                        replans=replans, stalls=stall_events)

    sensors.send(np.zeros(2))
    return dict(outcome="timeout", cycles=max_cycles, log=log,
                replans=replans, stalls=stall_events)
