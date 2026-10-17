"""The navigation loop: the supervisor that ties all five modules together.

Two clocks. The control step runs every cycle (~50 Hz) and only brakes or nudges.
A* runs rarely (~2 Hz, or on demand) and is the only thing that can find a detour.
"""

import copy

import numpy as np
from scipy import ndimage

from astar.astar import astar
from astar.costmap import build_costmap
from astar.path import prune_line_of_sight, to_reference_path
from icp.icp import icp_scan_match
from movement.controller import control_step
from state_estimation.quaternion import quat_to_rot

STALL_SPEED = 0.02        # m/s; below this the robot is not making progress
STALL_STEPS = 25          # consecutive stalled steps before declaring blocked
REPLAN_PERIOD = 25        # cycles between routine replans (25 @ 50 Hz = 0.5 s)
G_WORLD = np.array([0.0, 0.0, -9.80665])   # sign matches a_true = R(a_m - b) + g
RATE_FLOOR = 0.005        # m/s and rad/s; residual model error, so a perfectly
                          # steady interval still cannot claim infinite precision


def polar_to_body(angles, ranges):
    """(N,) polar scan -> (M, 2) body-frame points, non-returns dropped."""
    ok = np.isfinite(ranges)
    return np.column_stack([ranges[ok] * np.cos(angles[ok]),
                            ranges[ok] * np.sin(angles[ok])])


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

    # Bounds first: astar indexes lethal_mask directly and would raise IndexError
    # on a goal outside the map.
    if not (grid.in_bounds(start_cx, start_cy) and grid.in_bounds(goal_cx, goal_cy)):
        return None, lethal

    # The robot may legally sit inside the inflation band: it is wider than the
    # CBF's reach by design (params.planner_slack). A* cannot expand out of a
    # lethal region because `neighbors` refuses to enter lethal cells, so plan
    # from the nearest free cell instead. Pure pursuit tolerates starting off
    # the path, and the CBF keeps the robot off the wall on the way out.
    if lethal[start_cy, start_cx]:
        if lethal.all():
            return None, lethal                   # nowhere free to escape to
        _, idx = ndimage.distance_transform_edt(lethal, return_indices=True)
        # both from the ORIGINAL cell -- assigning start_cy first would corrupt
        # the lookup for start_cx
        start_cy, start_cx = (int(idx[0][start_cy, start_cx]),
                              int(idx[1][start_cy, start_cx]))

    cells = astar((start_cy, start_cx), (goal_cy, goal_cx), lethal,
                  connectivity=8, max_expansions=20000)
    if cells is None:
        return None, lethal
    pruned = prune_line_of_sight(cells, lethal)
    return to_reference_path(pruned, (grid.origin_x, grid.origin_y), grid.s, resample_ds=grid.s), lethal

def navigate(sensors, grid, ekf, goal_xy, params, dt=0.02,
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

    prev_pts = None          # body-frame points of the last completed scan
    x_at_prev_scan = None    # nominal state when that scan was taken
    cycle_at_prev_scan = 0   # for the interval the displacement was measured over
    w_m_at_prev_scan = None  # for the angular-acceleration estimate
    prev_seq = -1

    for cycle in range(max_cycles):

        # 1. SENSE
        (a_m, w_m), encoder, (scan_angles, scan_ranges), scan_seq = sensors.read()

        # 2. ESTIMATE -- must precede mapping, or the scan is laid down at a
        #    stale pose and every wall gets smeared
        ekf.predict(a_m, w_m, dt, G_WORLD)
        enc_ok = ekf.update_encoder(encoder)

        # ICP only on a NEW revolution. Matching a scan against itself would
        # report zero motion and fight the IMU eight cycles out of nine.
        icp_ok = None
        scan_is_new = scan_seq != prev_seq
        if scan_is_new:
            cur_pts = polar_to_body(scan_angles, scan_ranges)
            if prev_pts is not None and len(cur_pts) > 10 and len(prev_pts) > 10:
                # initial guess: the motion the filter has already propagated
                R_prev = quat_to_rot(x_at_prev_scan.q)
                dp = R_prev.T @ (ekf.x.p - x_at_prev_scan.p)
                dyaw = (np.arctan2(quat_to_rot(ekf.x.q)[1, 0], quat_to_rot(ekf.x.q)[0, 0])
                        - np.arctan2(R_prev[1, 0], R_prev[0, 0]))
                # source = current, reference = previous, so (phi, t) IS the
                # motion of the current frame expressed in the previous frame
                phi, t, cov, icp_info = icp_scan_match(cur_pts, prev_pts,
                                                       init_phi=dyaw,
                                                       init_t=(dp[0], dp[1]))
                dt_scan = max(cycle - cycle_at_prev_scan, 1) * dt
                if icp_info["converged"]:
                    # ICP measured a DISPLACEMENT over dt_scan, expressed in the
                    # previous body frame. Rotate it into the current frame and
                    # divide by the interval: that is a velocity, which is what
                    # scan matching can honestly bound. Feeding it as a position
                    # fix would shrink P against a starting point that was itself
                    # uncertain, which is what made the filter overconfident.
                    c, s = np.cos(-phi), np.sin(-phi)
                    T = np.array([[c, -s, 0.0],
                                  [s,  c, 0.0],
                                  [0.0, 0.0, 1.0]]) / dt_scan   # rotate, then /dt
                    rates = T @ np.array([t[0], t[1], phi])     # [vx, vy, yaw_rate]
                    R_v = T @ cov @ T.T

                    # ICP reports the AVERAGE rate over dt_scan; the filter holds
                    # the instantaneous one. For constant acceleration the gap is
                    # |a| * dt / 2 -- and a is measured, so this is derived rather
                    # than tuned: zero at cruise, large while accelerating.
                    a_body_xy = (a_m - ekf.x.a_b)[:2]
                    s_chord = 0.5 * np.linalg.norm(a_body_xy) * dt_scan + RATE_FLOOR
                    if w_m_at_prev_scan is None:
                        s_yaw = RATE_FLOOR
                    else:
                        # angular acceleration from the gyro difference, same form
                        s_yaw = 0.5 * abs(w_m[2] - w_m_at_prev_scan[2]) + RATE_FLOOR
                    R_v = R_v + np.diag([s_chord ** 2, s_chord ** 2, s_yaw ** 2])
                    icp_ok = ekf.update_icp_rates(rates[:2], rates[2], w_m, R_v)
                else:
                    icp_ok = False
            prev_pts = cur_pts
            x_at_prev_scan = copy.deepcopy(ekf.x)
            cycle_at_prev_scan = cycle
            w_m_at_prev_scan = w_m.copy()
            prev_seq = scan_seq

        p_xy, yaw, P3 = ekf.pose_with_covariance()
        pose = np.array([p_xy[0], p_xy[1], yaw])
        gates = (enc_ok, icp_ok)

        # 3. MAP -- only on a new revolution; re-laying the same scan nine times
        #    just re-saturates the same cells
        if scan_is_new:
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
        info["P3"] = P3.copy()          # for NEES: is the filter's confidence honest?
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
