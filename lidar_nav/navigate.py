"""Navigation loop: sense, estimate, map, plan, control, act."""

import copy
from dataclasses import replace

import numpy as np
from scipy import ndimage

from lidar_nav.planning.astar import astar
from lidar_nav.planning.costmap import build_costmap
from lidar_nav.planning.path import prune_line_of_sight, to_reference_path
from lidar_nav.scan_matching.icp import icp_scan_match
from lidar_nav.control.controller import control_step
from lidar_nav.estimation.quaternion import quat_to_rot

STALL_SPEED = 0.02
STALL_STEPS = 25
REPLAN_PERIOD = 25
BLOCKED_PATIENCE = 200
STALL_LOOKAHEAD = 0.15
WALL_COMFORT = 0.8
WALL_COST = 3.0
PRUNE_MARGIN = 0.15
STALL_RECOVERY = 100
G_WORLD = np.array([0.0, 0.0, -9.80665])
RATE_FLOOR = 0.005
WAITING = dict(deviation=0.0, active=False, h_min=np.nan, status="waiting", n_obstacles=0)


def polar_to_body(angles, ranges):
    """Turn a LiDAR scan (angle and range per beam) into x, y points around the robot.
    Beams that hit nothing come back as inf and are dropped."""
    ok = np.isfinite(ranges)
    return np.column_stack([ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])])


def yaw_of(q):
    """Pull the heading angle out of the EKF's quaternion orientation."""
    R = quat_to_rot(q)
    return np.arctan2(R[1, 0], R[0, 0])


def path_is_blocked(path, grid, lethal):
    """Check whether the current path now runs through a forbidden cell.
    True means the map changed under the path and it needs replanning.

    path, grid, lethal
    """
    for x, y in path:
        cx, cy = grid.world_to_cell(x, y)
        if grid.in_bounds(cx, cy) and lethal[cy, cx]:
            return True
    return False


def plan(grid, pose, goal_xy, params, occ_threshold=0.85):
    """Plan a route from the robot to the goal on the current map with A*.
    Grows walls by the robot's size, prefers routes that keep away from walls when
    there is room, escapes to a free cell if the robot is too close to a wall, then
    smooths the cell path into evenly spaced waypoints.

    grid, pose, goal_xy, params, occ_threshold
    returns path (K, 2) or None, and the lethal costmap
    """
    occupancy = (grid.logodds >= occ_threshold).astype(np.int8)
    lethal = build_costmap(occupancy, grid.s, params.r_robot, params.r_safety)
    sx, sy = grid.world_to_cell(pose[0], pose[1])
    gx, gy = grid.world_to_cell(goal_xy[0], goal_xy[1])
    if not (grid.in_bounds(sx, sy) and grid.in_bounds(gx, gy)):
        return None, lethal
    if lethal[sy, sx]:
        if lethal.all():
            return None, lethal
        _, idx = ndimage.distance_transform_edt(lethal, return_indices=True)
        sy, sx = int(idx[0][sy, sx]), int(idx[1][sy, sx])
    clearance = ndimage.distance_transform_edt(occupancy == 0) * grid.s
    near_wall = WALL_COST * np.clip((WALL_COMFORT - clearance) / WALL_COMFORT, 0.0, 1.0) ** 2
    cells = astar((sy, sx), (gy, gx), lethal, cost_field=near_wall, connectivity=8,
                  max_expansions=20000)
    if cells is None:
        return None, lethal
    keep_clear = lethal | (clearance < params.r_robot + params.r_safety + PRUNE_MARGIN)
    pruned = prune_line_of_sight(cells, keep_clear)
    path = to_reference_path(pruned, (grid.origin_x, grid.origin_y), grid.s, resample_ds=grid.s)
    return path, lethal


class ScanOdometry:
    """Compare each new LiDAR scan with the previous one to measure how the robot moved.
    The motion is handed to the EKF as a speed and turn rate, never as a position."""

    def __init__(self):
        self.prev_pts = None
        self.x_prev = None
        self.cycle_prev = 0
        self.w_prev = None
        self.seq_prev = -1

    def update(self, ekf, angles, ranges, seq, a_m, w_m, cycle, dt):
        """Run once per cycle. Does nothing unless a new scan has arrived.
        Returns (scan_is_new, icp_ok); icp_ok is None when ICP did not run."""
        if seq == self.seq_prev:
            return False, None
        pts = polar_to_body(angles, ranges)
        ok = None
        if self.prev_pts is not None and len(pts) > 10 and len(self.prev_pts) > 10:
            ok = self._fuse(ekf, pts, a_m, w_m, cycle, dt)
        self.prev_pts = pts
        self.x_prev = copy.deepcopy(ekf.x)
        self.cycle_prev = cycle
        self.w_prev = w_m.copy()
        self.seq_prev = seq
        return True, ok

    def _fuse(self, ekf, pts, a_m, w_m, cycle, dt):
        """Match the two scans, convert the motion to speed and turn rate with an
        uncertainty, and give it to the EKF. Returns whether the EKF accepted it."""
        R_prev = quat_to_rot(self.x_prev.q)
        dp = R_prev.T @ (ekf.x.p - self.x_prev.p)
        dyaw = yaw_of(ekf.x.q) - yaw_of(self.x_prev.q)
        phi, t, cov, info = icp_scan_match(pts, self.prev_pts, init_phi=dyaw,
                                           init_t=(dp[0], dp[1]))
        if not info["converged"]:
            return False
        dt_scan = max(cycle - self.cycle_prev, 1) * dt
        c, s = np.cos(-phi), np.sin(-phi)
        T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) / dt_scan
        rates = T @ np.array([t[0], t[1], phi])
        s_chord = 0.5 * np.linalg.norm((a_m - ekf.x.a_b)[:2]) * dt_scan + RATE_FLOOR
        s_yaw = 0.5 * abs(w_m[2] - self.w_prev[2]) + RATE_FLOOR
        R_v = T @ cov @ T.T + np.diag([s_chord ** 2, s_chord ** 2, s_yaw ** 2])
        return ekf.update_icp_rates(rates[:2], rates[2], w_m, R_v)


def navigate(sensors, grid, ekf, goal_xy, params, dt=0.02, goal_tol=0.10,
             max_cycles=10000, use_covariance=True, on_cycle=None):
    """The main loop: sense, estimate, map, plan, control, act, 50 times a second.
    Runs until the goal is reached, the route stays blocked, or time runs out.
    If the robot stalls, it replans and briefly shortens pure pursuit's look-ahead,
    which stops it cutting a corner into the safety filter's barrier.

    sensors         read() -> (imu, encoder, scan, scan_seq), send(u)
    grid            OccupancyGridMap, updated in place
    ekf             ESEKF
    goal_xy         (2,) goal in the map frame
    params          ControlParams
    dt              control period [s]
    goal_tol        arrival distance [m]
    max_cycles      cycle budget
    use_covariance  False runs the fixed-radius baseline
    on_cycle        optional callback(cycle, pose, P3, path, grid, info, lethal,
                    replanned, sensed), used by the dashboard
    returns dict(outcome, cycles, log, replans, stalls)
    """
    odom = ScanOdometry()
    path = lethal = None
    stalled = blocked_for = replans = stall_events = 0
    tight_until = -1
    log = []

    def finish(outcome, cycles):
        """Stop the motors and package the result. Shared by every way the run ends."""
        sensors.send(np.zeros(2))
        return dict(outcome=outcome, cycles=cycles, log=log, replans=replans,
                    stalls=stall_events)

    for cycle in range(max_cycles):
        (a_m, w_m), encoder, (angles, ranges), seq = sensors.read()

        ekf.predict(a_m, w_m, dt, G_WORLD)
        enc_ok = ekf.update_encoder(encoder)
        new_scan, icp_ok = odom.update(ekf, angles, ranges, seq, a_m, w_m, cycle, dt)
        p_xy, yaw, P3 = ekf.pose_with_covariance()
        pose = np.array([p_xy[0], p_xy[1], yaw])

        if new_scan:
            grid.integrate_scan(pose, angles, ranges)

        replanned = (path is None or cycle % REPLAN_PERIOD == 0 or stalled >= STALL_STEPS
                     or path_is_blocked(path, grid, lethal))
        if replanned:
            if stalled >= STALL_STEPS:
                stall_events += 1
                stalled = 0
                tight_until = cycle + STALL_RECOVERY
            path, lethal = plan(grid, pose, goal_xy, params)
            replans += 1
            blocked_for = blocked_for + 1 if path is None else 0
            if blocked_for > BLOCKED_PATIENCE:
                return finish("blocked", cycle)

        if path is None:
            u = np.zeros(2)
            info = dict(WAITING, R_eff=params.R, carrot=np.full(2, np.nan),
                        obstacles=np.zeros((0, 2)))
        else:
            ctrl = params if cycle >= tight_until else replace(
                params, L_min=STALL_LOOKAHEAD, L_max=STALL_LOOKAHEAD)
            u, info = control_step(pose, P3 if use_covariance else None, path, grid, ctrl)
        sensors.send(u)

        info.update(cycle=cycle, pose=pose.copy(), P3=P3.copy(), gates=(enc_ok, icp_ok))
        log.append(info)
        if on_cycle is not None:
            on_cycle(cycle=cycle, pose=pose, P3=P3, path=path, grid=grid, info=info,
                     lethal=lethal, replanned=replanned,
                     sensed=dict(a_m=a_m, w_m=w_m, encoder=encoder, angles=angles,
                                 ranges=ranges, new_scan=new_scan, enc_ok=enc_ok,
                                 icp_ok=icp_ok))

        stalled = 0 if path is None else (stalled + 1 if abs(u[0]) < STALL_SPEED else 0)
        if np.linalg.norm(pose[:2] - goal_xy) < goal_tol:
            return finish("reached", cycle)

    return finish("timeout", max_cycles)
