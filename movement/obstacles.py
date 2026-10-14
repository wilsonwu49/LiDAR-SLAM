import numpy as np


def sector_min(ranges, bearings, n_sectors):
    """
    Indices of the nearest point in each angular wedge (constraint domination).

    ranges     (N,) distance from the robot [m]
    bearings   (N,) direction in the body frame [rad], any wrapping
    n_sectors  wedges around the full circle

    returns    (M,) indices into ranges/bearings, one per occupied wedge
    """
    sector = np.floor((bearings + np.pi) / (2 * np.pi / n_sectors)).astype(int) % n_sectors

    best = np.full(n_sectors, -1, dtype=int)
    for i in range(ranges.size):
        s = sector[i]
        if best[s] < 0 or ranges[i] < ranges[best[s]]:
            best[s] = i
    return best[best >= 0]


def grid_to_obstacles(grid, pose, d_gate, n_sectors=36, occ_threshold=0.85):
    """
    Obstacle points from the occupancy grid: occupied cells near the robot, reduced.

    grid           OccupancyGridMap; uses .logodds[cy, cx], .s, .origin_x/.origin_y, .W/.H
    pose           (3,) [x, y, theta] in the map frame
    d_gate         drop anything farther than this; = R + v_max / gamma
    n_sectors      angular wedges to reduce over
    occ_threshold  log-odds at or above which a cell counts as occupied

    returns        (M, 2) obstacle points in the map frame, M <= n_sectors
    """
    # 1. robot cell via grid.world_to_cell; window half-width = ceil(d_gate / grid.s)
    #    slice the window out of grid.logodds, CLIPPED to [0, W) and [0, H)
    #    (a square window is the cheap superset of the circular gate -- slicing
    #     900 cells beats a distance test on all 40,000)
    # 2. occupied mask: window >= occ_threshold;  np.nonzero gives (rows, cols)
    #    relative to the window -- add the window's offsets to get absolute cx, cy
    # 3. cell CENTRES in world coords, not corners:
    #       wx = grid.origin_x + (cx + 0.5) * grid.s
    #    (using the corner costs half a cell = 2.5 cm of position error)
    # 4. range and bearing from the robot:
    #       dx = wx - pose[0];  dy = wy - pose[1]
    #       r  = hypot(dx, dy);  b = arctan2(dy, dx) - pose[2]
    # 5. exact circular gate: keep r < d_gate
    # 6. sector_min(r, b, n_sectors) -> winners
    # 7. return the world xy of the winners -- the grid is ALREADY in map frame,
    #    so unlike the scan version there is no transform back
    cx, cy = grid.world_to_cell(pose[0], pose[1])
    half_width = int(np.ceil(d_gate / grid.s))
    x0 = max(cx - half_width, 0)
    x1 = min(cx + half_width + 1, grid.W)
    y0 = max(cy - half_width, 0)
    y1 = min(cy + half_width + 1, grid.H)

    window = grid.logodds[y0:y1, x0:x1]
    occupied = window >= occ_threshold
    if not np.any(occupied):
        return np.zeros((0, 2))

    rows, cols = np.nonzero(occupied)
    wx = grid.origin_x + (cols + x0 + 0.5) * grid.s
    wy = grid.origin_y + (rows + y0 + 0.5) * grid.s

    dx, dy = wx - pose[0], wy - pose[1]
    r = np.hypot(dx, dy)
    b = np.arctan2(dy, dx) - pose[2]

    near = r < d_gate
    wx, wy, r, b = wx[near], wy[near], r[near], b[near]
    if r.size == 0:
        return np.zeros((0, 2))

    win = sector_min(r, b, n_sectors)
    return np.stack([wx[win], wy[win]], axis=1)



def scan_to_obstacles(ranges, bearings, pose, d_gate, n_sectors=36, r_min=0.05):
    """
    Reduce a raw scan to the few obstacle points that can actually bind the QP.

    ranges     (N,) range per beam [m], non-returns as inf or nan
    bearings   (N,) beam angle in the body frame [rad]
    pose       (3,) [x, y, theta] in the map frame
    d_gate     drop anything farther than this; = R + v_max / gamma
    n_sectors  angular wedges to reduce over; error is bounded by half a wedge
    r_min      drop returns closer than this (self-hits, specular noise)

    returns    (M, 2) obstacle points in the map frame, M << N
    """
    valid = np.isfinite(ranges) & (ranges > r_min) & (ranges < d_gate)
    r = ranges[valid]
    b = bearings[valid]
    if r.size == 0:
        return np.zeros((0, 2))

    win = sector_min(r, b, n_sectors)                     # index, not range
    survivor_ranges, survivor_bearings = r[win], b[win]


    xy_body = np.stack([survivor_ranges * np.cos(survivor_bearings), survivor_ranges * np.sin(survivor_bearings)], axis=1)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    xy_map = xy_body @ R.T + pose[:2]
    return xy_map
