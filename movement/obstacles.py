import numpy as np


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
    # 1. validity mask: finite, r_min < range < d_gate
    #    the gate is exact -- beyond it, gamma*h >= v_max and the row can
    #    never bind, so deleting it cannot change u*
    # 2. assign each surviving beam to a wedge:
    #       sector = floor((bearing + pi) / (2*pi / n_sectors))
    # 3. keep the minimum range in each wedge (constraint domination:
    #    same direction, smaller h => tighter bound => dominates its neighbours)
    # 4. body frame xy of the survivors:  r * [cos(bearing), sin(bearing)]
    # 5. rotate by theta and translate by [x, y] to get map frame
    valid = np.isfinite(ranges) & (ranges > r_min) & (ranges < d_gate)
    r = ranges[valid]
    b = bearings[valid]
    sector = np.floor((b + np.pi) / (2 * np.pi / n_sectors)).astype(int) % n_sectors

    best = np.full(n_sectors, -1, dtype=int)              # index, not range
    for i in range(r.size):
        s = sector[i]
        if best[s] < 0 or r[i] < r[best[s]]:
            best[s] = i
    win = best[best >= 0]
    survivor_ranges, survivor_bearings = r[win], b[win]

    if r.size == 0:
        return np.zeros((0, 2))
    
    xy_body = np.stack([survivor_ranges * np.cos(survivor_bearings), survivor_ranges * np.sin(survivor_bearings)], axis=1)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    xy_map = xy_body @ R.T + pose[:2]
    return xy_map
