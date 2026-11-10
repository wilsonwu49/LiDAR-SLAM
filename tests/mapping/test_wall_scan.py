import numpy as np
from lidar_nav.mapping.occupancy_grid import OccupancyGridMap
from tests.testkit import Suite

T = Suite("Map: a full scan of a wall")

S, Z_MAX = 0.05, 12.0
WALL_X, WALL_Y0, WALL_Y1 = 1.5, 0.5, 1.5
POSE = (1.0, 1.0, 0.0)


def wall_scan(n=360):
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ranges = np.full(n, Z_MAX)
    for i, th in enumerate(angles):
        if np.cos(th) > 1e-9:
            t = (WALL_X - POSE[0]) / np.cos(th)
            if WALL_Y0 <= POSE[1] + t * np.sin(th) <= WALL_Y1:
                ranges[i] = t
    return angles, ranges


angles, ranges = wall_scan()
g = OccupancyGridMap(width_m=2.0, height_m=2.0, resolution=S, origin_x=0.0, origin_y=0.0,
                     l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0, z_max=Z_MAX,
                     r_min=0.2, clearing_dist=8.0)
g.integrate_scan(POSE, angles, ranges)
cy, cx = np.where(g.logodds > 0)

T.section("one scan, wall 0.5 m ahead")
T.check("the wall shows up", len(cx) > 0, f"{len(cx)} occupied cells")
T.check("it is one cell thick, at the right column", len(cx) > 0 and np.all(cx == 30),
        f"columns {sorted(set(cx.tolist()))}")
T.check("it does not extend past the real wall's ends", len(cy) > 0 and cy.min() >= 9 and cy.max() <= 31,
        f"rows {cy.min() if len(cy) else '-'} to {cy.max() if len(cy) else '-'}")
T.check("space between robot and wall is free", all(g.logodds[20, c] < 0 for c in range(22, 30)))
T.check("space behind the wall stays unknown", all(g.logodds[20, c] == 0 for c in range(31, 40)))

T.section("twenty more scans")
for _ in range(20):
    g.integrate_scan(POSE, angles, ranges)
T.near("wall saturates", g.logodds[20, 30], 5.0, 0)
T.near("free space saturates", g.logodds[20, 25], -5.0, 0)

T.section("export to a costmap")
cm = g.export_costmap(unknown_as="occupied")
T.check("40 x 40, int8", cm.shape == (40, 40) and cm.dtype == np.int8)
T.check("wall is blocked", cm[20, 30] == 1)
T.check("free space is open", cm[20, 25] == 0)
T.check("unknown counted as blocked when asked", cm[20, 35] == 1)
T.check("unknown counted as open when asked", g.export_costmap(unknown_as="free")[20, 35] == 0)

T.done()
