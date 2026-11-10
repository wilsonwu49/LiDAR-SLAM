import numpy as np
from lidar_nav.mapping.occupancy_grid import OccupancyGridMap
from tests.testkit import Suite

T = Suite("Map: occupancy grid basics")


def fresh(**kw):
    return OccupancyGridMap(width_m=2.0, height_m=2.0, resolution=0.1, origin_x=0.0, origin_y=0.0,
                            l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0, z_max=12.0, **kw)


T.section("grid layout")
g = fresh()
T.check("2 m at 10 cm is 20 x 20 cells", g.W == 20 and g.H == 20 and g.logodds.shape == (20, 20))
T.check("world (0.05, 0.05) is cell (0, 0)", g.world_to_cell(0.05, 0.05) == (0, 0))
T.check("world (1.55, 1.05) is cell (15, 10)", g.world_to_cell(1.55, 1.05) == (15, 10))
T.check("world (1.999, 1.999) is the last cell", g.world_to_cell(1.999, 1.999) == (19, 19))
T.check("bounds check rejects cells off the grid",
        g.in_bounds(19, 19) and not g.in_bounds(20, 19) and not g.in_bounds(-1, 0))

T.section("evidence accumulates and saturates")
T.near("unknown cell is 50/50", g.logodds_to_prob(5, 5), 0.5, 1e-12)
g.update_cell(5, 5, 0.85)
T.near("one hit -> 70%", g.logodds_to_prob(5, 5), 0.700, 1e-3)
g.update_cell(5, 5, 0.85)
T.near("two hits -> 84.5%", g.logodds_to_prob(5, 5), 0.845, 1e-3)
for _ in range(50):
    g.update_cell(5, 5, 0.85)
T.near("many hits clamp at the limit", g.logodds[5, 5], 5.0, 0)
g.update_cell(100, 100, 0.85)
T.check("updating a cell off the grid is ignored, not a crash", True)
g.update_cell(3, 7, 0.85)
T.check("array is indexed [row=y, col=x]", g.logodds[7, 3] == 0.85 and g.logodds[3, 7] == 0.0)

POSE = (1.0, 1.0, 0.0)
T.section("one beam hitting something 0.5 m ahead")
g = fresh(r_min=0.2, clearing_dist=8.0)
g.integrate_scan(POSE, np.array([0.0]), np.array([0.5]))
T.near("hit cell gets one hit of evidence", g.logodds[10, 15], 0.85, 1e-9)
T.check("cells along the beam are marked free",
        all(abs(g.logodds[10, cx] + 0.40) < 1e-9 for cx in range(10, 15)))
T.check("cells behind the hit are left unknown", all(g.logodds[10, cx] == 0.0 for cx in range(16, 20)))
T.check("unrelated cells are untouched", g.logodds[5, 5] == 0.0)

T.section("a beam that hits nothing")
g2 = fresh(r_min=0.2, clearing_dist=8.0)
g2.integrate_scan(POSE, np.array([0.0]), np.array([12.0]))
T.check("no phantom obstacle at max range", not (g2.logodds > 0).any())
T.near("still carves free space", g2.logodds[10, 15], -0.40, 1e-9)

T.section("repeating the same scan")
g3 = fresh(r_min=0.2, clearing_dist=8.0)
for _ in range(20):
    g3.integrate_scan(POSE, np.array([0.0]), np.array([0.5]))
T.near("hit cell saturates at +5", g3.logodds[10, 15], 5.0, 0)
T.near("free cell saturates at -5", g3.logodds[10, 11], -5.0, 0)

T.section("garbage beams")
g4 = fresh(r_min=0.2, clearing_dist=8.0)
g4.integrate_scan(POSE, np.zeros(3), np.array([0.0, 0.1, float("nan")]))
T.check("zero, too-short and nan beams change nothing", not g4.logodds.any())

T.done()
