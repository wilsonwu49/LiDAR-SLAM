import numpy as np
from lidar_nav.mapping.occupancy_grid import OccupancyGridMap
from lidar_nav.control.obstacles import grid_to_obstacles
from tests.testkit import Suite

T = Suite("Safety filter: obstacles from the map")


def check(name, cond, extra=""):
    return T.check(name, cond, extra)


D_GATE, NS = 0.75, 36

def make(w=10.0, h=10.0, s=0.05, ox=0.0, oy=0.0):
    return OccupancyGridMap(w, h, s, ox, oy, l_occ=0.85, l_free=-0.40,
                            l_min=-5.0, l_max=5.0, z_max=12.0)

def occupy(g, wx, wy, val=5.0):
    cx, cy = g.world_to_cell(wx, wy)
    g.logodds[cy, cx] = val

T.section("empty grid")
g = make()
check("empty map gives no obstacles", grid_to_obstacles(g, np.array([5.,5.,0.]), D_GATE, NS).shape == (0, 2))

T.section("single occupied cell, 0.5 m ahead of a robot at (5,5)")
g = make(); occupy(g, 5.5, 5.0)
obs = grid_to_obstacles(g, np.array([5., 5., 0.]), D_GATE, NS)
check("one point", len(obs) == 1)
# cell containing 5.5 starts at 5.50, centre 5.525;  y cell starts 5.00, centre 5.025
check("placed at the cell centre, not its corner",
      np.allclose(obs[0], [5.525, 5.025]))

T.section("gate: a cell 1.0 m away is outside d_gate = 0.75")
g = make(); occupy(g, 6.0, 5.0)
check("ignored", len(grid_to_obstacles(g, np.array([5.,5.,0.]), D_GATE, NS)) == 0)

T.section("row/column order")
g = make(); occupy(g, 5.3, 5.0)          # 0.3 m in +x only
obs = grid_to_obstacles(g, np.array([5., 5., 0.]), D_GATE, NS)
check("found to the right, not above (row/column order correct)", obs[0][0] > 5.2 and abs(obs[0][1] - 5.025) < 1e-9)

T.section("domination: wall of cells, one per wedge")
g = make()
for wy in np.arange(4.4, 5.65, 0.05):
    occupy(g, 5.4, wy)
obs = grid_to_obstacles(g, np.array([5., 5., 0.]), D_GATE, NS)
T.info(f"{int(np.sum(g.logodds >= 0.85))} occupied cells -> {len(obs)} points")
check("at most one per wedge", len(obs) <= NS)
check("all inside the gate",
      np.all(np.linalg.norm(obs - np.array([5., 5.]), axis=1) < D_GATE + 1e-9))

T.section("map edge: robot at (0.2, 0.2) must not wrap to the far corner")
g = make(); occupy(g, 9.9, 9.9)          # diagonally opposite
obs = grid_to_obstacles(g, np.array([0.2, 0.2, 0.0]), D_GATE, NS)
check("nothing found (no negative-index wrap)", len(obs) == 0)
occupy(g, 0.4, 0.2)
obs = grid_to_obstacles(g, np.array([0.2, 0.2, 0.0]), D_GATE, NS)
check("the real nearby cell is found", len(obs) == 1 and abs(obs[0][0] - 0.425) < 1e-9)

T.section("negative origin")
g = make(ox=-5.0, oy=-5.0); occupy(g, 0.4, 0.0)
obs = grid_to_obstacles(g, np.array([0., 0., 0.]), D_GATE, NS)
check("map offset handled", np.allclose(obs[0], [0.425, 0.025]))

T.section("rotating the robot changes which cells win, but they must stay real wall cells")
g = make()
for wy in np.arange(4.6, 5.45, 0.05):
    occupy(g, 5.4, wy)
for th in [0.0, 0.5, 1.3, -2.0, np.pi]:
    obs = grid_to_obstacles(g, np.array([5., 5., th]), D_GATE, NS)
    on_wall = np.allclose(obs[:, 0], 5.425)
    inside = np.all(np.linalg.norm(obs - np.array([5., 5.]), axis=1) < D_GATE + 1e-9)
    check(f"theta = {th:5.2f}: {len(obs):2d} pts, all on the wall and in range",
          on_wall and inside)
# the looseness this costs is bounded by the in-wedge range spread
spread = 0.425 * (1 / np.cos(np.pi / NS) - 1)
T.info(f"worst-case extra slack from wedge rotation: {spread*1000:.1f} mm")

T.done()
