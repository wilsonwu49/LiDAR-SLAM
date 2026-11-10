import numpy as np
from lidar_nav.planning.astar import heuristic, neighbors
from tests.testkit import Suite

T = Suite("A*: heuristic and neighbours")

T.section("heuristic is exact on an empty grid")
T.near("4-connected Manhattan, (0,0) to (3,3)", heuristic((0, 0), (3, 3), 4), 6.0, 1e-9)
T.near("8-connected octile, (0,0) to (3,3)", heuristic((0, 0), (3, 3), 8), 4.2426, 1e-3)
T.near("8-connected octile, (0,0) to (5,2)", heuristic((0, 0), (5, 2), 8), 5.8284, 1e-3)
T.near("zero at the goal", heuristic((2, 2), (2, 2), 8), 0.0, 0.0)

T.section("neighbour counts")
free = np.zeros((5, 5), dtype=bool)
T.near("open cell, 8-connected", len(neighbors((2, 2), free, 8)), 8, 0)
T.near("open cell, 4-connected", len(neighbors((2, 2), free, 4)), 4, 0)
T.near("corner cell, 8-connected", len(neighbors((0, 0), free, 8)), 3, 0)
T.near("corner cell, 4-connected", len(neighbors((0, 0), free, 4)), 2, 0)

T.section("step costs")
costs = dict(neighbors((2, 2), free, 8))
T.near("diagonal step costs sqrt(2)", costs[(3, 3)], 1.4142, 1e-3)
T.near("straight step costs 1", costs[(3, 2)], 1.0, 1e-9)

T.section("no cutting corners past obstacles")
g = np.zeros((5, 5), dtype=bool)
g[1, 1] = g[1, 2] = g[1, 3] = True
g[2, 3] = True
g[3, 1] = g[3, 3] = True
g[4, 1] = True
nbrs = dict(neighbors((1, 0), g, 8))
T.check("diagonal past the obstacle at (1,1) is forbidden", (2, 1) not in nbrs)
T.check("straight step down is still allowed", (2, 0) in nbrs)

T.done()
