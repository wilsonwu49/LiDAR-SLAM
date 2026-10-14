import numpy as np
from astar.astar import heuristic, neighbors

# heuristic: exact on an empty grid
assert abs(heuristic((0,0), (3,3), 4) - 6.0) < 1e-9
assert abs(heuristic((0,0), (3,3), 8) - 4.2426) < 1e-3
assert abs(heuristic((0,0), (5,2), 8) - 5.8284) < 1e-3
assert heuristic((2,2), (2,2), 8) == 0

# neighbors: counts in open space and at a corner
free = np.zeros((5,5), dtype=bool)
assert len(neighbors((2,2), free, 8)) == 8
assert len(neighbors((2,2), free, 4)) == 4
assert len(neighbors((0,0), free, 8)) == 3
assert len(neighbors((0,0), free, 4)) == 2

# diagonal costs
costs = dict(neighbors((2,2), free, 8))
assert abs(costs[(3,3)] - 1.4142) < 1e-3
assert abs(costs[(3,2)] - 1.0) < 1e-9

# corner rule, straight from section 9.7
g = np.zeros((5,5), dtype=bool)
g[1,1] = g[1,2] = g[1,3] = True
g[2,3] = True
g[3,1] = g[3,3] = True
g[4,1] = True
nbrs = dict(neighbors((1,0), g, 8))
assert (2,1) not in nbrs, "diagonal past the obstacle at (1,1) must be forbidden"
assert (2,0) in nbrs

print("heuristic + neighbors PASSED")