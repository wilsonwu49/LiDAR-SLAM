import heapq
import itertools
import math
import numpy as np
from lidar_nav.planning.astar import heuristic, neighbors, astar
from tests.testkit import Suite

T = Suite("A*: search correctness")


def path_cost(path, connectivity=8):
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        total += math.sqrt(2) if (r0 != r1 and c0 != c1) else 1.0
    return total


def dijkstra_cost(start, goal, mask, connectivity=8):
    """Independent oracle: same graph, no heuristic."""
    dist = {start: 0.0}
    seen = set()
    ctr = itertools.count()
    q = [(0.0, next(ctr), start)]
    while q:
        d, _, cell = heapq.heappop(q)
        if cell in seen:
            continue
        if cell == goal:
            return d
        seen.add(cell)
        for nbr, w in neighbors(cell, mask, connectivity):
            if d + w < dist.get(nbr, math.inf):
                dist[nbr] = d + w
                heapq.heappush(q, (d + w, next(ctr), nbr))
    return None


T.section("3x3 grid, obstacle in the centre, 4-connected")
g = np.zeros((3, 3), dtype=bool)
g[1, 1] = True
p = astar((0, 0), (2, 2), g, connectivity=4)
T.check("goes around the obstacle in 4 steps", p is not None and abs(path_cost(p, 4) - 4.0) < 1e-9)
T.check("starts and ends in the right cells", p is not None and p[0] == (0, 0) and p[-1] == (2, 2))

T.section("guide example 9.7, where cutting corners is not allowed")
g = np.zeros((5, 5), dtype=bool)
for cell in [(1, 1), (1, 2), (1, 3), (2, 3), (3, 1), (3, 3), (4, 1)]:
    g[cell] = True
p = astar((0, 0), (4, 4), g, connectivity=8)
T.check("finds a path", p is not None)
if p:
    T.near("cost is exactly 8", path_cost(p), 8.0, 1e-9)
    T.check("never cuts a corner diagonally", all(r0 == r1 or c0 == c1 for (r0, c0), (r1, c1) in zip(p, p[1:])))
    T.check("never enters an obstacle", not any(g[c] for c in p))

T.section("edge cases")
T.check("start equal to goal returns just that cell", astar((2, 2), (2, 2), np.zeros((4, 4), bool)) == [(2, 2)])
boxed = np.zeros((5, 5), dtype=bool)
boxed[3, :] = True
T.check("walled-off goal returns no path", astar((0, 0), (4, 4), boxed) is None)
blocked = np.zeros((3, 3), dtype=bool)
blocked[2, 2] = True
T.check("goal inside an obstacle returns no path", astar((0, 0), (2, 2), blocked) is None)

T.section("200 random 12x12 grids against Dijkstra (which is optimal by construction)")
rng = np.random.default_rng(0)
reach, cost, through, jumps = [], [], [], []
for trial in range(200):
    m = rng.random((12, 12)) < 0.28
    m[0, 0] = m[11, 11] = False
    a = astar((0, 0), (11, 11), m, connectivity=8)
    d = dijkstra_cost((0, 0), (11, 11), m, connectivity=8)
    if (a is None) != (d is None):
        reach.append(trial)
    elif a is not None:
        if abs(path_cost(a) - d) > 1e-9:
            cost.append(trial)
        if any(m[c] for c in a):
            through.append(trial)
        if any(max(abs(r1 - r0), abs(c1 - c0)) != 1 for (r0, c0), (r1, c1) in zip(a, a[1:])):
            jumps.append(trial)
T.check("always agrees on whether a path exists", not reach, f"disagrees on {reach[:5]}" if reach else "")
T.check("always finds the shortest path", not cost, f"longer on {cost[:5]}" if cost else "")
T.check("never passes through an obstacle", not through)
T.check("every step moves to a neighbouring cell", not jumps)

T.done()
