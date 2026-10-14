import heapq
import itertools
import math
import numpy as np
from astar.astar import heuristic, neighbors, astar


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


ok = True

# --- 3x3, obstacle in the middle, 4-connected
g = np.zeros((3, 3), dtype=bool)
g[1, 1] = True
p = astar((0, 0), (2, 2), g, connectivity=4)
if p is None or abs(path_cost(p, 4) - 4.0) > 1e-9:
    print("FAIL 3x3:", p); ok = False
if p[0] != (0, 0) or p[-1] != (2, 2):
    print("FAIL 3x3 endpoints:", p); ok = False

# --- section 9.7 grid, corner rule active
g = np.zeros((5, 5), dtype=bool)
for cell in [(1,1),(1,2),(1,3),(2,3),(3,1),(3,3),(4,1)]:
    g[cell] = True
p = astar((0, 0), (4, 4), g, connectivity=8)
if p is None:
    print("FAIL 9.7: no path found"); ok = False
else:
    cost = path_cost(p)
    if abs(cost - 8.0) > 1e-9:
        print(f"FAIL 9.7 cost {cost:.3f}, expected 8.000"); ok = False
    for (r0, c0), (r1, c1) in zip(p, p[1:]):
        if r0 != r1 and c0 != c1:
            print(f"FAIL 9.7 used a diagonal {(r0,c0)}->{(r1,c1)}"); ok = False
    if any(g[cell] for cell in p):
        print("FAIL 9.7 path enters an obstacle"); ok = False

# --- degenerate and impossible cases
free = np.zeros((4, 4), dtype=bool)
if astar((2, 2), (2, 2), free) != [(2, 2)]:
    print("FAIL start == goal"); ok = False

boxed = np.zeros((5, 5), dtype=bool)
boxed[3, :] = True
if astar((0, 0), (4, 4), boxed) is not None:
    print("FAIL walled-off goal should be None"); ok = False

blocked = np.zeros((3, 3), dtype=bool)
blocked[2, 2] = True
if astar((0, 0), (2, 2), blocked) is not None:
    print("FAIL lethal goal should be None"); ok = False

# --- optimality oracle: A* must match Dijkstra on random grids
rng = np.random.default_rng(0)
mismatches = 0
for trial in range(200):
    m = rng.random((12, 12)) < 0.28
    m[0, 0] = m[11, 11] = False
    a = astar((0, 0), (11, 11), m, connectivity=8)
    d = dijkstra_cost((0, 0), (11, 11), m, connectivity=8)
    if (a is None) != (d is None):
        print(f"FAIL trial {trial}: reachability disagrees"); ok = False; mismatches += 1
    elif a is not None:
        ca = path_cost(a)
        if abs(ca - d) > 1e-9:
            print(f"FAIL trial {trial}: A* {ca:.4f} vs Dijkstra {d:.4f}"); ok = False; mismatches += 1
        if any(m[cell] for cell in a):
            print(f"FAIL trial {trial}: path crosses an obstacle"); ok = False
        for (r0, c0), (r1, c1) in zip(a, a[1:]):
            if max(abs(r1 - r0), abs(c1 - c0)) != 1:
                print(f"FAIL trial {trial}: path not contiguous"); ok = False

print("Module 4 core", "PASSED" if ok else "FAILED")