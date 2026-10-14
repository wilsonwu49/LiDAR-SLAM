import time, numpy as np
from astar.astar import astar

for n in (200, 400, 800):
    m = np.zeros((n, n), dtype=bool)
    m[n//2, :] = True                      # wall across the middle, goal unreachable
    t = time.perf_counter()
    astar((0, 0), (n-1, n-1), m)
    print(n, "unreachable:", round(time.perf_counter() - t, 3), "s")