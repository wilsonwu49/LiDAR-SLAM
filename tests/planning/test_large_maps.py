import time

import numpy as np
from lidar_nav.planning.astar import astar
from tests.testkit import Suite

T = Suite("A*: speed on large maps")

T.section("unreachable goal (worst case: A* must exhaust the reachable half)")
for n in (200, 400, 800):
    m = np.zeros((n, n), dtype=bool)
    m[n // 2, :] = True
    t = time.perf_counter()
    path = astar((0, 0), (n - 1, n - 1), m)
    dt = time.perf_counter() - t
    T.check(f"{n}x{n} map returns no path", path is None, f"{dt:.2f} s")

T.done()
