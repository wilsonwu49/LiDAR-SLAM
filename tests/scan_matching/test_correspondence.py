import time

import numpy as np
from lidar_nav.scan_matching.correspondence import Correspondences
from tests.testkit import Suite

T = Suite("ICP: nearest-neighbour correspondences")

rng = np.random.default_rng(0)
ref = rng.uniform(-10, 10, size=(500, 2))
src = rng.uniform(-10, 10, size=(500, 2))
idx, dist = Correspondences(ref).query(src, max_dist=np.inf)

T.section("correctness against brute force")
brute = [int(np.argmin(np.linalg.norm(ref - src[i], axis=1))) for i in range(20)]
bad = [i for i in range(20) if idx[i] != brute[i]]
T.check("KD-tree matches brute force on 20 queries", not bad,
        f"mismatches at {bad}" if bad else "20/20 identical")

T.section("speed")
t0 = time.perf_counter()
Correspondences(ref).query(src, max_dist=np.inf)
t_tree = time.perf_counter() - t0
t0 = time.perf_counter()
[np.argmin(np.linalg.norm(ref - s, axis=1)) for s in src]
t_brute = time.perf_counter() - t0
T.check("KD-tree is faster than brute force", t_tree < t_brute,
        f"{t_tree * 1000:.2f} ms vs {t_brute * 1000:.2f} ms")

T.done()
