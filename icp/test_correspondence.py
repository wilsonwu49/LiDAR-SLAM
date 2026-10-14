import numpy as np
import time
from icp.correspondence import Correspondences

rng = np.random.default_rng(0)
ref = rng.uniform(-10, 10, size=(500, 2))
src = rng.uniform(-10, 10, size=(500, 2))

corr = Correspondences(ref)
idx, dist = corr.query(src, max_dist=np.inf)   # no gating, isolating pure NN correctness

# --- 1. cross-check against brute-force for a subset ---
for i in range(20):
    brute_idx = np.argmin(np.linalg.norm(ref - src[i], axis=1))
    assert idx[i] == brute_idx, f"mismatch at query {i}: tree={idx[i]} brute={brute_idx}"
print("First 20 queries match brute-force exactly.")

# --- 2. timing comparison ---
t0 = time.perf_counter()
Correspondences(ref).query(src, max_dist=np.inf)
t_tree = time.perf_counter() - t0

t0 = time.perf_counter()
[np.argmin(np.linalg.norm(ref - s, axis=1)) for s in src]
t_brute = time.perf_counter() - t0

print(f"KD-tree: {t_tree*1000:.3f} ms   Brute-force: {t_brute*1000:.3f} ms")
assert t_tree < t_brute, "expected KD-tree to be faster"
print("Section 3 checkpoint PASSED")