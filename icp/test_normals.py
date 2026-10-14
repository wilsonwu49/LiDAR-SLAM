import numpy as np
from icp.normals import estimate_normals

# --- 1. straight wall: y = 2.0 ---
x = np.linspace(-2, 2, 100)
wall_pts = np.column_stack((x, np.full(100, 2.0)))
normals, planarity = estimate_normals(wall_pts, k=8)

# allow sign flip: check |dot(normal, [0,1])| close to 1
alignment = np.abs(normals @ np.array([0.0, 1.0]))
print(f"wall: min alignment = {alignment.min():.4f} (expect > ~0.996, i.e. within a few degrees)")
print(f"wall: mean planarity = {planarity.mean():.4f} (expect close to 1)")
assert alignment.min() > 0.99
assert planarity.mean() > 0.9

# --- 2. circle of radius 1, centered at origin ---
theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
circle_pts = np.column_stack((np.cos(theta), np.sin(theta)))
normals_c, planarity_c = estimate_normals(circle_pts, k=8)

# expected normal at each point is the radial direction: point / ||point|| = point itself here (radius 1)
radial = circle_pts / np.linalg.norm(circle_pts, axis=1, keepdims=True)
alignment_c = np.abs(np.sum(normals_c * radial, axis=1))
print(f"circle: min radial alignment = {alignment_c.min():.4f} (expect > ~0.98)")
assert alignment_c.min() > 0.97

# --- 3. random blob: no structure ---
rng = np.random.default_rng(0)
blob_pts = rng.uniform(-1, 1, size=(100, 2))
_, planarity_b = estimate_normals(blob_pts, k=8)
print(f"blob: mean planarity = {planarity_b.mean():.4f} (expect well below wall's ~1)")
assert planarity_b.mean() < 0.7

print("Section 4 checkpoint PASSED")