import numpy as np
from lidar_nav.scan_matching.normals import estimate_normals
from tests.testkit import Suite

T = Suite("ICP: surface normals")

T.section("straight wall")
x = np.linspace(-2, 2, 100)
normals, planarity = estimate_normals(np.column_stack((x, np.full(100, 2.0))), k=8)
align = np.abs(normals @ np.array([0.0, 1.0]))
T.check("every normal points straight off the wall", align.min() > 0.99, f"worst alignment {align.min():.4f}")
T.check("wall reads as flat", planarity.mean() > 0.9, f"planarity {planarity.mean():.3f}")

T.section("circle")
th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
circle = np.column_stack((np.cos(th), np.sin(th)))
normals_c, _ = estimate_normals(circle, k=8)
align_c = np.abs(np.sum(normals_c * circle, axis=1))
T.check("normals point radially outward", align_c.min() > 0.97, f"worst alignment {align_c.min():.4f}")

T.section("random blob")
_, planarity_b = estimate_normals(np.random.default_rng(0).uniform(-1, 1, size=(100, 2)), k=8)
T.check("scattered points do not read as flat", planarity_b.mean() < 0.7, f"planarity {planarity_b.mean():.3f}")

T.done()
