import numpy as np
from lidar_nav.scan_matching.costs import point_to_point_cost, point_to_line_cost
from tests.testkit import Suite

T = Suite("ICP: cost functions")

x = np.linspace(-1, 1, 100)
dst = np.column_stack((x, np.ones(100)))
src = np.column_stack((x + 0.05, np.ones(100)))
normals = np.tile(np.array([0.0, 1.0]), (100, 1))

T.section("the same wall, slid 5 cm along itself")
pl = point_to_line_cost(src, dst, normals, 0.0, np.zeros(2))
pp = point_to_point_cost(src, dst, 0.0, np.zeros(2))
T.check("point-to-line ignores sliding along the wall", pl < 1e-12, f"cost {pl:.2e}")
T.check("point-to-point penalises it", pp > 1e-4, f"cost {pp:.2e}")

T.done()
