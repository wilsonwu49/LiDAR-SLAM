import numpy as np
from lidar_nav.scan_matching.geometry import transform_points
from lidar_nav.scan_matching.costs import point_to_line_cost
from lidar_nav.scan_matching.ptl_solve import solve_point_to_line_step
from tests.testkit import Suite

T = Suite("ICP: point-to-line solver")

wall1 = np.column_stack((np.linspace(-1, 1, 50), np.zeros(50)))
wall2 = np.column_stack((np.zeros(50), np.linspace(0.05, 2, 50)))
dst = np.vstack([wall1, wall2])
normals = np.vstack([np.tile([0.0, 1.0], (50, 1)), np.tile([1.0, 0.0], (50, 1))])


def cost(pts):
    return point_to_line_cost(pts, dst, normals, 0.0, np.zeros(2))


T.section("one step on a small known misalignment")
src = transform_points(dst, 0.03, np.array([0.05, 0.02]))
d_phi, d_t = solve_point_to_line_step(src, dst, normals)
before, after = cost(src), cost(transform_points(src, d_phi, d_t))
T.check("one step removes almost all the error", after < before * 1e-3, f"cost {before:.1e} -> {after:.1e}")

T.section("repeated steps on a bigger misalignment")
src2 = transform_points(dst, 0.15, np.array([0.2, -0.1]))
costs = [cost(src2)]
for _ in range(5):
    d_phi, d_t = solve_point_to_line_step(src2, dst, normals)
    src2 = transform_points(src2, d_phi, d_t)
    costs.append(cost(src2))
T.check("cost never goes up", all(costs[i + 1] <= costs[i] + 1e-9 for i in range(5)),
        " -> ".join(f"{c:.1e}" for c in costs[::2]))

T.section("a single wall cannot pin down sliding along it")
nrm = np.tile([0.0, 1.0], (50, 1))
src3 = transform_points(wall1, 0.0, np.array([0.05, 0.0]))
A = np.array([nrm[:, 0], nrm[:, 1], -nrm[:, 0] * src3[:, 1] + nrm[:, 1] * src3[:, 0]]).T
eig = np.linalg.eigvalsh(A.T @ A)
with np.errstate(divide="ignore"):
    cond = eig[-1] / eig[0]
T.check("the solve is ill-conditioned along the wall", cond > 1e6,
        "exactly singular" if np.isinf(cond) else f"condition number {cond:.1e}")

T.done()
