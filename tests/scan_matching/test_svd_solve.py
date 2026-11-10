import numpy as np
from lidar_nav.scan_matching.geometry import transform_points
from lidar_nav.scan_matching.svd_solve import rigid_transform_svd
from tests.testkit import Suite

T = Suite("ICP: point-to-point (SVD) solver")

T.section("noise-free points with known matches")
src = np.random.default_rng(0).uniform(-2, 2, size=(30, 2))
phi_star, t_star = 0.4, np.array([0.3, -0.15])
phi_hat, t_hat = rigid_transform_svd(src, transform_points(src, phi_star, t_star))
T.near("exact rotation", phi_hat, phi_star, 1e-9, " rad")
T.near("exact translation", t_hat, t_star, 1e-9, " m")

T.section("points on a single line")
x = np.linspace(-1, 1, 20)
line = np.column_stack((x, np.zeros(20)))
phi2, _ = rigid_transform_svd(line, transform_points(line, 0.2, np.array([0.1, 0.05])))
R = np.array([[np.cos(phi2), -np.sin(phi2)], [np.sin(phi2), np.cos(phi2)]])
T.check("returns a rotation, never a mirror image", np.linalg.det(R) > 0.99, f"det(R) = {np.linalg.det(R):.4f}")

T.done()
