import numpy as np
from lidar_nav.scan_matching.geometry import transform_points
from lidar_nav.scan_matching.icp import icp_scan_match
from tests.testkit import Suite

T = Suite("ICP: full scan match")

phi_star, t_star = 0.05, np.array([0.05, 0.02])
wall1 = np.column_stack((np.linspace(-1, 1, 60), np.zeros(60)))
wall2 = np.column_stack((np.zeros(60), np.linspace(0.05, 2, 60)))

T.section("corner: two walls pin down every direction")
corner = np.vstack([wall1, wall2])
phi_c, t_c, cov_c, _ = icp_scan_match(corner, transform_points(corner, phi_star, t_star))
T.near("recovers the rotation", phi_c, phi_star, 1e-3, " rad")
T.near("recovers the translation", t_c, t_star, 2e-3, " m")

T.section("corridor: one wall, sliding along it is invisible")
_, _, cov_r, _ = icp_scan_match(wall1, transform_points(wall1, phi_star, t_star))
eig_c, eig_r = np.linalg.eigvalsh(cov_c), np.linalg.eigvalsh(cov_r)
T.check("corridor reports far more uncertainty than the corner",
        eig_r.max() > eig_c.max() * 10, f"largest eigenvalue {eig_r.max():.1e} vs {eig_c.max():.1e}")

T.done()
