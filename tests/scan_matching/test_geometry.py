import numpy as np
from lidar_nav.scan_matching.geometry import polar_scan_to_xy, transform_points
from tests.testkit import Suite

T = Suite("ICP: transforms and range gating")

T.section("transform round trip")
rng = np.random.default_rng(0)
P = rng.uniform(-5, 5, size=(20, 2))
phi_star, t_star = 0.05, np.array([0.10, -0.03])
Q = transform_points(P, phi_star, t_star)
R_inv = np.array([[np.cos(-phi_star), -np.sin(-phi_star)],
                  [np.sin(-phi_star), np.cos(-phi_star)]])
P_back = transform_points(Q, -phi_star, -R_inv @ t_star)
err = np.max(np.abs(P_back - P))
T.check("forward then inverse returns the original points", err < 1e-9, f"max error {err:.1e}")

T.section("bad beams are dropped")
angles = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
ranges = np.array([1.0, np.inf, np.nan, 0.0, 3.0])
pts = polar_scan_to_xy(angles, ranges, range_min=0.1, range_max=10.0)
T.check("no inf or nan survives", np.all(np.isfinite(pts)))
T.near("inf, nan and zero-range beams removed", len(pts), 2, 0)

T.done()
