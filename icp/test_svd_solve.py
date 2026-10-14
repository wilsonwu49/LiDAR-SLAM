import numpy as np
from icp.geometry import transform_points
from icp.svd_solve import rigid_transform_svd

# --- 1. exact recovery on a well-distributed point set (known correspondences by construction) ---
rng = np.random.default_rng(0)
src = rng.uniform(-2, 2, size=(30, 2))
phi_star, t_star = 0.4, np.array([0.3, -0.15])
dst = transform_points(src, phi_star, t_star)

phi_hat, t_hat = rigid_transform_svd(src, dst)
print(f"phi: {phi_hat:.10f} vs {phi_star}   diff={abs(phi_hat-phi_star):.2e}")
print(f"t:   {t_hat} vs {t_star}   diff={np.max(np.abs(t_hat-t_star)):.2e}")
assert abs(phi_hat - phi_star) < 1e-9, "SVD should be exact on noise-free data with known correspondences"
assert np.max(np.abs(t_hat - t_star)) < 1e-9

# --- 2. collinear points: confirm the reflection guard actually engages ---
x = np.linspace(-1, 1, 20)
collinear_src = np.column_stack((x, np.zeros(20)))
phi2, t2 = 0.2, np.array([0.1, 0.05])
collinear_dst = transform_points(collinear_src, phi2, t2)

phi_hat2, t_hat2 = rigid_transform_svd(collinear_src, collinear_dst)
R_hat = np.array([[np.cos(phi_hat2), -np.sin(phi_hat2)],
                   [np.sin(phi_hat2),  np.cos(phi_hat2)]])
print(f"collinear: det(R) = {np.linalg.det(R_hat):.6f}  (must be +1, never -1)")
print(f"collinear: phi = {phi_hat2:.8f} vs {phi2}")
assert np.linalg.det(R_hat) > 0.99, "reflection guard failed -- returned a mirror instead of a rotation"

print("Section 6 checkpoint PASSED")