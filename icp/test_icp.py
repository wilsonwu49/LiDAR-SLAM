import numpy as np
from icp.geometry import transform_points
from icp.icp import icp_scan_match
from icp.icp_ekf import icp_measurement_for_ekf

phi_star, t_star = 0.05, np.array([0.05, 0.02])

# --- corner: two perpendicular walls, well-constrained in both directions ---
wall1 = np.column_stack((np.linspace(-1, 1, 60), np.zeros(60)))
wall2 = np.column_stack((np.zeros(60), np.linspace(0.05, 2, 60)))
corner_base = np.vstack([wall1, wall2])
corner_reference = transform_points(corner_base, phi_star, t_star)

phi_c, t_c, cov_c, info_c = icp_scan_match(corner_base, corner_reference)
eig_c = np.linalg.eigvalsh(cov_c)
print(f"corner:   phi={phi_c:.4f} (true {phi_star}), t={t_c} (true {t_star})")
print(f"corner:   cov eigenvalues = {eig_c}")

# --- corridor: single straight wall, degenerate along its own length ---
corridor_base = wall1.copy()
corridor_reference = transform_points(corridor_base, phi_star, t_star)

phi_r, t_r, cov_r, info_r = icp_scan_match(corridor_base, corridor_reference)
eig_r = np.linalg.eigvalsh(cov_r)
print(f"corridor: cov eigenvalues = {eig_r}")

assert eig_r.max() > eig_c.max() * 10, \
    "corridor covariance should show a much larger eigenvalue than the corner's"

print("Full ICP module checkpoint PASSED")