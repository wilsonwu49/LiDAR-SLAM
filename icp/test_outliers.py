import numpy as np
from icp.geometry import transform_points
from icp.correspondence import Correspondences
from icp.normals import estimate_normals
from icp.ptl_solve import solve_point_to_line_step
from icp.icp import icp_scan_match

# --- build an L-corner reference shape ---
wall1 = np.column_stack((np.linspace(-1, 1, 60), np.zeros(60)))
wall2 = np.column_stack((np.zeros(60), np.linspace(0.05, 2, 60)))
base = np.vstack([wall1, wall2])

# --- known true motion ---
phi_star, t_star = 0.08, np.array([0.10, -0.05])
source_pts = base
reference_pts = transform_points(base, phi_star, t_star)

# --- corrupt 20% of the source points, simulating a person / spurious returns ---
rng = np.random.default_rng(1)
corrupted_source = source_pts.copy()
n_corrupt = int(0.2 * len(corrupted_source))
corrupt_idx = rng.choice(len(corrupted_source), size=n_corrupt, replace=False)
corrupted_source[corrupt_idx] += rng.uniform(-1.0, 1.0, size=(n_corrupt, 2))

# --- a stripped-down "plain" ICP for comparison: same pieces, no trimming/weighting ---
def plain_icp(source_pts, reference_pts, max_iterations=30, d_max=0.3, k_normals=8):
    normals, _ = estimate_normals(reference_pts, k=k_normals)
    corr = Correspondences(reference_pts)
    phi, t = 0.0, np.zeros(2)
    for _ in range(max_iterations):
        new_src = transform_points(source_pts, phi, t)
        idx, dist = corr.query(new_src, max_dist=d_max)
        valid = idx != -1
        matched_src = new_src[valid]
        matched_dst = reference_pts[idx[valid]]
        matched_normals = normals[idx[valid]]
        weights = np.ones(len(matched_src))          # deliberately no robustness
        d_phi, d_t = solve_point_to_line_step(matched_src, matched_dst, matched_normals, weights)
        R = np.array([[np.cos(d_phi), -np.sin(d_phi)], [np.sin(d_phi), np.cos(d_phi)]])
        t = R @ t + d_t
        phi = phi + d_phi
    return phi, t

phi_plain, t_plain = plain_icp(corrupted_source, reference_pts)
phi_robust, t_robust, cov, info = icp_scan_match(corrupted_source, reference_pts)

phi_err_plain  = abs(phi_plain  - phi_star)
phi_err_robust = abs(phi_robust - phi_star)
t_err_plain  = np.linalg.norm(t_plain  - t_star)
t_err_robust = np.linalg.norm(t_robust - t_star)

print(f"true:   phi={phi_star:.4f}  t={t_star}")
print(f"plain:  phi={phi_plain:.4f}  t={t_plain}   err(phi)={phi_err_plain:.4f}  err(t)={t_err_plain:.4f}")
print(f"robust: phi={phi_robust:.4f}  t={t_robust}   err(phi)={phi_err_robust:.4f}  err(t)={t_err_robust:.4f}")

assert phi_err_robust < 0.01 and t_err_robust < 0.02, "robust ICP should recover close to ground truth despite 20% outliers"
assert phi_err_plain > phi_err_robust or t_err_plain > t_err_robust, "plain ICP should visibly do worse than the robust version"

print("Section 7 checkpoint PASSED")