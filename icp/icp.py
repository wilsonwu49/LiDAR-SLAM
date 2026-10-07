import numpy as np
from icp.geometry import transform_points
from icp.correspondence import Correspondences
from icp.normals import estimate_normals
from icp.costs import point_to_line_cost
from icp.ptl_solve import solve_point_to_line_step
from icp.outliers import trim_correspondences, robust_weights
from icp.icp_ekf import icp_pose_covariance


def icp_scan_match(source_pts, reference_pts, init_phi=0.0, init_t=(0.0, 0.0),
                   max_iterations=30, tol_t=1e-3, tol_phi=1e-4, d_max=0.3, k_normals=8):
    """Full point-to-line ICP for two 2D scans.

    Returns
    -------
    phi, t          : estimated relative pose (reference <- source)
    cov             : 3x3 pose covariance estimate
    info            : dict(iterations, final_cost, num_inliers, converged, condition_number)
    """
    normals, planarity = estimate_normals(reference_pts, k=k_normals) # estimate normals using reference points
    corr = Correspondences(reference_pts) # create ckd tree once using reference

    # initial variables
    phi, t = init_phi, np.array(init_t, dtype=float)
    prev_cost = None
    converged = False
    info_iterations = 0

    for iterations in range(max_iterations):
        new_src = transform_points(source_pts, phi, t)
        idx, dist = corr.query(new_src, max_dist=d_max)

        valid = idx != -1   

        matched_src = new_src[valid]
        matched_dst = reference_pts[idx[valid]]
        matched_normals = normals[idx[valid]]
        residuals = np.sum(matched_normals * (matched_src - matched_dst), axis=1)

        weights = trim_correspondences(residuals, 0.8) # or robust_weights

        d_phi, d_t = solve_point_to_line_step(matched_src, matched_dst, matched_normals, weights)

        R_dphi = np.array([[np.cos(d_phi), -np.sin(d_phi)],
                            [np.sin(d_phi),  np.cos(d_phi)]])
        t = R_dphi @ t + d_t
        phi = phi + d_phi

        cost = point_to_line_cost(matched_src, matched_dst, matched_normals, 0.0, np.zeros(2))
        info_iterations = iterations + 1
        if np.linalg.norm(d_t) < tol_t and abs(d_phi) < tol_phi:
            converged = True
            break
        if prev_cost is not None and abs(cost - prev_cost) / prev_cost < 1e-6:
            converged = True
            break
        prev_cost = cost

    A = np.array([matched_normals[:,0], matched_normals[:,1],
                  -matched_normals[:,0]*matched_src[:,1] + matched_normals[:,1]*matched_src[:,0]]).T
    AtWA = A.T @ (weights[:, None] * A)
    eigvals = np.linalg.eigvalsh(AtWA)
    condition_number = eigvals[-1] / eigvals[0]
    num_inliers = int(np.count_nonzero(weights))

    cov = icp_pose_covariance(A, weights, residuals)

    info = dict(iterations=info_iterations, final_cost=cost, converged=converged, num_inliers=num_inliers, condition_number=condition_number)
    return phi, t, cov, info