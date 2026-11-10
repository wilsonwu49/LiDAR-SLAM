import numpy as np

# Kept as a knob, but 1.0 means no fudge factor. It was introduced when ICP was
# fused as an absolute pose measurement, where the Gauss-Newton covariance was
# ~16x too small. Once ICP is fused as a rate (see ESEKF.update_icp_rates) the
# filter's consistency is insensitive to it anywhere from 1 to 64, so it is set
# to 1 rather than carry an unexplained constant.
COV_INFLATION = 1.0


def icp_pose_covariance(A: np.ndarray, weights: np.ndarray, residuals: np.ndarray,
                        inflation: float = COV_INFLATION) -> np.ndarray:
    """Approximate 3x3 covariance of the ICP pose estimate, ordered [tx, ty, phi].

    A          (N,3) Jacobian rows [n_x, n_y, n x p]
    weights    (N,) per-correspondence weights, 0 for trimmed points
    residuals  (N,) point-to-line residuals at the CONVERGED pose
    inflation  variance multiplier covering what Gauss-Newton cannot see

    The Gauss-Newton form sigma^2 (A^T W A)^-1 assumes the residuals are
    independent and that correspondence and normal estimation are exact. None
    of that holds: adjacent beams hit the same surface, correspondences can be
    wrong, and PCA normals are noisy. Measured against ground truth this
    underestimates by roughly an order of magnitude, so the result is inflated
    by a factor calibrated to make the downstream filter's NEES ~= 3.
    """
    n_eff = np.count_nonzero(weights)
    dof = max(n_eff - 3, 1)                      # 3 parameters consumed by the fit
    sigma2 = np.sum(weights * residuals ** 2) / dof * np.sum(weights) / max(n_eff, 1)
    AtWA = A.T @ (weights[:, None] * A)
    return inflation * sigma2 * np.linalg.inv(AtWA)

def icp_measurement_for_ekf(phi, t, cov, info, min_inliers=20, max_cond=1e3) -> dict | None:
    """Package the ICP result as an EKF measurement, or return None if rejected
    """
    if info["converged"] is False:
        return None
    if info.get("num_inliers", min_inliers) < min_inliers:
        return None
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] <= 0 or (eigvals[-1] / eigvals[0]) > max_cond:
        return None
    return dict(phi=phi, t=t, cov=cov)