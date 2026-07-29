import numpy as np

def icp_pose_covariance(A: np.ndarray, weights: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Approximate 3x3 covariance of the ICP pose estimate
    """
    sigma2 = np.sum(weights * residuals**2) / np.sum(weights)   
    AtWA = A.T @ (weights[:, None] * A) 
    return sigma2 * np.linalg.inv(AtWA)

def icp_measurement_for_ekf(phi, t, cov, info, min_inliers=20, max_cond=1e8) -> dict | None:
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