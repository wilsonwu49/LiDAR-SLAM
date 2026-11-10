import numpy as np

def rigid_transform_svd(src: np.ndarray, dst: np.ndarray):
    """Closed-form least-squares SE(2) point-to-point fit (Arun et al. 1987).
    src[i] corresponds to dst[i]. Returns (phi, t) with a reflection guard.
    Must handle the det(R) = -1 case correctly.
    """
    src_mean = np.mean(src, axis = 0)
    dst_mean = np.mean(dst, axis = 0)
    centered_src = src - src_mean
    centered_dst = dst - dst_mean
    covariance = centered_src.T @ centered_dst

    U, S, Vt = np.linalg.svd(covariance)
    V = Vt.T
    R = V @ U.T

    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    phi = np.arctan2(R[1, 0], R[0, 0])
    t = dst_mean - R @ src_mean
    return phi, t