import numpy as np 
from icp.geometry import transform_points

def point_to_point_cost(src: np.ndarray, dst: np.ndarray, phi: float, t: np.ndarray) -> float:
    """Sum of squared Euclidean residuals for given correspondences.
    src[i] corresponds to dst[i]. Transform is applied to src.
    """
    new_src = transform_points(src, phi, t)
    residuals = new_src - dst
    return np.sum(residuals ** 2)


def point_to_line_cost(src: np.ndarray, dst: np.ndarray, normals: np.ndarray, phi: float, t: np.ndarray) -> float:
    """Sum of squared *perpendicular* residuals: (n_i . (T(src_i) - dst_i))^2.
    normals[i] is the unit normal at dst[i].
    """
    new_src = transform_points(src, phi, t)
    residuals = np.sum(normals * (new_src - dst), axis=1)
    return np.sum(residuals ** 2)