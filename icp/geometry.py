import numpy as np

def polar_scan_to_xy(angles: np.ndarray, ranges: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    """Convert a raw 2D LiDAR scan to Cartesian points in the sensor frame.
    returns: (M, 2) array of [x, y] points, M <= N after gating.
    """
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])
    return np.column_stack((x, y))

def transform_points(pts: np.ndarray, phi: float, t: np.ndarray) -> np.ndarray:
    """Apply an SE(2) transform (rotation phi, translation t=[tx,ty]) to (M,2) pts.
    """
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return pts @ R.T + t
    