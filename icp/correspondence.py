import numpy as np
from scipy.spatial import cKDTree

class Correspondences:
    def __init__(self, reference_pts: np.ndarray):
        """Build a KD-tree over the reference scan once."""
        self.tree = cKDTree(reference_pts)
        self.ref = reference_pts

    def query(self, src_pts: np.ndarray, max_dist: float):
        """For each src point, return (index_in_reference, distance) of its
        nearest reference point. Correspondences with distance > max_dist must
        be marked invalid (index = -1) so the caller can reject them (§7).

        Returns
        -------
        idx  : (M,) int, -1 where rejected.
        dist : (M,) float.
        """
        dist, idx = self.tree.query(src_pts)
        idx[dist > max_dist] = -1
        return idx, dist