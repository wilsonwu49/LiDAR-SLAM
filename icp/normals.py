import numpy as np
from scipy.spatial import cKDTree

def estimate_normals(pts: np.ndarray, k: int = 8):
    """Estimate a unit normal and a 'planarity' score per point for k nearest points via local PCA.
    """

    tree = cKDTree(pts)
    dists, neighbor_idx = tree.query(pts, k=k+1) # includes itself 
    neighbor_idx = neighbor_idx[:, 1:] 

    normals = np.zeros((len(pts), 2))
    planarity = np.zeros(len(pts))
    
    for i in range(len(pts)):
        neighbors = pts[neighbor_idx[i]]
        centered = neighbors - neighbors.mean(axis = 0)
        C = (centered.T @ centered) / k

        eigvals, eigvecs = np.linalg.eigh(C)
        normals[i] = eigvecs[:, 0]
        planarity[i] = (eigvals[1] - eigvals[0]) / (eigvals[1] + eigvals[0])

    return normals, planarity