import numpy as np

def solve_point_to_line_step(src: np.ndarray, dst: np.ndarray, normals: np.ndarray, weights: np.ndarray = None):
    """One step of point-to-line ICP.]\n
    src[i]     : current (already-transformed) source point, corresponds to dst[i].\n
    dst[i]     : reference point.\n
    normals[i] : unit normal at dst[i].\n
    weights[i] : optional per-correspondence weight (Section 7); default all ones.\n
    d_phi : float, incremental rotation.\n
    d_t   : (2,) incremental translation.
    """

    A = np.array([normals[:,0],normals[:,1],-normals[:,0]*src[:,1]+normals[:,1]*src[:,0]]).T
    B = np.sum(normals * (src - dst), axis = 1)
    
    W_A = weights[:,None] * A
    W_B = weights * B

    x = np.linalg.solve(A.T @ W_A, -A.T @ W_B)
    d_t = x[:2]
    d_phi = x[2]
    return d_phi, d_t