import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def quat_mul(a, b) -> np.ndarray:
    aw = a[0]
    av = a[1:4]
    bw = b[0]
    bv = b[1:4]
    w = aw * bw - np.dot(av,bv)
    vec = aw * bv + bw * av + np.cross(av,bv)
    return np.array([w, vec[0], vec[1], vec[2]])

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = [[1-2*(y**2+z**2),   2*(x*y-w*z),     2*(x*z+w*y)],
         [2*(x*y+w*z),       1-2*(x**2+z**2), 2*(y*z-w*x)],
         [2*(x*z-w*y),       2*(y*z+w*x),     1-2*(x**2+y**2)]]
    return np.array(R)

def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(r)
    if abs(theta) < 0.05:
        return np.array([1, r[0]/2, r[1]/2, r[2]/2])
    else:
        axis = r / theta
        w = np.cos(theta/2)
        xyz = axis * np.sin(theta/2)
        return np.array([w, xyz[0], xyz[1], xyz[2]])

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return (q / n)

def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))