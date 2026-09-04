import numpy as np
from state_estimation.state import NominalState, IDX
from state_estimation.quaternion import skew, quat_mul, rotvec_to_quat, quat_normalize


def kalman_update(P: np.ndarray, H: np.ndarray, R: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine a measurement with the current estimate to get an error correction and updated covariance. \n
    P: current error-state covariance (15,15) \n
    H: measurement Jacobian (measurement size, 15) \n
    R: measurement noise covariance \n
    y: difference between sensor measurement and predicted \n
    dx_hat: the estimated error state (15,) \n
    P_plus: updated covariance (15x15)
    """
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    dx = K @ y
    P_plus = (np.eye(15) - K @ H) @ P @ (np.eye(15) - K @ H).T + K @ R @ K.T
    P_plus = 0.5 * (P_plus + P_plus.T)

    return dx, P_plus


def inject_and_reset(x: NominalState, dx_hat: np.ndarray, P: np.ndarray) -> tuple[NominalState, np.ndarray]:
    """Fold the estimated error into the nominal state and reset the error state to zero. \n
    x: current nominal state \n
    dx_hat: error correction (15,) \n
    P: covariance/uncertainty in dx_hat (15x15)\n
    new_x: corrected NominalState \n
    P_reset: covariance after applying the reset Jacobian
    """
    G = np.zeros((15,15))
    G[IDX['dp'],IDX['dp']] = np.eye(3)
    G[IDX['dv'],IDX['dv']] = np.eye(3)
    G[IDX['dth'],IDX['dth']] = np.eye(3) - skew(0.5 * dx_hat[IDX['dth']])
    G[IDX['dab'],IDX['dab']] = np.eye(3)
    G[IDX['dwb'],IDX['dwb']] = np.eye(3)

    P = G @ P @ G.T
 
    new_x = NominalState(p = x.p + dx_hat[IDX['dp']],
                         v = x.v + dx_hat[IDX['dv']],
                         q = quat_normalize(quat_mul(x.q, rotvec_to_quat(dx_hat[IDX['dth']]))),
                         a_b = x.a_b + dx_hat[IDX['dab']],
                         w_b = x.w_b + dx_hat[IDX['dwb']])

    return new_x, P