from state import NominalState, IDX
import numpy as np
from quaternion import quat_to_rot, rotvec_to_quat, quat_mul, quat_normalize, skew

def propagate_nominal(x: NominalState, a_m: np.ndarray, w_m: np.ndarray, dt: float, g: np.ndarray) -> NominalState:
    """Integrate the nominal state forward by dt using one IMU sample.
    x: initial NominalState \n
    a_m: raw measured acceleration, body frame [m/s^2], shape (3,) \n
    w_m: raw measured angular velocity, body frame [rad/s], shape (3,) \n
    dt: time between IMU samples \n
    g: world frame gravity \n
    returns new NominalState one step forward.
    """
    R = quat_to_rot(x.q)

    a_true = R @ (a_m - x.a_b) + g
    new_p = x.p + x.v * dt + 0.5 * a_true * dt**2
    new_v = x.v + a_true * dt
    
    w_true = w_m - x.w_b
    phi = w_true * dt
    dq = rotvec_to_quat(phi)
    new_q = quat_normalize(quat_mul(x.q,dq))
    
    new_a_b = x.a_b
    new_w_b = x.w_b

    newState = NominalState(p = new_p,
                            v = new_v,
                            q = new_q,
                            a_b = new_a_b, 
                            w_b = new_w_b)
    return newState

def error_state_F(x: NominalState, a_m, w_m, dt) -> np.ndarray:
    """Return the 15x15 discrete error-state transition matrix F.\n
    returns F: new_error = F @ old_error"""

    F = np.zeros((15,15))
    F[IDX['dp'],IDX['dp']] = np.eye(3)
    F[IDX['dp'],IDX['dv']] = dt * np.eye(3)

    F[IDX['dv'],IDX['dv']] = np.eye(3)
    F[IDX['dv'],IDX['dth']] = -quat_to_rot(x.q) @ skew(a_m - x.a_b) * dt
    F[IDX['dv'],IDX['dab']] = -quat_to_rot(x.q) * dt

    F[IDX['dth'],IDX['dth']] = quat_to_rot(rotvec_to_quat(-(w_m - x.w_b) * dt))
    F[IDX['dth'],IDX['dwb']] = -dt * np.eye(3)

    F[IDX['dab'],IDX['dab']] = np.eye(3)
    F[IDX['dwb'],IDX['dwb']] = np.eye(3)
    
    return F

def build_Q(sigma_a, sigma_g, sigma_ba, sigma_bg, dt) -> np.ndarray:
    """Return the 15x15 block-diagonal discrete process-noise covariance:\n
    Q (15,15) added to P every predict step"""

    Q = np.zeros((15,15))
    Q[IDX['dv'],IDX['dv']] = sigma_a**2 * dt * np.eye(3)
    Q[IDX['dth'],IDX['dth']] = sigma_g**2 * dt * np.eye(3)
    Q[IDX['dab'],IDX['dab']] = sigma_ba**2 * dt * np.eye(3)
    Q[IDX['dwb'],IDX['dwb']] = sigma_bg**2 * dt * np.eye(3)

    return Q


