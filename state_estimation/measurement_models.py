from state_estimation.state import NominalState, IDX
import numpy as np
from state_estimation.quaternion import quat_to_rot, skew

def encoder_h_and_H(x: NominalState) -> tuple[np.ndarray, np.ndarray]:
    """Predicted encoder measurement. Only depends on velocity and attitude. \n
    h: predicted body-frame velocity reading \n
    H: measurement jacobian, (3,15), describes unit shift of h for each error component
    """
    H = np.zeros((3,15))
    H[:,IDX['dv']] = quat_to_rot(x.q).T
    H[:,IDX['dth']] = skew(quat_to_rot(x.q).T @ x.v)

    h = quat_to_rot(x.q).T @ x.v

    return h, H
