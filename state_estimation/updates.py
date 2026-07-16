import numpy as np
from state import NominalState
from measurement_models import encoder_h_and_H
from kalman import kalman_update, inject_and_reset

def update_encoder(x: NominalState, P: np.ndarray, v_body_meas: np.ndarray, R_enc: np.ndarray) -> tuple[NominalState, np.ndarray]:
    """EKF correction from a body-frame velocity measurement \n
    x: current nominal state \n
    P: current error-state covariance (15,15) \n
    v_body_meas: measured body-frame velocity from the encoders, (3,) \n
    R_enc: encoder measurement noise covariance (3,3) \n
    returns: new_x, P_reset - corrected state and covariance
    """
    h, H = encoder_h_and_H(x)
    y = v_body_meas - h # error
    dx_hat, P_plus = kalman_update(P, H, R_enc, y)
    new_x, P_reset = inject_and_reset(x, dx_hat, P_plus)
    return new_x, P_reset

# def update_icp(x_prev: NominalState, x_now: NominalState, P: np.ndarray, rel_pose_meas: np.ndarray, R_icp: np.ndarray) -> tuple[NominalState, np.ndarray]: