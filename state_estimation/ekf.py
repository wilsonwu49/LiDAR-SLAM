import numpy as np
from scipy.stats import chi2

from state import NominalState, IDX
from process_model import propagate_nominal, error_state_F, build_Q
from measurement_models import encoder_h_and_H
from kalman import kalman_update, inject_and_reset
from consistency import nis


class ESEKF:
    """Error-State EKF fusing BMI270 IMU (predict) with encoder + ICP (update).
    Holds the persistent nominal state and covariance between calls, so callers
    don't have to thread x/P through by hand.
    """

    def __init__(self, x0: NominalState, P0: np.ndarray, Q_params, R_enc: np.ndarray, R_icp: np.ndarray):
        """Store initial state/covariance and the noise parameters used by every predict/update call.
        x0: initial nominal state
        P0: initial covariance
        Q_params: sigma_a, sigma_g, sigma_ba, sigma_bg
        R_enc: encoder measurement noise covariance (3x3).
        R_icp: ICP measurement noise covariance (3x3, planar)
        """
        self.x = x0
        self.P = P0
        self.sigma_a, self.sigma_g, self.sigma_ba, self.sigma_bg = Q_params   # unpack however Q_params is shaped
        self.R_enc = R_enc
        self.R_icp = R_icp

    def predict(self, a_m: np.ndarray, w_m: np.ndarray, dt: float, g: np.ndarray) -> None:
        """Advance x and P forward by one IMU sample. Updates state.\n
        a_m: raw measured acceleration, body frame [m/s^2], shape (3,) \n
        w_m: raw measured angular velocity, body frame [rad/s], shape (3,) \n
        dt: time between IMU samples \n
        g: gravity vector
        """
        F = error_state_F(self.x, a_m, w_m, dt)
        Q = build_Q(self.sigma_a, self.sigma_g, self.sigma_ba, self.sigma_bg, dt)
        self.x = propagate_nominal(self.x, a_m, w_m, dt, g)
        self.P = F @ self.P @ F.T + Q

    def update_encoder(self, v_body_meas: np.ndarray) -> bool:
        """NIS-gate, then correct self.x/self.P from an encoder velocity reading.
        v_body_meas: measured body-frame velocity from the encoders, shape (3,).
        returns: True if the measurement was accepted and applied, False if it was rejected by the NIS gate.
        """
        h, H = encoder_h_and_H(self.x)
        y = v_body_meas - h
        S = H @ self.P @ H.T + self.R_enc

        if nis(y, S) > chi2.ppf(0.99, df=3):    # m=3 for encoder; 0.99 = single-step gate from §9.2
            return False
        
        dx_hat, P_plus = kalman_update(self.P, H, self.R_enc, y)
        self.x, self.P = inject_and_reset(self.x, dx_hat, P_plus)
        return True

    def update_icp(self, rel_pose_meas: np.ndarray) -> bool:
        """NIS-gate, then correct self.x/self.P from a Module-2 relative pose reading.
        rel_pose_meas: (Δx, Δy, Δyaw) from Module 2's scan matcher.
        returns: True if accepted and applied, False if rejected by the NIS gate.
        """
        pass

    def pose_with_covariance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the (x, y, yaw) pose estimate and its 3x3 covariance, for Module 4 (A*) and Module 5 (CBF-QP).
        returns:
            p_xy: world-frame (x, y) position, shape (2,).
            yaw: heading angle [rad], extracted from self.x.q.
            P_pose: 3x3 covariance over (x, y, yaw) — the honest uncertainty Module 5 inflates its safety margin with.
        """
        p_xy = self.x.p[:2]
        w = self.x.q[0]
        x = self.x.q[1]
        y = self.x.q[2]
        z = self.x.q[3]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        idx = [0, 1, 8]
        P_pose = self.P[np.ix_(idx, idx)]
        return p_xy, yaw, P_pose