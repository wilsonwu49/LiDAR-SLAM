import numpy as np
from dataclasses import dataclass
from state_estimation.quaternion import quat_to_rot

@dataclass
class NominalState:
    """
    Nominal state: \n
    p: world position [m], shape (3,) \n
    v: world velocity [m/s], shape (3,) \n
    q: body-to-world unit quaternion [w,x,y,z], shape (4,) \n
    a_b: body imu accelerometer bias [m/s^2], shape (3,) \n
    w_b: body imu gyroscope bias [rad/s], shape (3,)
    """

    p: np.ndarray 
    v: np.ndarray
    q: np.ndarray
    a_b: np.ndarray
    w_b: np.ndarray

# Error State
IDX = dict(dp=slice(0,3), 
           dv=slice(3,6), 
           dth=slice(6,9), 
           dab=slice(9,12), 
           dwb=slice(12,15))

POSE2D_IDX = [0, 1, 8]        # x, y, yaw

def pose2d_from_ekf(x, P15):
    """Map-frame [x, y, theta] and its 3x3 covariance, sliced from the 15-dim filter.

    x     NominalState
    P15   (15,15) error-state covariance
    """
    Rwb = quat_to_rot(x.q)
    theta = np.arctan2(Rwb[1, 0], Rwb[0, 0])
    pose = np.array([x.p[0], x.p[1], theta])
    P3 = P15[np.ix_(POSE2D_IDX, POSE2D_IDX)]
    return pose, P3