import numpy as np
from dataclasses import dataclass

@dataclass
class NominalState:
    """
    Nominal state, NO covariance.
    p: world position [m], shape (3,)
    v: world velocity [m/s], shape (3,)
    q: body-to-world unit quaternion [w,x,y,z], shape (4,)
    a_b: body imu accelerometer bias [m/s^2], shape (3,)
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