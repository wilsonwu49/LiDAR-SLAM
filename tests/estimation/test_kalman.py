import numpy as np
from lidar_nav.estimation.state import NominalState, IDX
from lidar_nav.estimation.quaternion import quat_to_rot, rotvec_to_quat, quat_normalize
from lidar_nav.estimation.kalman import kalman_update, inject_and_reset

from tests.testkit import Suite

T = Suite("EKF: injecting a correction")

rng = np.random.default_rng(0)
x = NominalState(p=np.array([1.0, 2.0, 0.5]), v=np.array([0.3, -0.1, 0.05]),
                 q=quat_normalize(rotvec_to_quat(np.array([0.1, 0.05, 0.2]))),
                 a_b=np.array([0.01, -0.02, 0.01]), w_b=np.array([0.001, 0.002, -0.001]))
A = rng.standard_normal((15, 15))
P = A @ A.T + 1e-3 * np.eye(15)
dx_hat = rng.standard_normal(15) * 0.01
new_x, P_reset = inject_and_reset(x, dx_hat, P)

T.section("after injecting a small random correction")
T.near("orientation stays a unit quaternion", np.linalg.norm(new_x.q), 1.0, 1e-12)
min_eig = np.linalg.eigvalsh(P_reset).min()
T.check("P stays a valid covariance", np.allclose(P_reset, P_reset.T, atol=1e-8) and min_eig > 0,
        f"smallest eigenvalue {min_eig:.2e}")
T.near("position gets the correction added", new_x.p, x.p + dx_hat[IDX['dp']], 1e-12)
T.near("velocity gets the correction added", new_x.v, x.v + dx_hat[IDX['dv']], 1e-12)
T.near("accel bias gets the correction added", new_x.a_b, x.a_b + dx_hat[IDX['dab']], 1e-12)
T.near("gyro bias gets the correction added", new_x.w_b, x.w_b + dx_hat[IDX['dwb']], 1e-12)

T.done()
