import numpy as np
from lidar_nav.estimation.state import NominalState
from lidar_nav.estimation.ekf import ESEKF
from tests.testkit import Suite

T = Suite("EKF: predict, update, gating")

x0 = NominalState(p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]),
                  a_b=np.zeros(3), w_b=np.zeros(3))
ekf = ESEKF(x0, np.eye(15) * 0.01, (0.05, 0.01, 0.001, 0.001), np.eye(3) * 0.01, np.eye(3) * 0.01)
g = np.array([0.0, 0.0, -9.81])
a_still, w_still, dt = np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.01

T.section("50 predict steps sitting still")
for _ in range(50):
    ekf.predict(a_still, w_still, dt, g)
T.near("position does not drift", ekf.x.p, np.zeros(3), 1e-6, " m")
T.check("heading does not drift", np.linalg.norm(ekf.x.q - [1, 0, 0, 0]) < 1e-6)
T.check("uncertainty grows with no measurements", ekf.P[0, 0] > 0.01, f"P[0,0] 0.0100 -> {ekf.P[0, 0]:.4f}")

T.section("encoder update")
P33 = ekf.P[3, 3]
T.check("a plausible reading is accepted", ekf.update_encoder(np.array([0.001, 0.0, 0.0])))
T.check("and shrinks the velocity uncertainty", ekf.P[3, 3] < P33, f"P[3,3] {P33:.4f} -> {ekf.P[3, 3]:.4f}")
x_before, P_before = ekf.x, ekf.P.copy()
T.check("a wild reading is rejected by the gate", not ekf.update_encoder(np.array([100.0, 100.0, 100.0])))
T.check("and leaves the state and P untouched", ekf.x is x_before and np.array_equal(ekf.P, P_before))

T.section("2D pose output")
p_xy, yaw, P_pose = ekf.pose_with_covariance()
T.check("returns x, y and a 3x3 covariance", p_xy.shape == (2,) and P_pose.shape == (3, 3))
T.check("covariance is symmetric", np.allclose(P_pose, P_pose.T))

T.done()
