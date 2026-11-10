import numpy as np
from lidar_nav.estimation.state import NominalState
from lidar_nav.estimation.ekf import ESEKF
from tests.testkit import Suite

T = Suite("EKF: long run with encoder and a scan-match fix")

v_true = np.array([0.5, 0.0, 0.0])
g, a_m, w_m, dt = np.array([0.0, 0.0, -9.81]), np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.01
x0 = NominalState(p=np.zeros(3), v=v_true.copy(), q=np.array([1.0, 0.0, 0.0, 0.0]),
                  a_b=np.zeros(3), w_b=np.zeros(3))
R_icp = np.eye(3) * 0.0005
ekf = ESEKF(x0, np.eye(15) * 0.01, (0.05, 0.01, 0.001, 0.001), np.eye(3) * 0.001, R_icp)
rng = np.random.default_rng(0)

n_steps, x_first = 200, None
rejected, bad_q, bad_sym, bad_pd = [], [], [], []
for step in range(n_steps):
    ekf.predict(a_m, w_m, dt, g)
    if step == 0:
        x_first = ekf.x
    if step % 10 == 0 and step > 0 and not ekf.update_encoder(v_true + rng.normal(0, 0.01, 3)):
        rejected.append(step)
    if abs(np.linalg.norm(ekf.x.q) - 1.0) >= 1e-9:
        bad_q.append(step)
    if not np.allclose(ekf.P, ekf.P.T, atol=1e-8):
        bad_sym.append(step)
    if np.linalg.eigvalsh(ekf.P).min() <= 0:
        bad_pd.append(step)

T.section(f"{n_steps} steps at 0.5 m/s, encoder every 10")
T.check("every encoder reading accepted", not rejected, f"rejected at {rejected[:5]}" if rejected else "")
T.check("orientation stays a unit quaternion", not bad_q)
T.check("P stays symmetric", not bad_sym)
T.check("P stays positive-definite", not bad_pd)

T.section("scan-match position fix (the old position-style update, kept for comparison)")
true_dp = v_true[:2] * (n_steps - 1) * dt
meas = np.array([true_dp[0], true_dp[1], 0.0]) + rng.normal(0, 0.005, 3)
T.check("fix is accepted", ekf.update_icp(meas, R_icp, x_first))
p_xy, yaw, _ = ekf.pose_with_covariance()
T.near("final position matches the true motion", p_xy, v_true[:2] * n_steps * dt, 0.02, " m")

T.done()
