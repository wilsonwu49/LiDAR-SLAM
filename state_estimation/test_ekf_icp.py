import numpy as np
from state_estimation.state import NominalState
from state_estimation.ekf import ESEKF

# --- true motion: constant forward velocity, no rotation, for a clean test ---
v_true = np.array([0.5, 0.0, 0.0])
g = np.array([0.0, 0.0, -9.81])
a_m = np.array([0.0, 0.0, 9.81])   # cancels gravity exactly -- consistent with constant velocity
w_m = np.zeros(3)
dt = 0.01

x0 = NominalState(
    p=np.zeros(3), v=v_true.copy(),
    q=np.array([1.0, 0.0, 0.0, 0.0]),
    a_b=np.zeros(3), w_b=np.zeros(3),
)
P0 = np.eye(15) * 0.01
Q_params = (0.05, 0.01, 0.001, 0.001)
R_enc = np.eye(3) * 0.001
R_icp = np.eye(3) * 0.0005

ekf = ESEKF(x0, P0, Q_params, R_enc, R_icp)
rng = np.random.default_rng(0)

x_prev_snapshot = None
n_steps = 200

for step in range(n_steps):
    ekf.predict(a_m, w_m, dt, g)

    if step == 0:
        x_prev_snapshot = ekf.x   # keyframe for the later ICP correction

    if step % 10 == 0 and step > 0:
        v_meas = v_true + rng.normal(0, 0.01, size=3)
        accepted_enc = ekf.update_encoder(v_meas)
        assert accepted_enc, f"encoder update unexpectedly rejected at step {step}"

    # sanity checks every step: state and covariance must stay valid
    assert abs(np.linalg.norm(ekf.x.q) - 1.0) < 1e-9, f"quaternion drifted at step {step}"
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-8), f"P not symmetric at step {step}"
    assert np.linalg.eigvalsh(ekf.P).min() > 0, f"P not positive-definite at step {step}"

# --- one ICP correction near the end, comparing current state to the early snapshot ---
elapsed = (n_steps - 1) * dt
true_dp = v_true[:2] * elapsed
rel_pose_meas = np.array([true_dp[0], true_dp[1], 0.0]) + rng.normal(0, 0.005, size=3)

accepted_icp = ekf.update_icp(rel_pose_meas, R_icp, x_prev_snapshot)
print(f"ICP update accepted: {accepted_icp}")
assert accepted_icp, "ICP update unexpectedly rejected"

p_xy, yaw, P_pose = ekf.pose_with_covariance()
print(f"final p_xy = {p_xy}  (true ~= {v_true[:2] * n_steps * dt})")
print(f"final yaw  = {yaw:.6f}  (true = 0.0)")
print(f"P_pose eigenvalues: {np.linalg.eigvalsh(P_pose)}")

print("Full ESEKF integration checkpoint PASSED")