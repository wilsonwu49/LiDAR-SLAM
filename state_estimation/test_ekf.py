import numpy as np
from state_estimation.state import NominalState
from state_estimation.ekf import ESEKF

# --- initial conditions: robot sitting still ---
x0 = NominalState(
    p=np.zeros(3), v=np.zeros(3),
    q=np.array([1.0, 0.0, 0.0, 0.0]),
    a_b=np.zeros(3), w_b=np.zeros(3),
)
P0 = np.eye(15) * 0.01

Q_params = (0.05, 0.01, 0.001, 0.001)   # sigma_a, sigma_g, sigma_ba, sigma_bg
R_enc = np.eye(3) * 0.01
R_icp = np.eye(3) * 0.01                 # unused for now, just needed by __init__

ekf = ESEKF(x0, P0, Q_params, R_enc, R_icp)

g = np.array([0.0, 0.0, -9.81])
a_m_still = np.array([0.0, 0.0, 9.81])   # accelerometer reading when stationary (cancels g exactly)
w_m_still = np.zeros(3)
dt = 0.01

# --- 1. predict-only: confirm state stays put, P grows ---
for _ in range(50):
    ekf.predict(a_m_still, w_m_still, dt, g)

assert np.allclose(ekf.x.p, 0, atol=1e-6), "position drifted under zero-motion input"
assert np.linalg.norm(ekf.x.q - np.array([1, 0, 0, 0])) < 1e-6, "attitude drifted under zero-motion input"
print(f"After 50 predicts: P[0,0] = {ekf.P[0,0]:.6f}  (should be > 0.01, the initial value)")

# --- 2. accepted encoder correction: near-true reading should pass the gate and shrink P ---
accepted = ekf.update_encoder(np.array([0.001, 0.0, 0.0]))
print(f"Plausible reading accepted: {accepted}  (should be True)")
assert accepted, "a small, plausible reading should NOT be NIS-rejected"
print(f"P[3,3] after correction = {ekf.P[3,3]:.6f}  (should have shrunk)")

# --- 3. rejected encoder correction: wild outlier should be blocked, state untouched ---
x_before, P_before = ekf.x, ekf.P.copy()
rejected = ekf.update_encoder(np.array([100.0, 100.0, 100.0]))
print(f"Wild outlier accepted: {rejected}  (should be False)")
assert not rejected, "a huge, implausible reading should be NIS-rejected"
assert ekf.x is x_before, "state object should be unchanged after a rejected update"
assert np.array_equal(ekf.P, P_before), "P should be unchanged after a rejected update"

# --- 4. pose_with_covariance sanity ---
p_xy, yaw, P_pose = ekf.pose_with_covariance()
print(f"p_xy={p_xy}, yaw={yaw:.6f}, P_pose shape={P_pose.shape}")
assert p_xy.shape == (2,)
assert P_pose.shape == (3, 3)
assert np.allclose(P_pose, P_pose.T), "P_pose should be symmetric"

print("ESEKF checkpoint PASSED")