import numpy as np
from state_estimation.state import NominalState, IDX
from state_estimation.quaternion import quat_to_rot, rotvec_to_quat, quat_normalize
from state_estimation.kalman import kalman_update, inject_and_reset

# def random_covariance(n, scale=1.0, rng=None):
#     A = rng.standard_normal((n, n)) * scale
#     return A @ A.T + 1e-6 * np.eye(n)   # sandwich trick — guaranteed valid, same proof as before

# rng = np.random.default_rng(0)
# P0 = random_covariance(15, scale=1.0, rng=rng)

# P_joseph = P0.copy()
# P_naive = P0.copy()
# n_steps = 100_000
# joseph_failed_at = None
# naive_failed_at = None

# for step in range(n_steps):
#     H = rng.standard_normal((3, 15)) * 0.5
#     R = random_covariance(3, scale=0.5, rng=rng)
#     y = rng.standard_normal(3) * 0.1

#     _, P_joseph = kalman_update(P_joseph, H, R, y)          # your function

#     S = H @ P_naive @ H.T + R                                 # naive form, computed by hand for comparison
#     K = np.linalg.solve(S, H @ P_naive).T
#     P_naive = (np.eye(15) - K @ H) @ P_naive

#     if joseph_failed_at is None and not (np.allclose(P_joseph, P_joseph.T, atol=1e-8)
#                                           and np.linalg.eigvalsh(P_joseph).min() > 0):
#         joseph_failed_at = step
#     if naive_failed_at is None and not (np.allclose(P_naive, P_naive.T, atol=1e-8)
#                                          and np.linalg.eigvalsh(P_naive).min() > 0):
#         naive_failed_at = step

# print(f"Joseph failed at step: {joseph_failed_at}")
# print(f"Naive failed at step: {naive_failed_at}")
# assert joseph_failed_at is None, "Joseph form broke — this should never happen"

# P = random_covariance(15, scale=0.1, rng=rng)
# H = rng.standard_normal((3, 15)) * 0.1
# R = random_covariance(3, scale=0.1, rng=rng)
# y = rng.standard_normal(3) * 0.1

# _, P_joseph = kalman_update(P, H, R, y)
# S = H @ P @ H.T + R
# K = np.linalg.solve(S, H @ P).T
# P_naive = (np.eye(15) - K @ H) @ P

# max_diff = np.max(np.abs(P_joseph - P_naive))
# print(f"Joseph vs naive max diff: {max_diff}")
# assert max_diff < 1e-9, "Disagreement bigger than expected — check K"

rng = np.random.default_rng(0)

x = NominalState(
    p=np.array([1.0, 2.0, 0.5]),
    v=np.array([0.3, -0.1, 0.05]),
    q=quat_normalize(rotvec_to_quat(np.array([0.1, 0.05, 0.2]))),
    a_b=np.array([0.01, -0.02, 0.01]),
    w_b=np.array([0.001, 0.002, -0.001]),
)

A = rng.standard_normal((15, 15))
P = A @ A.T + 1e-3 * np.eye(15)          # random valid covariance, sandwich trick
dx_hat = rng.standard_normal(15) * 0.01  # small, plausible correction

new_x, P_reset = inject_and_reset(x, dx_hat, P)

# Check 1: unit quaternion after renormalization
q_norm = np.linalg.norm(new_x.q)
print(f"||q|| = {q_norm}")
assert abs(q_norm - 1.0) < 1e-12, "quaternion not unit norm!"

# Check 2: P_reset is still a valid covariance (symmetric, positive-definite)
is_symmetric = np.allclose(P_reset, P_reset.T, atol=1e-8)
min_eig = np.linalg.eigvalsh(P_reset).min()
print(f"P_reset symmetric: {is_symmetric}, min eigenvalue: {min_eig}")
assert is_symmetric and min_eig > 0, "P_reset is not a valid covariance!"

# Check 3: the injection arithmetic itself is exactly what's expected
assert np.allclose(new_x.p, x.p + dx_hat[IDX['dp']])
assert np.allclose(new_x.v, x.v + dx_hat[IDX['dv']])
assert np.allclose(new_x.a_b, x.a_b + dx_hat[IDX['dab']])
assert np.allclose(new_x.w_b, x.w_b + dx_hat[IDX['dwb']])

print("inject_and_reset checkpoint PASSED")