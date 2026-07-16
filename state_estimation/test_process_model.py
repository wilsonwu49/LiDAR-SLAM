import numpy as np
from state import NominalState, IDX
from process_model import propagate_nominal, quat_mul, rotvec_to_quat, quat_normalize, error_state_F, build_Q
from measurement_models import encoder_h_and_H


### TESTING PROPAGATE NOMINAL
g = np.array([0.0, 0.0, -9.81])

# --- Test 1: zero motion, 1000 steps ---
x = NominalState(p=np.zeros(3), v=np.zeros(3),
                  q=np.array([1.0, 0.0, 0.0, 0.0]),
                  a_b=np.zeros(3), w_b=np.zeros(3))
dt = 0.005
a_m = np.array([0.0, 0.0, 9.81])
w_m = np.zeros(3)

for _ in range(1000):
    x = propagate_nominal(x, a_m, w_m, dt, g)

assert np.allclose(x.p, 0, atol=1e-6), f"FAIL: p drifted, {x.p}"
assert np.allclose(x.v, 0, atol=1e-6), f"FAIL: v drifted, {x.v}"
assert np.allclose(x.q, [1,0,0,0], atol=1e-6), f"FAIL: q drifted, {x.q}"
print("Test 1 (zero motion) PASSED")

# --- Test 2: pure yaw, 1 rad/s for 1 second ---
x2 = NominalState(p=np.zeros(3), v=np.zeros(3),
                   q=np.array([1.0, 0.0, 0.0, 0.0]),
                   a_b=np.zeros(3), w_b=np.zeros(3))
w_m2 = np.array([0.0, 0.0, 1.0])
n_steps = 200

for _ in range(n_steps):
    x2 = propagate_nominal(x2, a_m, w_m2, dt, g)

yaw = 2 * np.arctan2(x2.q[3], x2.q[0])   # independent formula, not reusing your own quat code
assert abs(yaw - 1.0) < 1e-4, f"FAIL: yaw={yaw}, expected ~1.0"
assert np.allclose(x2.p, 0, atol=1e-4), f"FAIL: p moved during pure rotation, {x2.p}"
print(f"Test 2 (pure yaw) PASSED — yaw = {yaw:.6f} rad")

### TESTING ERROR STATE F
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def inject(x: NominalState, dx: np.ndarray) -> NominalState:
    """Apply a 15-vector error dx onto nominal state x (the (+) composition, §2)."""
    new_q = quat_normalize(quat_mul(x.q, rotvec_to_quat(dx[IDX['dth']])))
    return NominalState(
        p=x.p + dx[IDX['dp']], v=x.v + dx[IDX['dv']], q=new_q,
        a_b=x.a_b + dx[IDX['dab']], w_b=x.w_b + dx[IDX['dwb']],
    )

def extract_error(x_ref: NominalState, x_pert: NominalState) -> np.ndarray:
    """Recover the 15-vector error of x_pert relative to x_ref (inverse of inject)."""
    dx = np.zeros(15)
    dx[IDX['dp']] = x_pert.p - x_ref.p
    dx[IDX['dv']] = x_pert.v - x_ref.v
    q_diff = quat_mul(quat_conjugate(x_ref.q), x_pert.q)
    dx[IDX['dth']] = 2 * q_diff[1:4]          # small-angle: q{phi} ~ [1, phi/2]
    dx[IDX['dab']] = x_pert.a_b - x_ref.a_b
    dx[IDX['dwb']] = x_pert.w_b - x_ref.w_b
    return dx

# --- a generic, non-degenerate test point (avoid identity/zero everywhere,
#     so every coupling term actually gets exercised) ---
g = np.array([0.0, 0.0, -9.81])
x0 = NominalState(
    p=np.zeros(3), v=np.array([0.5, 0.0, 0.0]),
    q=quat_normalize(rotvec_to_quat(np.array([0.1, 0.05, 0.2]))),
    a_b=np.array([0.01, -0.02, 0.01]), w_b=np.array([0.001, 0.002, -0.001]),
)
a_m = np.array([0.2, 0.1, 9.75])
w_m = np.array([0.05, -0.02, 0.3])
dt = 0.005

F_analytic = error_state_F(x0, a_m, w_m, dt)
x1 = propagate_nominal(x0, a_m, w_m, dt, g)     # unperturbed reference

eps = 1e-6
F_numeric = np.zeros((15, 15))
for i in range(15):
    dx = np.zeros(15)
    dx[i] = eps
    x1_pert = propagate_nominal(inject(x0, dx), a_m, w_m, dt, g)
    F_numeric[:, i] = extract_error(x1, x1_pert) / eps

max_diff = np.max(np.abs(F_numeric - F_analytic))

diff = np.abs(F_numeric - F_analytic)
block_names = ['dp', 'dv', 'dth', 'dab', 'dwb']
for i, ri in enumerate(block_names):
    for j, rj in enumerate(block_names):
        block_max = diff[IDX[ri], IDX[rj]].max()
        if block_max > 1e-5:
            print(f"{ri}-{rj}: max diff = {block_max:.6f}")

print(f"max abs difference: {max_diff}")
assert max_diff < 2e-4, "F doesn't match the finite-difference Jacobian"
print("F checkpoint PASSED")

### TESTING Q

g = np.array([0.0, 0.0, -9.81])
sigma_a, sigma_g = 1.57e-3, 1.40e-4
sigma_ba, sigma_bg = 0.0, 0.0            # isolate pure white-noise growth for this check

x = NominalState(p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]),
                  a_b=np.zeros(3), w_b=np.zeros(3))
P = np.zeros((15, 15))

dt = 0.005
a_m = np.array([0.0, 0.0, 9.81])         # zero-motion, level (from §3.5)
w_m = np.zeros(3)

checkpoints = [25, 50, 75, 100]           # seconds
results = {}
n_steps = int(100 / dt)
next_idx = 0

for step in range(1, n_steps + 1):
    F = error_state_F(x, a_m, w_m, dt)
    Q = build_Q(sigma_a, sigma_g, sigma_ba, sigma_bg, dt)
    P = F @ P @ F.T + Q
    x = propagate_nominal(x, a_m, w_m, dt, g)

    t = step * dt
    if next_idx < len(checkpoints) and abs(t - checkpoints[next_idx]) < dt / 2:
        results[checkpoints[next_idx]] = {
            "dth_var": np.diag(P[IDX['dth'], IDX['dth']]).copy(),
            "dp_var":  np.diag(P[IDX['dp'],  IDX['dp']]).copy(),
        }
        next_idx += 1

for t in checkpoints:
    print(f"t={t:>3}s  dth_var={results[t]['dth_var']}  dp_var={results[t]['dp_var']}")

# Check 1: attitude variance grows linearly, matches sigma_g^2 * t within 1%
expected_dth = sigma_g**2 * 100
rel_err = np.abs(results[100]["dth_var"] - expected_dth) / expected_dth
print(f"attitude variance rel. error at t=100s: {rel_err}")
assert np.all(rel_err < 0.01), "attitude variance doesn't match sigma_g^2 * t"

# Check 2: position variance grows much faster than linear (roughly t^3)
ratio = results[100]["dp_var"] / results[50]["dp_var"]
print(f"position variance ratio (t=100 / t=50): {ratio}  -- expect ~8 if cubic, ~2 if only linear")

### TESTING ENCODER

x0 = NominalState(
    p=np.zeros(3), v=np.array([0.5, 0.1, -0.05]),
    q=quat_normalize(rotvec_to_quat(np.array([0.1, 0.05, 0.2]))),
    a_b=np.zeros(3), w_b=np.zeros(3),
)

h0, H_analytic = encoder_h_and_H(x0)

eps = 1e-6
H_numeric = np.zeros((3, 15))
for i in range(15):
    dx = np.zeros(15)
    dx[i] = eps
    x_pert = inject(x0, dx)
    h_pert, _ = encoder_h_and_H(x_pert)
    H_numeric[:, i] = (h_pert - h0) / eps

max_diff = np.max(np.abs(H_numeric - H_analytic))
print(f"max abs difference: {max_diff}")
assert max_diff < 1e-4, "H doesn't match finite-difference Jacobian"
print("H checkpoint PASSED")