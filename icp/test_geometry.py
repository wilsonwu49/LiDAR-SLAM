import numpy as np
from icp.geometry import polar_scan_to_xy, transform_points

# --- 1. inverse-transform round-trip ---
rng = np.random.default_rng(0)
P = rng.uniform(-5, 5, size=(20, 2))

phi_star = 0.05
t_star = np.array([0.10, -0.03])

Q = transform_points(P, phi_star, t_star)

R_inv = np.array([[np.cos(-phi_star), -np.sin(-phi_star)],
                   [np.sin(-phi_star),  np.cos(-phi_star)]])
t_inv = -R_inv @ t_star          # NOT just -t_star -- see explanation above

P_recovered = transform_points(Q, -phi_star, t_inv)

max_diff = np.max(np.abs(P_recovered - P))
print(f"max diff after round-trip: {max_diff:.2e}  (target < 1e-9)")
assert max_diff < 1e-9, "rotation/translation convention inconsistent"

# --- 2. invalid-range gating ---
angles = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
ranges = np.array([1.0, np.inf, np.nan, 0.0, 3.0])   # includes inf, nan, AND a 0-range (§1.6 pitfall)
pts = polar_scan_to_xy(angles, ranges, range_min=0.1, range_max=10.0)
assert np.all(np.isfinite(pts)), "inf/NaN survived gating"
print(f"gated points (expect 2 of 5 survive):\n{pts}")

print("Section 1 checkpoint PASSED")