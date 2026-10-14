import numpy as np
from icp.geometry import transform_points
from icp.costs import point_to_line_cost
from icp.ptl_solve import solve_point_to_line_step

# --- build an L-shaped corner reference: two perpendicular walls ---
wall1 = np.column_stack((np.linspace(-1, 1, 50), np.zeros(50)))          # along y=0, normal [0,1]
wall2 = np.column_stack((np.zeros(50), np.linspace(0.05, 2, 50)))        # along x=0, normal [1,0]
dst = np.vstack([wall1, wall2])
normals = np.vstack([
    np.tile([0.0, 1.0], (50, 1)),
    np.tile([1.0, 0.0], (50, 1)),
])

# --- 1. single-step recovery on a small known misalignment ---
phi_err, t_err = 0.03, np.array([0.05, 0.02])
src = transform_points(dst, phi_err, t_err)   # src = dst, displaced by a small known error

cost_before = point_to_line_cost(src, dst, normals, 0.0, np.zeros(2))
d_phi, d_t = solve_point_to_line_step(src, dst, normals)
corrected = transform_points(src, d_phi, d_t)
cost_after = point_to_line_cost(corrected, dst, normals, 0.0, np.zeros(2))

print(f"cost before correction: {cost_before:.6f}")
print(f"cost after correction:  {cost_after:.6f}  (expect near 0, noise-free corner)")
assert cost_after < cost_before * 1e-3, "one step should nearly fully solve a small, noise-free, well-constrained misalignment"

# --- 2. monotonic cost over a few manual iterations (previews §8) ---
src2 = transform_points(dst, 0.15, np.array([0.2, -0.1]))   # a slightly bigger misalignment
costs = [point_to_line_cost(src2, dst, normals, 0.0, np.zeros(2))]
for _ in range(5):
    d_phi, d_t = solve_point_to_line_step(src2, dst, normals)
    src2 = transform_points(src2, d_phi, d_t)
    costs.append(point_to_line_cost(src2, dst, normals, 0.0, np.zeros(2)))

print(f"cost trace: {[f'{c:.5f}' for c in costs]}")
assert all(costs[i+1] <= costs[i] + 1e-9 for i in range(len(costs)-1)), "cost increased somewhere -- sign bug"

# --- 3. degenerate geometry: single wall, no corner ---
wall_only = wall1.copy()
normals_only = np.tile([0.0, 1.0], (50, 1))
src3 = transform_points(wall_only, 0.0, np.array([0.05, 0.0]))   # pure along-wall slide

A = np.array([normals_only[:,0], normals_only[:,1],
              -normals_only[:,0]*src3[:,1] + normals_only[:,1]*src3[:,0]]).T
eigvals = np.linalg.eigvalsh(A.T @ A)
condition_number = eigvals[-1] / eigvals[0]
print(f"single-wall A^T A eigenvalues: {eigvals}, condition number: {condition_number:.2e}")
assert condition_number > 1e6, "expected near-singular A^T A along the wall direction"

print("Section 5 checkpoint PASSED")