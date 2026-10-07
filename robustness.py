"""Robustness sweeps across the modules. Not pass/fail -- these measure where things break."""

import numpy as np

from icp.geometry import transform_points
from icp.icp import icp_scan_match
from astar.astar import astar
from astar.costmap import build_costmap
from movement.pure_pursuit import pure_pursuit_command


# ---------------------------------------------------------------- scan geometry

def corner_scan(n=240, span=2.0):
    """L-shaped corner: one wall along +x, one along +y."""
    half = n // 2
    a = np.column_stack([np.linspace(0.0, span, half), np.zeros(half)])
    b = np.column_stack([np.zeros(half), np.linspace(0.0, span, half)])
    return np.vstack([a, b])


def corridor_scan(n=240, span=4.0, width=1.5):
    """Two parallel walls -- motion along the corridor is unobservable."""
    half = n // 2
    a = np.column_stack([np.linspace(0.0, span, half), np.zeros(half)])
    b = np.column_stack([np.linspace(0.0, span, half), np.full(half, width)])
    return np.vstack([a, b])


def make_pair(scan, phi, t, noise=0.0, outlier_frac=0.0, rng=None):
    """source -> reference by (phi, t), with noise and outliers on the reference."""
    rng = rng or np.random.default_rng(0)
    ref = transform_points(scan, phi, np.array(t, dtype=float))
    if noise > 0:
        ref = ref + rng.normal(0.0, noise, ref.shape)
    if outlier_frac > 0:
        k = int(outlier_frac * len(ref))
        idx = rng.choice(len(ref), k, replace=False)
        ref[idx] += rng.uniform(-0.5, 0.5, (k, 2))
    return scan, ref


# ---------------------------------------------------------------- ICP sweeps

PHI_T, T_T = 0.05, (0.08, 0.03)


def icp_error(ref_scan, noise=0.0, outlier_frac=0.0, init_err=0.0, seed=0):
    rng = np.random.default_rng(seed)
    src, ref = make_pair(ref_scan, PHI_T, T_T, noise, outlier_frac, rng)
    init_phi = PHI_T + init_err
    init_t = (T_T[0] + init_err, T_T[1] + init_err)
    phi, t, cov, info = icp_scan_match(src, ref, init_phi=init_phi, init_t=init_t)
    pos_err = np.linalg.norm(np.array(t) - np.array(T_T))
    return pos_err, abs(phi - PHI_T), info


print("=" * 66)
print("ICP vs range noise   (corner, perfect initial guess)")
print(f"{'noise sigma':>12} {'pos err mm':>12} {'yaw err mrad':>13} {'iters':>6} {'cond':>11}")
scan = corner_scan()
for s in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
    p, y, info = icp_error(scan, noise=s)
    print(f"{s:12.3f} {p*1000:12.2f} {y*1000:13.2f} {info['iterations']:6d} {info['condition_number']:11.2e}")

print()
print("=" * 66)
print("ICP vs initial-guess error   (corner, 5 mm noise) -- basin of convergence")
print(f"{'init err':>12} {'pos err mm':>12} {'yaw err mrad':>13} {'converged':>10}")
for e in [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]:
    p, y, info = icp_error(scan, noise=0.005, init_err=e)
    print(f"{e:12.2f} {p*1000:12.2f} {y*1000:13.2f} {str(info['converged']):>10}")

print()
print("=" * 66)
print("ICP vs outlier fraction   (corner, 5 mm noise, trim keeps 80%)")
print(f"{'outliers':>12} {'pos err mm':>12} {'yaw err mrad':>13} {'inliers':>8}")
for f in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
    p, y, info = icp_error(scan, noise=0.005, outlier_frac=f)
    print(f"{f:12.2f} {p*1000:12.2f} {y*1000:13.2f} {info['num_inliers']:8d}")

print()
print("=" * 66)
print("ICP geometry: corner vs corridor   (same noise, same transform)")
for name, sc in [("corner", corner_scan()), ("corridor", corridor_scan())]:
    p, y, info = icp_error(sc, noise=0.005)
    print(f"  {name:9s} pos err {p*1000:8.2f} mm   cond {info['condition_number']:.2e}   "
          f"inliers {info['num_inliers']}")


# ---------------------------------------------------------------- pure pursuit

def simulate(path, pose, L_fixed, steps=900, dt=0.02, v=0.5):
    pose = np.array(pose, dtype=float)
    trail = [pose[:2].copy()]
    omegas = []
    for _ in range(steps):
        u, _ = pure_pursuit_command(pose, path, v, 1.0, L_fixed, L_fixed, 0.2)
        omegas.append(u[1])
        pose[0] += u[0] * np.cos(pose[2]) * dt
        pose[1] += u[0] * np.sin(pose[2]) * dt
        pose[2] += u[1] * dt
        trail.append(pose[:2].copy())
    return np.array(trail), np.array(omegas)


def cross_track(trail, path):
    """Max distance from any trail point to the nearest path point."""
    d = np.linalg.norm(trail[:, None, :] - path[None, :, :], axis=2)
    return d.min(axis=1).max()


def sign_flips(omegas, tol=1e-3):
    s = np.sign(omegas[np.abs(omegas) > tol])
    return int(np.sum(s[1:] != s[:-1]))


straight = np.column_stack([np.arange(0, 6.0, 0.05), np.zeros(120)])
right_angle = np.vstack([
    np.column_stack([np.arange(0, 3.0, 0.05), np.zeros(60)]),
    np.column_stack([np.full(60, 3.0), np.arange(0, 3.0, 0.05)]),
])

print()
print("=" * 66)
print("Pure pursuit vs lookahead distance   (right-angle path, v = 0.5)")
print(f"{'L_d (m)':>9} {'max cross-track mm':>20} {'omega sign flips':>18} {'max |omega|':>12}")
for L in [0.10, 0.20, 0.35, 0.50, 0.80, 1.20, 2.00]:
    trail, om = simulate(right_angle, [0.0, 0.0, 0.0], L)
    print(f"{L:9.2f} {cross_track(trail, right_angle)*1000:20.1f} "
          f"{sign_flips(om):18d} {np.abs(om).max():12.3f}")

print()
print("Pure pursuit started 0.5 m off the path   (straight path)")
print(f"{'L_d (m)':>9} {'max cross-track mm':>20} {'final cross-track mm':>22}")
for L in [0.20, 0.50, 1.00]:
    trail, om = simulate(straight, [0.0, 0.5, 0.0], L)
    d = np.linalg.norm(trail[-1] - straight, axis=1).min()
    print(f"{L:9.2f} {cross_track(trail, straight)*1000:20.1f} {d*1000:22.1f}")


# ---------------------------------------------------------------- A* clutter

print()
print("=" * 66)
print("A* vs obstacle density   (200x200 grid, 5 cm cells, 40 trials each)")
print(f"{'density':>9} {'solved':>8} {'capped':>8} {'no path':>9} {'mean exp':>10} {'max exp':>9}")
rng = np.random.default_rng(1)
for dens in [0.0, 0.05, 0.10, 0.20, 0.30]:
    solved = capped = nopath = 0
    exps = []
    for _ in range(40):
        occ = (rng.random((200, 200)) < dens).astype(np.int8)
        occ[:6, :6] = 0
        occ[-6:, -6:] = 0
        lethal = build_costmap(occ, 0.05, 0.15, 0.05)
        if lethal[0, 0] or lethal[199, 199]:
            continue
        p = astar((0, 0), (199, 199), lethal, connectivity=8, max_expansions=20000)
        if p is None:
            nopath += 1
        else:
            solved += 1
            exps.append(len(p))
    print(f"{dens:9.2f} {solved:8d} {capped:8d} {nopath:9d} "
          f"{(np.mean(exps) if exps else 0):10.0f} {(max(exps) if exps else 0):9d}")

print()
print("=" * 66)
print("done")
