import numpy as np
from movement import pure_pursuit as pp
from movement.pure_pursuit import pure_pursuit_command


# ---------------------------------------------------------------- unit checks

ok = True

# A.7: carrot handed in directly via a two-point path straight ahead of the robot
path = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
u, carrot = pure_pursuit_command(np.array([0.0, 0.0, 0.0]), path,
                                 v_desired=0.5, lookahead_gain=5.0,
                                 L_min=0.3, L_max=2.5, wheelbase=0.2)
if abs(carrot[0] - 2.0) > 1e-9 or abs(carrot[1] - 1.446) > 1e-3:
    print("FAIL worked-example carrot:", carrot); ok = False
if abs(u[1] - 0.237) > 1e-3:
    print("FAIL worked-example omega:", u[1]); ok = False

# carrot dead ahead -> no turn;  mirrored carrot -> mirrored turn
straight = np.column_stack([np.arange(0.0, 4.0, 0.05), np.zeros(80)])
u_s, _ = pure_pursuit_command(np.array([0.0, 0.0, 0.0]), straight, 0.5, 1.0, 0.5, 0.5, 0.2)
if abs(u_s[1]) > 1e-9:
    print("FAIL straight path should give omega 0:", u_s[1]); ok = False

mirrored = straight * np.array([1.0, -1.0])
up = np.column_stack([np.arange(0.0, 2.0, 0.05), np.arange(0.0, 2.0, 0.05)])
dn = up * np.array([1.0, -1.0])
u_up, _ = pure_pursuit_command(np.array([0.0, 0.0, 0.0]), up, 0.5, 1.0, 0.5, 0.5, 0.2)
u_dn, _ = pure_pursuit_command(np.array([0.0, 0.0, 0.0]), dn, 0.5, 1.0, 0.5, 0.5, 0.2)
if u_up[1] <= 0 or abs(u_up[1] + u_dn[1]) > 1e-9:
    print("FAIL sign symmetry:", u_up[1], u_dn[1]); ok = False

# heading offset: path straight ahead but robot yawed left -> should steer right
u_y, _ = pure_pursuit_command(np.array([0.0, 0.0, 0.3]), straight, 0.5, 1.0, 0.5, 0.5, 0.2)
if u_y[1] >= 0:
    print("FAIL yawed robot should steer back toward the path:", u_y[1]); ok = False

# arrival: sitting on the last waypoint must command a full stop
u_a, _ = pure_pursuit_command(np.array([3.95, 0.0, 0.0]), straight, 0.5, 1.0, 0.5, 0.5, 0.2)
if not (u_a[0] == 0.0 and u_a[1] == 0.0):
    print("FAIL should have stopped at the goal:", u_a); ok = False

print("unit checks", "PASSED" if ok else "FAILED")


# ---------------------------------------------------------------- closed loop

def simulate(path, pose, L_fixed, max_steps=2000, dt=0.02, v=0.5):
    pose = np.array(pose, dtype=float)
    trail, omegas = [pose[:2].copy()], []
    for n in range(max_steps):
        u, _ = pure_pursuit_command(pose, path, v, 1.0, L_fixed, L_fixed, 0.2)
        if u[0] == 0.0 and u[1] == 0.0:
            return np.array(trail), np.array(omegas), n, True
        omegas.append(u[1])
        pose[0] += u[0] * np.cos(pose[2]) * dt
        pose[1] += u[0] * np.sin(pose[2]) * dt
        pose[2] += u[1] * dt
        trail.append(pose[:2].copy())
    return np.array(trail), np.array(omegas), max_steps, False


def cross_track(trail, path):
    d = np.linalg.norm(trail[:, None, :] - path[None, :, :], axis=2)
    return d.min(axis=1).max()


def sign_flips(om, tol=1e-3):
    s = np.sign(om[np.abs(om) > tol])
    return int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0


right_angle = np.vstack([
    np.column_stack([np.arange(0.0, 3.0, 0.05), np.zeros(60)]),
    np.column_stack([np.full(60, 3.0), np.arange(0.0, 3.0, 0.05)]),
])

print()
print("lookahead sweep, right-angle path, v = 0.5 m/s")
print(f"{'L_d (m)':>8} {'arrived':>8} {'steps':>7} {'max x-track mm':>16} "
      f"{'flips':>6} {'max |w|':>9} {'w at arrival':>13}")
for L in [0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00]:
    trail, om, n, arrived = simulate(right_angle, [0.0, 0.0, 0.0], L)
    print(f"{L:8.2f} {str(arrived):>8} {n:7d} {cross_track(trail, right_angle)*1000:16.1f} "
          f"{sign_flips(om):6d} {np.abs(om).max():9.3f} {abs(om[-1]):13.3f}")

print()
print("recovery from 0.5 m off a straight path")
print(f"{'L_d (m)':>8} {'max x-track mm':>16} {'settled x-track mm':>20} {'flips':>6}")
straight6 = np.column_stack([np.arange(0.0, 6.0, 0.05), np.zeros(120)])
for L in [0.20, 0.50, 1.00, 1.50]:
    trail, om, n, arrived = simulate(straight6, [0.0, 0.5, 0.0], L)
    half = len(trail) // 2
    settled = np.abs(trail[half:, 1]).max()
    print(f"{L:8.2f} {cross_track(trail, straight6)*1000:16.1f} {settled*1000:20.1f} "
          f"{sign_flips(om):6d}")

print()
print("terminal omega spike vs MIN_GOAL_DISTANCE   (straight path, L_d = 0.5)")
print(f"{'MIN_GOAL':>10} {'max |w| last 20 steps':>24}")
original = pp.MIN_GOAL_DISTANCE
for g in [0.05, 0.10, 0.15, 0.20, 0.30]:
    pp.MIN_GOAL_DISTANCE = g
    _, om, _, _ = simulate(straight6, [0.0, 0.0, 0.0], 0.5)
    print(f"{g:10.2f} {np.abs(om[-20:]).max():24.3f}")
pp.MIN_GOAL_DISTANCE = original
