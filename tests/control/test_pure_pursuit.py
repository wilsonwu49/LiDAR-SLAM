import numpy as np
from lidar_nav.control.pure_pursuit import pure_pursuit_command
from tests.testkit import Suite

T = Suite("Control: pure pursuit")
ORIGIN = np.array([0.0, 0.0, 0.0])
straight = np.column_stack([np.arange(0.0, 4.0, 0.05), np.zeros(80)])


def cmd(pose, path, L=0.5):
    return pure_pursuit_command(np.asarray(pose, float), path, 0.5, 1.0, L, L, 0.2)


T.section("worked example from the guide (hand-computed)")
path = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
u, carrot = pure_pursuit_command(ORIGIN, path, v_desired=0.5, lookahead_gain=5.0,
                                 L_min=0.3, L_max=2.5, wheelbase=0.2)
T.near("carrot lands where the circle crosses the path", carrot, [2.0, 1.446], 1e-3, " m")
T.near("turn rate", u[1], 0.237, 1e-3, " rad/s")

T.section("steering direction")
T.near("path dead ahead: no turning", cmd(ORIGIN, straight)[0][1], 0.0, 1e-9)
up = np.column_stack([np.arange(0.0, 2.0, 0.05), np.arange(0.0, 2.0, 0.05)])
w_up, w_dn = cmd(ORIGIN, up)[0][1], cmd(ORIGIN, up * [1.0, -1.0])[0][1]
T.check("path curving left turns left", w_up > 0, f"{w_up:+.3f} rad/s")
T.near("mirror-image path turns exactly the other way", w_up + w_dn, 0.0, 1e-9)
T.check("robot pointing off the path steers back", cmd([0.0, 0.0, 0.3], straight)[0][1] < 0)

T.section("arrival")
T.near("sitting on the last waypoint commands a full stop", cmd([3.95, 0.0, 0.0], straight)[0], [0.0, 0.0], 0)


def simulate(path, pose, L, max_steps=2000, dt=0.02):
    pose = np.array(pose, dtype=float)
    trail, omegas = [pose[:2].copy()], []
    for n in range(max_steps):
        u, _ = cmd(pose, path, L)
        if u[0] == 0.0 and u[1] == 0.0:
            return np.array(trail), np.array(omegas), True
        omegas.append(u[1])
        pose += np.array([u[0] * np.cos(pose[2]), u[0] * np.sin(pose[2]), u[1]]) * dt
        trail.append(pose[:2].copy())
    return np.array(trail), np.array(omegas), False


def cross_track(trail, path):
    return np.linalg.norm(trail[:, None] - path[None], axis=2).min(axis=1).max()


T.section("closed loop: a right-angle turn at 0.5 m/s")
corner = np.vstack([np.column_stack([np.arange(0, 3, 0.05), np.zeros(60)]),
                    np.column_stack([np.full(60, 3.0), np.arange(0, 3, 0.05)])])
runs = {L: simulate(corner, ORIGIN, L) for L in (0.2, 0.5, 1.2)}
T.check("arrives with every lookahead tried", all(r[2] for r in runs.values()))
xt = {L: cross_track(r[0], corner) * 1000 for L, r in runs.items()}
T.check("longer lookahead cuts the corner more", xt[0.2] < xt[0.5] < xt[1.2],
        f"{xt[0.2]:.0f} / {xt[0.5]:.0f} / {xt[1.2]:.0f} mm off the path")
wmax = {L: np.abs(r[1]).max() for L, r in runs.items()}
T.check("shorter lookahead turns harder", wmax[0.2] > wmax[0.5] > wmax[1.2],
        f"peak {wmax[0.2]:.2f} / {wmax[0.5]:.2f} / {wmax[1.2]:.2f} rad/s")
T.check("default 0.5 m stays inside the 0.33 m planner margin", xt[0.5] < 330, f"{xt[0.5]:.0f} mm")

T.section("closed loop: starting 0.5 m off a straight path")
line = np.column_stack([np.arange(0.0, 6.0, 0.05), np.zeros(120)])
trail, _, _ = simulate(line, [0.0, 0.5, 0.0], 0.5)
settled = np.abs(trail[len(trail) // 2:, 1]).max() * 1000
T.check("pulls back onto the path", settled < 10, f"{settled:.1f} mm off by the second half")

T.done()
