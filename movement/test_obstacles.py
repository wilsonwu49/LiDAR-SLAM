"""Checks for scan_to_obstacles: the distance gate, wedge domination, frame transform."""

import numpy as np
from movement.obstacles import scan_to_obstacles

D_GATE = 0.75      # R + v_max/gamma = 0.25 + 0.5/1.0
N_SECTORS = 36


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def ray_cast_wall(bearings, wall_x, r_max=12.0):
    """Range to the vertical wall x = wall_x from the origin facing +x."""
    cosb = np.cos(bearings)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(cosb > 1e-9, wall_x / cosb, np.inf)
    return np.where(r > r_max, np.inf, r)


ok = True
bearings = np.linspace(-np.pi, np.pi, 1080, endpoint=False)

print()
print("gate: a wall beyond d_gate must produce nothing")
obs = scan_to_obstacles(ray_cast_wall(bearings, 1.0), bearings,
                        np.array([0.0, 0.0, 0.0]), D_GATE, N_SECTORS)
ok &= check(f"wall at 1.0 m with gate 0.75 -> {len(obs)} points", len(obs) == 0)

print()
print("reduction: a wall at 0.5 m")
obs = scan_to_obstacles(ray_cast_wall(bearings, 0.5), bearings,
                        np.array([0.0, 0.0, 0.0]), D_GATE, N_SECTORS)
print(f"        1080 beams -> {len(obs)} obstacle points")
ok &= check("at most one point per wedge", len(obs) <= N_SECTORS)
ok &= check("nonempty", len(obs) > 0)
ok &= check("every survivor lies on the wall (x = 0.5)",
            np.allclose(obs[:, 0], 0.5, atol=1e-9))
ok &= check("every survivor is inside the gate",
            np.all(np.linalg.norm(obs, axis=1) < D_GATE + 1e-9))

print()
print("domination: two returns in one wedge, only the nearer survives")
b2 = np.array([0.00, 0.02])          # both well inside one 10 deg wedge
r2 = np.array([0.60, 0.40])
obs = scan_to_obstacles(r2, b2, np.array([0.0, 0.0, 0.0]), D_GATE, N_SECTORS)
ok &= check("one point kept", len(obs) == 1)
ok &= check("it is the nearer one (r = 0.40)",
            np.isclose(np.linalg.norm(obs[0]), 0.40, atol=1e-9))

print()
print("frame transform")
obs = scan_to_obstacles(np.array([0.4]), np.array([0.0]),
                        np.array([0.0, 0.0, 0.0]), D_GATE, N_SECTORS)
ok &= check("beam ahead, pose at origin -> (0.4, 0)", np.allclose(obs[0], [0.4, 0.0]))

obs = scan_to_obstacles(np.array([0.4]), np.array([0.0]),
                        np.array([1.0, 2.0, np.pi / 2]), D_GATE, N_SECTORS)
ok &= check("same beam, pose [1, 2, 90 deg] -> (1.0, 2.4)",
            np.allclose(obs[0], [1.0, 2.4], atol=1e-9))

obs = scan_to_obstacles(np.array([0.4]), np.array([np.pi / 2]),
                        np.array([0.0, 0.0, 0.0]), D_GATE, N_SECTORS)
ok &= check("beam to the left -> (0, 0.4)", np.allclose(obs[0], [0.0, 0.4], atol=1e-9))

print()
print("degenerate inputs")
ok &= check("all inf -> empty",
            len(scan_to_obstacles(np.full(100, np.inf), bearings[:100],
                                  np.zeros(3), D_GATE, N_SECTORS)) == 0)
ok &= check("nan is dropped",
            len(scan_to_obstacles(np.array([np.nan, 0.4]), np.array([0.0, 0.1]),
                                  np.zeros(3), D_GATE, N_SECTORS)) == 1)
ok &= check("sub-r_min return is dropped",
            len(scan_to_obstacles(np.array([0.01]), np.array([0.0]),
                                  np.zeros(3), D_GATE, N_SECTORS)) == 0)
ok &= check("bearing exactly +pi does not IndexError",
            len(scan_to_obstacles(np.array([0.4]), np.array([np.pi]),
                                  np.zeros(3), D_GATE, N_SECTORS)) == 1)

print()
print("corridor: 1 m wide, robot in the middle")
sinb = np.abs(np.sin(bearings))
with np.errstate(divide="ignore", invalid="ignore"):
    walls = np.where(sinb > 1e-9, 0.5 / sinb, np.inf)
obs = scan_to_obstacles(walls, bearings, np.zeros(3), D_GATE, N_SECTORS)
print(f"        1080 beams -> {len(obs)} obstacle points  "
      f"(both walls 0.5 m away, gate {D_GATE})")
ok &= check("bounded by wedge count", len(obs) <= N_SECTORS)

print()
print("ALL CHECKS", "PASSED" if ok else "FAILED")
print()
