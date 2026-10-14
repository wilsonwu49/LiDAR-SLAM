import numpy as np
from occupancy.occupancy_grid import OccupancyGridMap

S, Z_MAX = 0.05, 12.0
WALL_X, WALL_Y0, WALL_Y1 = 1.5, 0.5, 1.5
POSE = (1.0, 1.0, 0.0)


def fresh():
    return OccupancyGridMap(width_m=2.0, height_m=2.0, resolution=S,
                            origin_x=0.0, origin_y=0.0,
                            l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0,
                            z_max=Z_MAX, r_min=0.2, clearing_dist=8.0)


def synth_wall_scan(n_beams=360):
    """True range from the robot to the wall for each beam, or z_max if it misses."""
    angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    ranges = np.full(n_beams, Z_MAX)
    px, py, _ = POSE
    for i, th in enumerate(angles):
        c = np.cos(th)
        if c <= 1e-9:
            continue
        t = (WALL_X - px) / c
        y = py + t * np.sin(th)
        if WALL_Y0 <= y <= WALL_Y1 and t > 0:
            ranges[i] = t
    return angles, ranges


angles, ranges = synth_wall_scan()
g = fresh()
g.integrate_scan(POSE, angles, ranges)

ok = True
occ_cy, occ_cx = np.where(g.logodds > 0)

if len(occ_cx) == 0:
    print("FAIL no wall detected at all"); ok = False

if not np.all(occ_cx == 30):
    print("FAIL wall not a thin band at cx=30, got columns",
          sorted(set(occ_cx.tolist()))); ok = False

if occ_cy.min() < 9 or occ_cy.max() > 31:
    print("FAIL wall extends outside its true span, cy range",
          occ_cy.min(), occ_cy.max()); ok = False

for cx in range(22, 30):
    if g.logodds[20, cx] >= 0:
        print(f"FAIL cell ({cx},20) between robot and wall not free:",
              g.logodds[20, cx]); ok = False

for cx in range(31, 40):
    if g.logodds[20, cx] != 0.0:
        print(f"FAIL cell ({cx},20) behind the wall was touched:",
              g.logodds[20, cx]); ok = False

for _ in range(20):
    g.integrate_scan(POSE, angles, ranges)
if g.logodds[20, 30] != 5.0:
    print("FAIL wall did not saturate:", g.logodds[20, 30]); ok = False
if g.logodds[20, 25] != -5.0:
    print("FAIL free space did not saturate:", g.logodds[20, 25]); ok = False

cm = g.export_costmap(unknown_as="occupied")
if cm.shape != (40, 40) or cm.dtype != np.int8:
    print("FAIL costmap shape/dtype:", cm.shape, cm.dtype); ok = False
if cm[20, 30] != 1:
    print("FAIL wall not blocked in costmap"); ok = False
if cm[20, 25] != 0:
    print("FAIL free space blocked in costmap"); ok = False
if cm[20, 35] != 1:
    print("FAIL shadow behind wall not conservative under unknown_as='occupied'"); ok = False
if g.export_costmap(unknown_as="free")[20, 35] != 0:
    print("FAIL unknown_as='free' not honoured"); ok = False

print("Section 8.3 checkpoint", "PASSED" if ok else "FAILED")