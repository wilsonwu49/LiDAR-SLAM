import numpy as np
from occupancy.occupancy_grid import OccupancyGridMap
from occupancy.bresenham import bresenham_cells
from astar.astar import astar
from astar.costmap import build_costmap
from astar.path import prune_line_of_sight, to_reference_path

RES = 0.05
WALL_X = 5.0
DOOR_LO, DOOR_HI = 4.5, 5.5
Y_MIN, Y_MAX = 0.0, 10.0
Z_MAX = 12.0
R_ROBOT, R_SAFETY = 0.15, 0.05


def true_range(px, py, bearing):
    """Range to the wall at x=5 (which has a doorway), or z_max if the beam misses."""
    c = np.cos(bearing)
    if abs(c) < 1e-9:
        return Z_MAX
    t = (WALL_X - px) / c
    if t <= 0:
        return Z_MAX
    y = py + t * np.sin(bearing)
    if Y_MIN <= y <= Y_MAX and not (DOOR_LO <= y <= DOOR_HI):
        return t
    return Z_MAX


def synth_scan(px, py, n=1080):
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ranges = np.array([true_range(px, py, a) for a in angles])
    return angles, ranges


ok = True

# ---------- Module 3: build the map from synthetic scans ----------
grid = OccupancyGridMap(width_m=10.0, height_m=10.0, resolution=RES,
                        origin_x=0.0, origin_y=0.0,
                        l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0,
                        z_max=Z_MAX, r_min=0.2, clearing_dist=8.0)

for px, py in [(2.0, 5.0), (3.5, 5.0), (5.0, 5.0), (6.5, 5.0), (8.0, 6.5)]:
    angles, ranges = synth_scan(px, py)
    grid.integrate_scan((px, py, 0.0), angles, ranges)

occ = grid.export_costmap(unknown_as="occupied")

if occ.shape != (200, 200):
    print("FAIL costmap shape:", occ.shape); ok = False

wall_col = occ[:, 100]
blocked = int(wall_col[:90].sum() + wall_col[110:].sum())
if blocked < 100:
    print(f"FAIL wall barely detected: {blocked} of 180 cells outside the doorway"); ok = False
if wall_col[95:105].any():
    print("FAIL doorway is blocked in the raw map"); ok = False

# ---------- Module 4: inflate, plan, prune, convert ----------
lethal = build_costmap(occ, RES, R_ROBOT, R_SAFETY)

# world_to_cell returns (cx, cy); astar wants (r, c) = (cy, cx)
scx, scy = grid.world_to_cell(2.0, 5.0)
gcx, gcy = grid.world_to_cell(8.0, 8.0)
start, goal = (scy, scx), (gcy, gcx)

print(f"map: free={int((occ == 0).sum())} blocked={int((occ == 1).sum())}")
print(f"start cell {start} logodds {grid.logodds[start]:+.2f}")
print(f"goal  cell {goal} logodds {grid.logodds[goal]:+.2f}")

if lethal[start]:
    print("FAIL start is inside inflated space"); ok = False
if lethal[goal]:
    print("FAIL goal unobserved or too close to a wall"); ok = False

path = astar(start, goal, lethal, connectivity=8)

if path is None:
    print("FAIL no path found through the doorway"); ok = False
else:
    if path[0] != start or path[-1] != goal:
        print("FAIL path endpoints wrong"); ok = False
    if any(lethal[c] for c in path):
        print("FAIL path enters lethal space"); ok = False

    crossings = [r for (r, c) in path if c == 100]
    if not crossings:
        print("FAIL path never crosses the wall line"); ok = False
    elif not all(90 <= r <= 110 for r in crossings):
        print(f"FAIL crosses the wall outside the doorway: rows {crossings}"); ok = False

    pruned = prune_line_of_sight(path, lethal)
    if pruned[0] != path[0] or pruned[-1] != path[-1]:
        print("FAIL pruning changed the endpoints"); ok = False
    if len(pruned) >= len(path):
        print(f"FAIL no pruning happened: {len(path)} -> {len(pruned)}"); ok = False
    for (r0, c0), (r1, c1) in zip(pruned, pruned[1:]):
        if any(lethal[r, c] for r, c in bresenham_cells(r0, c0, r1, c1)):
            print("FAIL pruned segment crosses lethal space"); ok = False

    # ---------- handoff to Module 5 ----------
    ref = np.asarray(to_reference_path(pruned, origin=(0.0, 0.0), resolution=RES))
    if ref.shape != (len(pruned), 2):
        print("FAIL reference path shape:", ref.shape); ok = False
    if not ((ref >= 0.0).all() and (ref <= 10.0).all()):
        print("FAIL waypoints outside the map extent"); ok = False
    if abs(ref[0, 0] - 2.0) > RES or abs(ref[0, 1] - 5.0) > RES:
        print(f"FAIL first waypoint {ref[0]} should be near (2.0, 5.0)"); ok = False
    if abs(ref[-1, 0] - 8.0) > RES or abs(ref[-1, 1] - 8.0) > RES:
        print(f"FAIL last waypoint {ref[-1]} should be near (8.0, 8.0)"); ok = False

    res = np.asarray(to_reference_path(pruned, origin=(0.0, 0.0),
                                       resolution=RES, resample_ds=0.1))
    gaps = np.linalg.norm(np.diff(res, axis=0), axis=1)
    if gaps[:-1].max() > 0.1 + 1e-6:
        print(f"FAIL resampled spacing too large: {gaps[:-1].max():.4f}"); ok = False
    if len(res) <= len(ref):
        print("FAIL resampling produced no extra points"); ok = False

    print(f"path {len(path)} cells -> pruned {len(pruned)} -> resampled {len(res)}")

print("Modules 3+4 full stack", "PASSED" if ok else "FAILED")
