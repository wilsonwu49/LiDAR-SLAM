import numpy as np
from lidar_nav.mapping.occupancy_grid import OccupancyGridMap
from lidar_nav.mapping.bresenham import bresenham_cells
from lidar_nav.planning.astar import astar
from lidar_nav.planning.costmap import build_costmap
from lidar_nav.planning.path import prune_line_of_sight, to_reference_path
from tests.testkit import Suite

T = Suite("Planning: map a doorway, then plan through it")

RES, Z_MAX = 0.05, 12.0
WALL_X, DOOR_LO, DOOR_HI = 5.0, 4.5, 5.5


def true_range(px, py, bearing):
    c = np.cos(bearing)
    if abs(c) < 1e-9:
        return Z_MAX
    t = (WALL_X - px) / c
    y = py + t * np.sin(bearing)
    return t if t > 0 and 0.0 <= y <= 10.0 and not DOOR_LO <= y <= DOOR_HI else Z_MAX


grid = OccupancyGridMap(width_m=10.0, height_m=10.0, resolution=RES, origin_x=0.0, origin_y=0.0,
                        l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0, z_max=Z_MAX,
                        r_min=0.2, clearing_dist=8.0)
angles = np.linspace(-np.pi, np.pi, 1080, endpoint=False)
for px, py in [(2.0, 5.0), (3.5, 5.0), (5.0, 5.0), (6.5, 5.0), (8.0, 6.5)]:
    grid.integrate_scan((px, py, 0.0), angles, np.array([true_range(px, py, a) for a in angles]))
occ = grid.export_costmap(unknown_as="occupied")

T.section("map from five scans")
col = occ[:, 100]
T.check("wall is detected", col[:90].sum() + col[110:].sum() >= 100,
        f"{int(col[:90].sum() + col[110:].sum())} of 180 wall cells")
T.check("doorway is left open", not col[95:105].any())

lethal = build_costmap(occ, RES, 0.15, 0.05)
scx, scy = grid.world_to_cell(2.0, 5.0)
gcx, gcy = grid.world_to_cell(8.0, 8.0)
start, goal = (scy, scx), (gcy, gcx)

T.section("A* from (2, 5) to (8, 8)")
T.check("start and goal are in free space", not lethal[start] and not lethal[goal])
path = astar(start, goal, lethal, connectivity=8)
T.check("a path is found", path is not None)
if path:
    crossings = [r for r, c in path if c == 100]
    T.check("it goes through the doorway", crossings and all(90 <= r <= 110 for r in crossings),
            f"crosses at rows {min(crossings)}-{max(crossings)}" if crossings else "never crosses")
    T.check("it never enters forbidden space", not any(lethal[c] for c in path))

    pruned = prune_line_of_sight(path, lethal)
    T.section("smoothing")
    T.check("pruning keeps the endpoints and drops corners",
            pruned[0] == path[0] and pruned[-1] == path[-1] and len(pruned) < len(path),
            f"{len(path)} cells -> {len(pruned)} corners")
    T.check("straight-line shortcuts stay clear of walls",
            not any(lethal[r, c] for (r0, c0), (r1, c1) in zip(pruned, pruned[1:])
                    for r, c in bresenham_cells(r0, c0, r1, c1)))

    ref = np.asarray(to_reference_path(pruned, origin=(0.0, 0.0), resolution=RES))
    T.near("first waypoint is the start", ref[0], [2.0, 5.0], RES, " m")
    T.near("last waypoint is the goal", ref[-1], [8.0, 8.0], RES, " m")
    res = np.asarray(to_reference_path(pruned, origin=(0.0, 0.0), resolution=RES, resample_ds=0.1))
    gaps = np.linalg.norm(np.diff(res, axis=0), axis=1)
    T.check("resampled to 10 cm spacing", gaps[:-1].max() <= 0.1 + 1e-6,
            f"{len(res)} waypoints, widest gap {gaps[:-1].max() * 100:.1f} cm")

T.done()
