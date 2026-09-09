import numpy as np
from occupancy.bresenham import bresenham_cells


def prune_line_of_sight(path_cells, lethal_mask):
    """
    Drop intermediate waypoints whose skip-segment is collision-free.

    path_cells   list of (r, c) from astar
    lethal_mask  bool array (H, W), True = impassable
    returns      shorter list of (r, c), first and last preserved
    """
    corner_cells = [path_cells[0]]
    anchor = 0
    while corner_cells[-1] != path_cells[-1]:
        best = anchor + 1
        for i in range(len(path_cells) - 1, anchor, -1):
            subpath_cells = bresenham_cells(corner_cells[-1][0],corner_cells[-1][1],path_cells[i][0],path_cells[i][1])
            if not any(lethal_mask[x, y] for x, y in subpath_cells):
                best = i
                break
        corner_cells.append(path_cells[best])
        anchor = best
    return corner_cells


def to_reference_path(path_cells, origin, resolution, resample_ds=None):
    """
    Convert grid cells to the world-frame waypoint array Module 5 tracks.

    path_cells    list of (r, c)
    origin        (origin_x, origin_y) world position of cell (0,0) [m]
    resolution    cell size [m/cell]
    resample_ds   target spacing between output waypoints [m]
    returns       (N, 2) float array of world (x, y) waypoints
    """

    cells = np.asarray(path_cells, dtype=float)[:, ::-1]     # (r, c) -> (c, r)
    world_points = (cells + 0.5) * resolution + origin
    if resample_ds == None or len(world_points) < 2:
        return world_points

    # Summing up distances
    seg = np.linalg.norm(np.diff(world_points, axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])

    if d[-1] < 1e-9:
        return world_points

    # Query based on target distances knowing total distances
    s = np.arange(0.0, d[-1], resample_ds) 
    x = np.interp(s, d, world_points[:, 0])
    y = np.interp(s, d, world_points[:, 1])

    return np.vstack([np.column_stack([x, y]), world_points[-1]])