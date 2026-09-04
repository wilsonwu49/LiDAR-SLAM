import numpy as np
from occupancy.bresenham import bresenham_cells


def prune_line_of_sight(path_cells, lethal_mask):
    """
    Drop intermediate waypoints whose skip-segment is collision-free.

    path_cells   list of (r, c) from astar
    lethal_mask  bool array (H, W), True = impassable
    returns      shorter list of (r, c), first and last preserved
    """
    # walk forward from the current kept waypoint, find the furthest later
    # waypoint reachable by a straight line that crosses no lethal cell
    # (use bresenham_cells to enumerate that line), keep it, repeat
    ...


def to_reference_path(path_cells, origin, resolution, resample_ds):
    """
    Convert grid cells to the world-frame waypoint array Module 5 tracks.

    path_cells    list of (r, c)
    origin        (origin_x, origin_y) world position of cell (0,0) [m]
    resolution    cell size [m/cell]
    resample_ds   target spacing between output waypoints [m]
    returns       (N, 2) float array of world (x, y) waypoints
    """
    # 1. cell -> world CENTRE:  origin + (index + 0.5) * resolution
    #    note (r, c) is (row, col) = (cy, cx), so x comes from c and y from r
    # 2. resample along the polyline to roughly uniform spacing resample_ds
    ...