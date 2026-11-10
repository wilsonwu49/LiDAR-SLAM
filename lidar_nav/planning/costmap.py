import numpy as np
from scipy import ndimage


def build_costmap(occupancy, resolution, r_robot, r_safety):
    """
    Inflate an occupancy map so A* can treat the robot as a single point.

    occupancy    int8 array (H, W) from OccupancyGridMap.export_costmap, 1 = blocked
    resolution   cell size [m/cell]
    r_robot      robot footprint radius [m]
    r_safety     extra clearance margin [m]
    returns      lethal_mask bool array (H, W), True = cannot plan through
    """
    if not np.any(occupancy):
        return np.zeros(occupancy.shape, dtype=bool)
    
    dist = ndimage.distance_transform_edt(occupancy == 0)
    r_total = r_robot + r_safety
    r_total_cells = r_total / resolution
    return dist < r_total_cells