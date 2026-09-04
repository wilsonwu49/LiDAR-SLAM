import numpy as np
from bresenham import bresenham_cells
from inv_sensor_model import classify_beam

# Convention, fixed for this whole file:
#   logodds has shape (H, W), indexed logodds[cy, cx]
#   cx indexes columns (world x), cy indexes rows (world y)
#   cell (0,0) covers world [origin_x, origin_x+s) x [origin_y, origin_y+s)
#   positive log-odds = occupied


class OccupancyGridMap:

    def __init__(self, width_m, height_m, resolution, origin_x, origin_y,
                 l_occ, l_free, l_min, l_max, z_max, r_min=0.2, clearing_dist=8.0):
        """Allocate the log-odds array and store every parameter the other methods need.
        width_m, height_m   world extent covered by the map [m]
        resolution          cell size [m/cell]
        origin_x, origin_y  world position of the lower-left corner of cell (0,0) [m]
        l_occ, l_free       log-odds added for a hit and for a pass-through
        l_min, l_max        clamp bounds on any cell's accumulated log-odds
        z_max               sensor maximum range [m]
        r_min               sensor minimum reliable range [m]
        clearing_dist       how far to carve free space on a no-return beam [m]"""

        self.s = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.W = int(np.ceil(width_m / resolution))
        self.H = int(np.ceil(height_m / resolution))
        self.logodds = np.zeros((self.H, self.W), dtype=float)

        self.l_occ = l_occ
        self.l_free = l_free
        self.l_min = l_min
        self.l_max = l_max

        self.z_max = z_max
        self.r_min = r_min
        self.clearing_dist = clearing_dist

    def world_to_cell(self, wx, wy):
        """Convert a world position in meters to integer cell indices (cx, cy)."""
        cx = int(np.floor((wx - self.origin_x) / self.s))
        cy = int(np.floor((wy - self.origin_y) / self.s))
        return cx, cy

    def in_bounds(self, cx, cy):
        """True if (cx, cy) indexes a real cell in the array."""
        return 0 <= cx < self.W and 0 <= cy < self.H

    def update_cell(self, cx, cy, delta):
        """Add delta to a cell's log-odds and clamp; silently ignore out-of-bounds cells."""
        if not self.in_bounds(cx,cy):
            return
        
        self.logodds[cy,cx] = self.logodds[cy,cx] + delta
        if self.logodds[cy,cx] >= self.l_max:
            self.logodds[cy,cx] = self.l_max
        elif self.logodds[cy,cx] <= self.l_min:
            self.logodds[cy,cx] = self.l_min

    def logodds_to_prob(self, cx, cy):
        """Return a cell's occupancy probability, sigmoid of its stored log-odds."""
        log = self.logodds[cy, cx]
        return 1 / (1 + np.exp(-log))

    def integrate_scan(self, pose_xytheta, angles, ranges):
        """
        Fold one full scan into the map using the ICP-corrected pose.

        pose_xytheta   (px, py, psi) robot pose in the map frame [m, m, rad]
        angles         per-beam bearings in the sensor frame [rad]
        ranges         per-beam measured ranges [m], same length as angles
        """
        px, py, psi = pose_xytheta

        # the pose is fixed for the whole scan, so this is computed once
        rx, ry = self.world_to_cell(px, py)

        for theta, z in zip(angles, ranges):
            result = classify_beam(z, self.z_max, self.r_min, self.clearing_dist)
            if result is None:
                continue
            verdict, ray_range = result

            # world endpoint: ray_range metres out along bearing (psi + theta)
            bearing = psi + theta
            endx_world = px + ray_range * np.cos(bearing)
            endy_world = py + ray_range * np.sin(bearing)
            endx, endy   = self.world_to_cell(endx_world, endy_world)
            cells = bresenham_cells(rx, ry, endx, endy)

            # every cell before the last one is free
            for x,y in cells[:-1]:
                self.update_cell(x, y, self.l_free)

            # last cell: occupied on a valid hit, free on a no-return
            if verdict == "valid_hit":
                self.update_cell(endx, endy, self.l_occ)
            else:
                self.update_cell(endx, endy, self.l_free)


    def export_costmap(self, unknown_as="occupied", t=0.85):
        """Threshold the log-odds into free / occupied / unknown for A* (Module 4)."""
        occupied = self.logodds > t
        free = self.logodds < -t
        unknown = ~(occupied | free)

        costmap = np.zeros((self.H, self.W), dtype=np.int8)
        costmap[occupied] = 1
        if unknown_as == "occupied":
            costmap[unknown] = 1
        return costmap