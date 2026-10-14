"""Top-level navigation control step: scan -> obstacles -> pure pursuit -> safety filter."""

from dataclasses import dataclass

import numpy as np

from movement.obstacles import scan_to_obstacles, grid_to_obstacles
from movement.pure_pursuit import pure_pursuit_command
from movement.safety_filter_qp import safety_filter


@dataclass
class ControlParams:
    """
    Every tuned constant the control stack needs, in one place.

    r_robot         physical half-width of the robot [m]
    r_safety        planner inflation on top of r_robot [m]
    R               CBF safety radius [m]
    ell             feeler offset ahead of the robot centre [m]
    alpha_gain      gamma in alpha(h) = gamma * h [1/s]
    k               standard deviations of pose error to cover
    v_max           max forward speed [m/s]
    omega_max       max turn rate [rad/s]
    v_desired       cruise speed handed to pure pursuit [m/s]
    lookahead_gain  L_d = lookahead_gain * v_desired, clipped to [L_min, L_max]
    L_min           minimum lookahead distance [m]
    L_max           maximum lookahead distance [m]
    wheelbase       track width, for the (v, omega) -> wheel speed conversion [m]
    n_sectors       angular wedges in the scan reduction
    """
    r_robot: float = 0.15
    r_safety: float = 0.18
    R: float = 0.25
    ell: float = 0.05
    alpha_gain: float = 1.0
    k: float = 2.0
    v_max: float = 0.5
    omega_max: float = 2.0
    v_desired: float = 0.4
    lookahead_gain: float = 1.0
    L_min: float = 0.3
    L_max: float = 0.5
    wheelbase: float = 0.2
    n_sectors: int = 36

    # A* measures clearance cell-centre to cell-centre, the CBF measures it in
    # metres. An occupied cell is a square, so its surface can be half a cell
    # nearer than its centre -- the planner must beat the CBF by at least that
    # much or it will hand over paths the filter refuses to drive.
    grid_resolution: float = 0.05

    def __post_init__(self):
        inflation = self.r_robot + self.r_safety
        reach = self.R + self.ell
        required = reach + 0.5 * self.grid_resolution
        if inflation < required:
            raise ValueError(
                f"planner inflation {inflation:.3f} m must be at least "
                f"{required:.3f} m = CBF reach {reach:.3f} m + half a cell "
                f"({0.5 * self.grid_resolution:.3f} m of discretization slack); "
                f"otherwise A* hands over paths the filter brakes on")

    @property
    def planner_slack(self):
        """Metres by which the planner out-reserves the CBF."""
        return (self.r_robot + self.r_safety) - (self.R + self.ell)

    @property
    def d_gate(self):
        """Range beyond which a constraint provably cannot bind."""
        return self.R + self.v_max / self.alpha_gain


def control_step(pose, P, path, grid, params):
    """
    One control cycle: nominal command from the path, then projected onto the safe set.

    pose      (3,) [x, y, theta] in the map frame, from the EKF
    P         (3,3) pose covariance, or None to use the fixed radius
    path      (K, 2) reference path in the map frame, from Module 4
    grid      OccupancyGridMap from Module 3; the obstacle source
    params    ControlParams

    returns
      u      (2,) [v, omega] to send to the motors
      info   dict from safety_filter, plus n_obstacles and the carrot
    """
    obstacles = grid_to_obstacles(grid, pose, params.d_gate, params.n_sectors)
    u_nom, carrot = pure_pursuit_command(pose, path, params.v_desired, params.lookahead_gain, params.L_min, params.L_max, params.wheelbase)
    u, info = safety_filter(u_nom, pose, obstacles, params.v_max, params.omega_max, params.R, params.ell, params.alpha_gain, P=P, k=params.k)
    info['n_obstacles'] = obstacles.shape[0]
    info['carrot'] = carrot
    return u, info
