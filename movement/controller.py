"""Top-level navigation control step: scan -> obstacles -> pure pursuit -> safety filter."""

from dataclasses import dataclass

import numpy as np

from movement.obstacles import scan_to_obstacles
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
    r_safety: float = 0.15
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

    def __post_init__(self):
        # the planner must be more conservative than the reactive layer, or the
        # CBF fights A* over obstacles the planner already routed around
        if self.r_robot + self.r_safety < self.R + self.ell:
            raise ValueError(
                f"planner inflation {self.r_robot + self.r_safety:.3f} m is smaller "
                f"than CBF reach {self.R + self.ell:.3f} m; the filter will brake "
                f"on walls A* considers clear")

    @property
    def d_gate(self):
        """Range beyond which a constraint provably cannot bind."""
        return self.R + self.v_max / self.alpha_gain


def control_step(pose, P, path, ranges, bearings, params):
    """
    One control cycle: nominal command from the path, then projected onto the safe set.

    pose      (3,) [x, y, theta] in the map frame, from the EKF
    P         (3,3) pose covariance, or None to use the fixed radius
    path      (K, 2) reference path in the map frame, from Module 4
    ranges    (N,) scan ranges [m]
    bearings  (N,) scan bearings in the body frame [rad]
    params    ControlParams

    returns
      u      (2,) [v, omega] to send to the motors
      info   dict from safety_filter, plus n_obstacles and the carrot
    """
    # 1. scan_to_obstacles(...) with params.d_gate and params.n_sectors
    # 2. pure_pursuit_command(...) -> u_nom, carrot
    # 3. safety_filter(u_nom, ...) -> u, info      (pass P and params.k)
    # 4. add n_obstacles and carrot to info, return
    ...
