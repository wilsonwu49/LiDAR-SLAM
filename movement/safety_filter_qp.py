import numpy as np
from qpsolvers import solve_qp

from movement.cbf_baseline import fixed_radius_cbf_constraint
from movement.cbf_uncertainty import uncertainty_margin


def safety_filter(u_nom, pose, obstacles, v_max, omega_max,
                  R, ell, alpha_gain, H=None, P=None, k=0.0):
    """
    Project a nominal command onto the safe set (Parts C and G).

    u_nom       (2,) [v, omega] from pure pursuit
    pose        (3,) [x, y, theta]
    obstacles   (M, 2) obstacle centres in the map frame
    v_max       forward speed limit [m/s]
    omega_max   turn rate limit [rad/s]
    R           base safety radius [m]
    ell         feeler offset [m]
    alpha_gain  gamma
    H           (2,2) deviation weighting, identity if None
    P           (3,3) pose covariance from the EKF, or None for fixed radius
    k           margin multiplier; R_eff = R + k * sigma_los when P is given

    returns
      u_star  (2,) the filtered command
      info    dict(active, deviation, h_min, status, R_eff)
    """
    # 1. H defaults to identity;  P_qp = 2H,  q = -2 H u_nom
    # 2. for each obstacle:
    #       R_eff = R  (+ uncertainty_margin(...) when P is not None)
    #       one row from fixed_radius_cbf_constraint, stacked into G and h_qp
    # 3. actuator bounds:  lb = [-v_max, -omega_max], ub = [v_max, omega_max]
    # 4. u_star = solve_qp(P_qp, q, G, h_qp, lb=lb, ub=ub, solver="quadprog")
    # 5. CHECK THE STATUS. if the solve failed, fall back to a safe default
    #    (stop, or the last known-safe command) -- never pass garbage to motors
    # 6. info: was any constraint active, ||u_star - u_nom||, min h over
    #    obstacles, solver status, the R_eff used
    ...