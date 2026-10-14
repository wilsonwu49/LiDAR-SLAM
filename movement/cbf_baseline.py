import numpy as np


def fixed_radius_cbf_constraint(pose, obstacle_xy, R, ell, alpha_gain):
    """
    One CBF constraint row for a circular obstacle, measured at a point
    `ell` ahead of the robot (Part D.4).

    pose         (3,) [x, y, theta] in the map frame
    obstacle_xy  (2,) obstacle centre in the map frame
    R            safety radius around the obstacle [m]
    ell          feeler offset ahead of the robot [m]
    alpha_gain   gamma, from alpha(h) = gamma * h

    returns
      g_row    (2,)   the QP row, [-a_v, -a_omega]
      h_bound  float  alpha_gain * h, the QP right-hand side
      h        float  the barrier value itself, for logging

    constraint is   g_row @ u <= h_bound
    """
    # 1. feeler point  p_L = (x + ell*cos th, y + ell*sin th)
    # 2. e = p_L - obstacle_xy,  d = norm(e),  h = d - R
    # 3. e_hat = e / d        (guard d near zero)
    # 4. a_v = e_hat . [ cos th,  sin th ]
    # 5. a_w = ell * ( e_hat . [ -sin th, cos th ] )
    # 6. return np.array([-a_v, -a_w]), alpha_gain * h, h

    feeler = np.array([pose[0] + ell * np.cos(pose[2]), pose[1] + ell * np.sin(pose[2])]) # extended point
    e = feeler - obstacle_xy # error from extended vs obstacle
    d = np.linalg.norm(e) # distances from extended vs obstacle
    h = d - R # difference between distance and safe radius
    e_norm = e / d if d > 1e-9 else np.zeros_like(e) # normalized error vector
    a_v = e_norm @ np.array([np.cos(pose[2]), np.sin(pose[2])]) # how much velocity is pointing towards the obstacle
    # turning swings the feeler sideways at ell*omega; this is how much of that
    # motion lies along the escape direction (ell is the lever arm)
    a_w = ell * (e_norm @ np.array([-np.sin(pose[2]), np.cos(pose[2])]))
    return np.array([-a_v, -a_w]), alpha_gain * h, h