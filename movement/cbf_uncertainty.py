import numpy as np


def uncertainty_margin(pose, P, obstacle_xy, k, ell=0.0):
    """
    Extra safety radius implied by the EKF's pose covariance (Part E).

    pose         (3,) [x, y, theta]
    P            (3,3) pose covariance over [x, y, theta] from the EKF
    obstacle_xy  (2,) obstacle centre
    k            how many standard deviations to allow for
    ell          feeler offset, so heading uncertainty contributes too

    returns      float, the amount to add to R
    """
    # the idea: uncertainty only matters ALONG the line of sight to the
    # obstacle -- lateral error does not bring you closer to it.
    #
    # 1. line-of-sight unit vector from the guarded point to the obstacle
    # 2. Jacobian of that point's position w.r.t. [x, y, theta]
    #    (the theta column is where `ell` enters)
    # 3. project P through it to get sigma_los along that direction
    # 4. return k * sigma_los
    #
    # read E.2 for the derivation -- the projection is the whole content
    feeler = np.array([pose[0] + ell * np.cos(pose[2]), pose[1] + ell * np.sin(pose[2])])
    e = feeler - obstacle_xy
    d = np.linalg.norm(e)
    if d < 1e-9:
        return 0.0
    e_norm = e / d

    J = np.array([e_norm[0], e_norm[1], ell * (e_norm @ np.array([-np.sin(pose[2]), np.cos(pose[2])]))])
    sigma_los = np.sqrt(J @ P @ J.T)
    return k * sigma_los



