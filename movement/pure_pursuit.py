import numpy as np

MIN_GOAL_DISTANCE = 0.05  # m

def pure_pursuit_command(pose, path, v_desired, lookahead_gain,
                         L_min, L_max, wheelbase):
    """
    Compute the nominal differential-drive command from pure pursuit.

    Parameters
    ----------
    pose : (3,) array  -> [x, y, theta] in the map frame (from Module 1 EKF).
    path : (N, 2) array -> reference waypoints in the map frame (from Module 4 A*).
    v_desired : float   -> commanded forward speed (m/s).
    lookahead_gain : float  -> k_v; L_d = clamp(k_v * v_desired, L_min, L_max).
    L_min, L_max : float    -> lookahead clamps (m).
    wheelbase : float       -> distance between wheels (m).

    Returns
    -------
    u_nom : (2,) array -> [v, omega], the nominal command (omega in rad/s).
    lookahead_pt_body : (2,) array -> lookahead point in the BODY frame (for debugging/plots).
    """

    L_d = np.clip(lookahead_gain * v_desired, L_min, L_max)
    dx = path[:,0] - pose[0]
    dy = path[:,1] - pose[1]

    theta = pose[2]
    x_body = dx * np.cos(theta) + dy * np.sin(theta)
    y_body = -dx * np.sin(theta) + dy * np.cos(theta)

    distances = np.sqrt(x_body**2 + y_body**2)
    i_near = int(np.argmin(distances))
    
    found = False
    for i in range(i_near, len(path)):
        if distances[i] >= L_d:
            found = True
            break

    if not found and L_actual < MIN_GOAL_DISTANCE:
        return np.array([0.0, 0.0]), np.array([x_L, y_L])

    elif L_actual < 1e-9:
        return np.array([v_desired, 0.0]), np.array([x_L, y_L])

    elif i == i_near:
        x_L, y_L = x_body[i], y_body[i]

    else:
        t = (L_d - distances[i-1]) / (distances[i] - distances[i-1])
        x_L = x_body[i-1] + t * (x_body[i] - x_body[i-1])
        y_L = y_body[i-1] + t * (y_body[i] - y_body[i-1])

    L_actual = np.max(np.sqrt(x_L**2 + y_L**2), MIN_GOAL_DISTANCE)
    gamma = 2 * y_L / L_actual**2
    omega = gamma * v_desired
    return np.array([v_desired, omega]), np.array([x_L, y_L])