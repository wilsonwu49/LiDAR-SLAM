import numpy as np
from lidar_nav.control.safety_filter_qp import safety_filter
from tests.testkit import Suite

T = Suite("Safety filter: QP behaviour")


def check(name, got, want=True, tol=1e-6):
    if isinstance(want, (str, bool, np.bool_)) or isinstance(got, (bool, np.bool_)):
        return T.check(name, got == want)
    return T.near(name, got, want, tol)


R, ELL, GAMMA = 0.3, 0.15, 1.0
V_MAX, W_MAX = 1.0, 2.0
POSE = np.array([0.0, 0.0, 0.0])
U_NOM = np.array([0.5, 0.2])


def run(obstacles, u_nom=U_NOM, pose=POSE):
    obs = np.array(obstacles, dtype=float).reshape(-1, 2)
    return safety_filter(u_nom, pose, obs, V_MAX, W_MAX, R, ELL, GAMMA)


T.section("obstacle 5 m away")
u, info = run([[5.0, 0.0]])
check("command passes through unchanged", u, U_NOM)
check("filter reports itself idle", info["active"], False)
T.info(f"h_min = {info['h_min']:.3f}   deviation = {info['deviation']:.2e}")

T.section("no obstacles at all")
u, info = safety_filter(U_NOM, POSE, np.zeros((0, 2)), V_MAX, W_MAX, R, ELL, GAMMA)
check("command passes through unchanged", u, U_NOM)
check("no barrier value to report", np.isinf(info["h_min"]), True)

T.section("obstacle 0.5 m dead ahead (hand-computed)")
# feeler [0.15, 0];  e = [-0.35, 0];  d = 0.35;  h = 0.05
# a_v = -1, a_w = 0  ->  g_row = [1, 0], h_bound = 0.05  ->  v <= 0.05
u, info = run([[0.5, 0.0]])
check("5 cm of margin left", info["h_min"], 0.05)
check("speed cut to 0.05 m/s, turn left alone", u, [0.05, 0.2])
check("filter reports itself active", info["active"], True)
T.info(f"deviation = {info['deviation']:.3f} m/s of speed given up")

T.section("obstacle 0.5 m to the left, steering hard left (hand-computed)")
# feeler [0.15, 0];  e = [0, -0.5];  d = 0.5;  h = 0.2
# a_v = 0, a_w = -0.15  ->  g_row = [0, 0.15], h_bound = 0.2  ->  omega <= 1.3333
u, info = run([[0.15, 0.5]], u_nom=np.array([0.5, 2.0]))
check("20 cm of margin left", info["h_min"], 0.2)
check("turn cut to 1.33 rad/s, speed left alone", u, [0.5, 0.2 / 0.15])

T.section("already too close")
# obstacle at 0.3:  feeler 0.15 away, d = 0.15, h = -0.15  ->  v <= -0.15
u, info = run([[0.3, 0.0]])
check("robot is already inside the barrier", info["h_min"] < 0, True)
check("filter backs it out at 0.15 m/s", u, [-0.15, 0.2])

T.section("boxed in front and behind: no safe command exists")
# front at 0.25  ->  v <= -0.20
# behind at -0.3 ->  v >= -0.15      (empty)
u, info = run([[0.25, 0.0], [-0.3, 0.0]])
check("falls back to a full stop", u, [0.0, 0.0])
check("and reports the failure", info["status"], "QP failed")

T.section("asking for more speed than the motors have")
u, info = safety_filter(np.array([5.0, 0.0]), POSE, np.array([[20.0, 0.0]]),
                        V_MAX, W_MAX, R, ELL, GAMMA)
check("speed capped at the motor limit", u, [V_MAX, 0.0])

T.done()
