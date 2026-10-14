import numpy as np
from movement.safety_filter_qp import safety_filter

R, ELL, GAMMA = 0.3, 0.15, 1.0
V_MAX, W_MAX = 1.0, 2.0
POSE = np.array([0.0, 0.0, 0.0])
U_NOM = np.array([0.5, 0.2])


def run(obstacles, u_nom=U_NOM, pose=POSE):
    obs = np.array(obstacles, dtype=float).reshape(-1, 2)
    return safety_filter(u_nom, pose, obs, V_MAX, W_MAX, R, ELL, GAMMA)


def check(name, got, want, tol=1e-6):
    if isinstance(want, str):
        ok = got == want
    else:
        ok = np.allclose(got, want, atol=tol)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {np.asarray(got)}")
        print(f"        want {np.asarray(want)}")
    return ok


ok = True
print()
print("C.5  minimal invasiveness: obstacle 5 m away should change nothing")
u, info = run([[5.0, 0.0]])
ok &= check("u* == u_nom", u, U_NOM)
ok &= check("active is False", info["active"], False)
print(f"        h_min = {info['h_min']:.3f}   deviation = {info['deviation']:.2e}")

print()
print("     no obstacles at all (G = None)")
u, info = safety_filter(U_NOM, POSE, np.zeros((0, 2)), V_MAX, W_MAX, R, ELL, GAMMA)
ok &= check("u* == u_nom", u, U_NOM)
ok &= check("h_min is inf", np.isinf(info["h_min"]), True)

print()
print("D.7  obstacle 0.5 m dead ahead -- hand-computed bound")
# feeler [0.15, 0];  e = [-0.35, 0];  d = 0.35;  h = 0.05
# a_v = -1, a_w = 0  ->  g_row = [1, 0], h_bound = 0.05  ->  v <= 0.05
u, info = run([[0.5, 0.0]])
ok &= check("h = 0.05", info["h_min"], 0.05)
ok &= check("u* == [0.05, 0.2]  (v clipped, turn untouched)", u, [0.05, 0.2])
ok &= check("active is True", info["active"], True)
print(f"        deviation = {info['deviation']:.3f} m/s of speed given up")

print()
print("     obstacle 0.5 m to the LEFT, pure pursuit asking for a hard left")
# feeler [0.15, 0];  e = [0, -0.5];  d = 0.5;  h = 0.2
# a_v = 0, a_w = -0.15  ->  g_row = [0, 0.15], h_bound = 0.2  ->  omega <= 1.3333
u, info = run([[0.15, 0.5]], u_nom=np.array([0.5, 2.0]))
ok &= check("h = 0.2", info["h_min"], 0.2)
ok &= check("u* == [0.5, 4/3]  (turn clipped, speed untouched)", u, [0.5, 0.2 / 0.15])

print()
print("     already inside the barrier -> filter commands REVERSE")
# obstacle at 0.3:  feeler 0.15 away, d = 0.15, h = -0.15  ->  v <= -0.15
u, info = run([[0.3, 0.0]])
ok &= check("h < 0", info["h_min"] < 0, True)
ok &= check("u* == [-0.15, 0.2]", u, [-0.15, 0.2])

print()
print("     infeasible: obstacle in front AND behind, no v satisfies both")
# front at 0.25  ->  v <= -0.20
# behind at -0.3 ->  v >= -0.15      (empty)
u, info = run([[0.25, 0.0], [-0.3, 0.0]])
ok &= check("fail-safe stop", u, [0.0, 0.0])
ok &= check("status reports failure", info["status"], "QP failed")

print()
print("     actuator box still binds when the CBF is slack")
u, info = safety_filter(np.array([5.0, 0.0]), POSE, np.array([[20.0, 0.0]]),
                        V_MAX, W_MAX, R, ELL, GAMMA)
ok &= check("v capped at v_max", u, [V_MAX, 0.0])

print()
print("ALL CHECKPOINTS", "PASSED" if ok else "FAILED")
print()
