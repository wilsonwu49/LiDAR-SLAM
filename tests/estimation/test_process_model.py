import numpy as np
from lidar_nav.estimation.state import NominalState, IDX
from lidar_nav.estimation.process_model import (propagate_nominal, quat_mul, rotvec_to_quat,
                                            quat_normalize, error_state_F, build_Q)
from lidar_nav.estimation.measurement_models import encoder_h_and_H
from tests.testkit import Suite

T = Suite("EKF: motion model and Jacobians")
g = np.array([0.0, 0.0, -9.81])
still = dict(p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]),
             a_b=np.zeros(3), w_b=np.zeros(3))

T.section("propagation")
x = NominalState(**still)
for _ in range(1000):
    x = propagate_nominal(x, np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.005, g)
T.check("sitting still for 5 s goes nowhere",
        np.allclose(x.p, 0, atol=1e-6) and np.allclose(x.v, 0, atol=1e-6)
        and np.allclose(x.q, [1, 0, 0, 0], atol=1e-6))
x2 = NominalState(**still)
for _ in range(200):
    x2 = propagate_nominal(x2, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 1.0]), 0.005, g)
yaw = 2 * np.arctan2(x2.q[3], x2.q[0])
T.near("spinning at 1 rad/s for 1 s turns 1 rad", yaw, 1.0, 1e-4, " rad")
T.near("spinning in place does not move it", x2.p, np.zeros(3), 1e-4, " m")


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def inject(x: NominalState, dx: np.ndarray) -> NominalState:
    """Apply a 15-vector error dx onto nominal state x (the (+) composition, §2)."""
    new_q = quat_normalize(quat_mul(x.q, rotvec_to_quat(dx[IDX['dth']])))
    return NominalState(
        p=x.p + dx[IDX['dp']], v=x.v + dx[IDX['dv']], q=new_q,
        a_b=x.a_b + dx[IDX['dab']], w_b=x.w_b + dx[IDX['dwb']],
    )

def extract_error(x_ref: NominalState, x_pert: NominalState) -> np.ndarray:
    """Recover the 15-vector error of x_pert relative to x_ref (inverse of inject)."""
    dx = np.zeros(15)
    dx[IDX['dp']] = x_pert.p - x_ref.p
    dx[IDX['dv']] = x_pert.v - x_ref.v
    q_diff = quat_mul(quat_conjugate(x_ref.q), x_pert.q)
    dx[IDX['dth']] = 2 * q_diff[1:4]          # small-angle: q{phi} ~ [1, phi/2]
    dx[IDX['dab']] = x_pert.a_b - x_ref.a_b
    dx[IDX['dwb']] = x_pert.w_b - x_ref.w_b
    return dx


T.section("Jacobians match finite differences")
x0 = NominalState(p=np.zeros(3), v=np.array([0.5, 0.0, 0.0]),
                  q=quat_normalize(rotvec_to_quat(np.array([0.1, 0.05, 0.2]))),
                  a_b=np.array([0.01, -0.02, 0.01]), w_b=np.array([0.001, 0.002, -0.001]))
a_m, w_m, dt, eps = np.array([0.2, 0.1, 9.75]), np.array([0.05, -0.02, 0.3]), 0.005, 1e-6
F = error_state_F(x0, a_m, w_m, dt)
x1 = propagate_nominal(x0, a_m, w_m, dt, g)
F_num = np.zeros((15, 15))
for i in range(15):
    dx = np.zeros(15)
    dx[i] = eps
    F_num[:, i] = extract_error(x1, propagate_nominal(inject(x0, dx), a_m, w_m, dt, g)) / eps
err = np.max(np.abs(F_num - F))
T.check("motion Jacobian F", err < 2e-4, f"max difference {err:.1e}")

xe = NominalState(p=np.zeros(3), v=np.array([0.5, 0.1, -0.05]),
                  q=quat_normalize(rotvec_to_quat(np.array([0.1, 0.05, 0.2]))),
                  a_b=np.zeros(3), w_b=np.zeros(3))
h0, H = encoder_h_and_H(xe)
H_num = np.zeros((3, 15))
for i in range(15):
    dx = np.zeros(15)
    dx[i] = eps
    H_num[:, i] = (encoder_h_and_H(inject(xe, dx))[0] - h0) / eps
err = np.max(np.abs(H_num - H))
T.check("encoder Jacobian H", err < 1e-4, f"max difference {err:.1e}")

T.section("process noise over 100 s sitting still")
sa, sg = 1.57e-3, 1.40e-4
x, P, dt = NominalState(**still), np.zeros((15, 15)), 0.005
var = {}
for step in range(1, int(100 / dt) + 1):
    Fk = error_state_F(x, np.array([0.0, 0.0, 9.81]), np.zeros(3), dt)
    P = Fk @ P @ Fk.T + build_Q(sa, sg, 0.0, 0.0, dt)
    x = propagate_nominal(x, np.array([0.0, 0.0, 9.81]), np.zeros(3), dt, g)
    if step in (int(50 / dt), int(100 / dt)):
        var[step * dt] = (np.diag(P[IDX['dth'], IDX['dth']]).copy(), np.diag(P[IDX['dp'], IDX['dp']]).copy())
rel = np.abs(var[100.0][0] - sg ** 2 * 100) / (sg ** 2 * 100)
T.check("heading variance grows as sigma_g^2 * t", np.all(rel < 0.01), f"off by {rel.max() * 100:.2f}%")
ratio = var[100.0][1] / var[50.0][1]
T.check("position variance grows much faster than linearly", np.all(ratio > 4),
        f"doubling time multiplies it by {ratio.min():.1f} (cubic would be 8)")

T.done()
