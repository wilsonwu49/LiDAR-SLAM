import numpy as np
from lidar_nav.estimation.state import NominalState
from lidar_nav.estimation.measurement_models import encoder_h_and_H
from lidar_nav.estimation.kalman import kalman_update, inject_and_reset
from lidar_nav.estimation.consistency import nis, nis_bounds
from tests.testkit import Suite


def run_nis_checkpoint(R_true, R_used, N=500, seed=0):
    rng = np.random.default_rng(seed)
    true_v = np.array([1.0, 0.0, 0.0])            # the real body velocity, known only to this test

    x = NominalState(
        p=np.zeros(3), v=np.array([0.5, 0.0, 0.0]),   # deliberately wrong starting velocity
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        a_b=np.zeros(3), w_b=np.zeros(3),
    )
    P = np.eye(15) * 0.1

    nis_values = []
    for _ in range(N):
        v_body_meas = true_v + rng.multivariate_normal(np.zeros(3), R_true)   # simulated noisy reading

        h, H = encoder_h_and_H(x)              # predicted reading + Jacobian, BEFORE this update
        S = H @ P @ H.T + R_used               # what the filter EXPECTS this surprise to look like
        y = v_body_meas - h
        nis_values.append(nis(y, S))

        dx_hat, P_plus = kalman_update(P, H, R_used, y)    # actually apply the correction
        x, P = inject_and_reset(x, dx_hat, P_plus)         # ungated on purpose: see note below

    mean_nis = np.mean(nis_values)
    low, high = nis_bounds(m=3, N=N)
    return mean_nis, low, high


T = Suite("EKF: NIS consistency check")
R_true = np.diag([0.01, 0.01, 0.01])

T.section("500 encoder updates, mean NIS should sit in the 95% band around 3")
m, lo, hi = run_nis_checkpoint(R_true=R_true, R_used=R_true)
T.check("filter told the true noise is consistent", lo <= m <= hi, f"mean NIS {m:.2f}, band [{lo:.2f}, {hi:.2f}]")
m, lo, hi = run_nis_checkpoint(R_true=R_true, R_used=R_true / 2)
T.check("filter told half the noise is caught as overconfident", m > hi, f"mean NIS {m:.2f} > {hi:.2f}")

T.done()
