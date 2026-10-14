import numpy as np
from state_estimation.state import NominalState
from state_estimation.measurement_models import encoder_h_and_H
from state_estimation.kalman import kalman_update, inject_and_reset
from state_estimation.consistency import nis, nis_bounds


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
    print(f"mean(NIS)={mean_nis:.3f}  expected band=[{low:.3f},{high:.3f}]  "
          f"in_band={low <= mean_nis <= high}")
    return mean_nis


if __name__ == "__main__":
    R_true = np.diag([0.01, 0.01, 0.01])       # the sensor's REAL noise, used to generate data

    print("Case 1 — filter told the correct R:")
    run_nis_checkpoint(R_true=R_true, R_used=R_true)

    print("Case 2 — filter told R is half as big (overconfident):")
    run_nis_checkpoint(R_true=R_true, R_used=R_true / 2)