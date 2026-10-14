"""Checkpoint E.7 -- uncertainty_margin, plus the P = 0 regression on safety_filter."""

import numpy as np
from movement.cbf_uncertainty import uncertainty_margin
from movement.safety_filter_qp import safety_filter

ELL, K = 0.15, 2.0
POSE = np.array([0.0, 0.0, 0.0])

AHEAD = np.array([0.5, 0.0])      # J = [-1, 0, 0]
LEFT = np.array([0.15, 0.5])      # J = [0, -1, -0.15]

P_BOTH = np.diag([1e-4, 1e-4, 2.5e-3])    # 1 cm, 1 cm, 0.05 rad
P_HEADING = np.diag([0.0, 0.0, 2.5e-3])   # heading uncertainty only


def check(name, got, want, tol=1e-9):
    ok = got == want if isinstance(want, (bool, str)) else np.allclose(got, want, atol=tol)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {np.asarray(got)}")
        print(f"        want {np.asarray(want)}")
    return ok


ok = True

print()
print("E.7a  zero covariance must produce zero margin")
ok &= check("margin = 0", uncertainty_margin(POSE, np.zeros((3, 3)), AHEAD, K, ELL), 0.0)

print()
print("E.7b  obstacle dead ahead -- only x uncertainty should count")
# J = [-1, 0, 0];  var = 1*1e-4 = 1e-4;  sigma = 0.01;  k*sigma = 0.02
ok &= check("margin = 0.020", uncertainty_margin(POSE, P_BOTH, AHEAD, K, ELL), 0.02)
ok &= check("heading-only uncertainty contributes nothing",
            uncertainty_margin(POSE, P_HEADING, AHEAD, K, ELL), 0.0)

print()
print("E.7c  obstacle to the LEFT -- y and theta both count")
# J = [0, -1, -0.15]
# var = 1e-4 + 0.0225*2.5e-3 = 1e-4 + 5.625e-5 = 1.5625e-4;  sigma = 0.0125
ok &= check("margin = 0.025", uncertainty_margin(POSE, P_BOTH, LEFT, K, ELL), 0.025)
# heading only:  sigma = 0.15 * 0.05 = 0.0075  ->  k*sigma = 0.015
ok &= check("heading-only margin = 0.015",
            uncertainty_margin(POSE, P_HEADING, LEFT, K, ELL), 0.015)

print()
print("E.7d  correlation changes the answer (the cross terms are live)")
P_pos = np.array([[1e-4, 0.0, 0.0],
                  [0.0, 1e-4, 4e-4],
                  [0.0, 4e-4, 2.5e-3]])
P_neg = P_pos.copy()
P_neg[1, 2] = P_neg[2, 1] = -4e-4
m_pos = uncertainty_margin(POSE, P_pos, LEFT, K, ELL)
m_neg = uncertainty_margin(POSE, P_neg, LEFT, K, ELL)
m_zero = uncertainty_margin(POSE, P_BOTH, LEFT, K, ELL)
print(f"        Cov(y,theta) < 0 : {m_neg:.4f}     = 0 : {m_zero:.4f}     > 0 : {m_pos:.4f}")
ok &= check("positive correlation inflates, negative deflates",
            m_neg < m_zero < m_pos, True)
ok &= check("both P matrices are valid (PSD)",
            min(np.linalg.eigvalsh(P_pos).min(), np.linalg.eigvalsh(P_neg).min()) >= -1e-15,
            True)

print()
print("E.7e  margin grows with k and with uncertainty")
ok &= check("k = 3 is 1.5x k = 2",
            uncertainty_margin(POSE, P_BOTH, AHEAD, 3.0, ELL),
            1.5 * uncertainty_margin(POSE, P_BOTH, AHEAD, 2.0, ELL))
ok &= check("4x the variance doubles the margin",
            uncertainty_margin(POSE, 4 * P_BOTH, AHEAD, K, ELL),
            2 * uncertainty_margin(POSE, P_BOTH, AHEAD, K, ELL))

print()
print("E.7f  REGRESSION: safety_filter with P = 0 must match P = None exactly")
R, GAMMA, V_MAX, W_MAX = 0.3, 1.0, 1.0, 2.0
u_nom = np.array([0.5, 0.2])
obs = np.array([[0.5, 0.0]])
u_none, i_none = safety_filter(u_nom, POSE, obs, V_MAX, W_MAX, R, ELL, GAMMA)
u_zero, i_zero = safety_filter(u_nom, POSE, obs, V_MAX, W_MAX, R, ELL, GAMMA,
                               P=np.zeros((3, 3)), k=K)
ok &= check("same command", u_zero, u_none)
ok &= check("same R_eff", i_zero["R_eff"], i_none["R_eff"])

print()
print("E.7g  with real covariance the filter brakes EARLIER")
u_unc, i_unc = safety_filter(u_nom, POSE, obs, V_MAX, W_MAX, R, ELL, GAMMA,
                             P=P_BOTH, k=K)
print(f"        P = None : R_eff {i_none['R_eff']:.3f}   v* {u_none[0]:.4f}")
print(f"        P = real : R_eff {i_unc['R_eff']:.3f}   v* {u_unc[0]:.4f}")
ok &= check("R_eff inflated by 0.02", i_unc["R_eff"], i_none["R_eff"] + 0.02)
ok &= check("v* is smaller", u_unc[0] < u_none[0], True)
# h shrinks by exactly the margin, so the v bound does too: 0.05 - 0.02 = 0.03
ok &= check("v* = 0.03 by hand", u_unc[0], 0.03, tol=1e-6)

print()
print("E.7h  margins must NOT accumulate across obstacles")
# three obstacles, all 0.5 m dead ahead-ish. each deserves its own ~0.02 m margin.
# if R_eff accumulates, the third sees R + 0.06 and v* collapses.
far = np.array([[0.5, 0.0], [0.5, 0.001], [0.5, 0.002]])
u_many, i_many = safety_filter(u_nom, POSE, far, V_MAX, W_MAX, R, ELL, GAMMA,
                               P=P_BOTH, k=K)
u_one, i_one = safety_filter(u_nom, POSE, far[:1], V_MAX, W_MAX, R, ELL, GAMMA,
                             P=P_BOTH, k=K)
print(f"        1 obstacle : R_eff {i_one['R_eff']:.4f}   v* {u_one[0]:.4f}")
print(f"        3 obstacles: R_eff {i_many['R_eff']:.4f}   v* {u_many[0]:.4f}")
ok &= check("R_eff does not grow with obstacle count",
            i_many["R_eff"], i_one["R_eff"], tol=1e-6)

print()
print("ALL CHECKPOINTS", "PASSED" if ok else "FAILED")
print()
