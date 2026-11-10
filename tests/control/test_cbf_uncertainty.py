"""Checkpoint E.7 -- uncertainty_margin, plus the P = 0 regression on safety_filter."""

import numpy as np
from lidar_nav.control.cbf_uncertainty import uncertainty_margin
from lidar_nav.control.safety_filter_qp import safety_filter
from tests.testkit import Suite

T = Suite("Safety filter: uncertainty margin")


def check(name, got, want=True, tol=1e-6):
    if isinstance(want, (str, bool, np.bool_)) or isinstance(got, (bool, np.bool_)):
        return T.check(name, got == want)
    return T.near(name, got, want, tol)


ELL, K = 0.15, 2.0
POSE = np.array([0.0, 0.0, 0.0])

AHEAD = np.array([0.5, 0.0])      # J = [-1, 0, 0]
LEFT = np.array([0.15, 0.5])      # J = [0, -1, -0.15]

P_BOTH = np.diag([1e-4, 1e-4, 2.5e-3])    # 1 cm, 1 cm, 0.05 rad
P_HEADING = np.diag([0.0, 0.0, 2.5e-3])   # heading uncertainty only



T.section("zero covariance must produce zero margin")
check("margin = 0", uncertainty_margin(POSE, np.zeros((3, 3)), AHEAD, K, ELL), 0.0)

T.section("obstacle dead ahead: only x uncertainty should count")
# J = [-1, 0, 0];  var = 1*1e-4 = 1e-4;  sigma = 0.01;  k*sigma = 0.02
check("margin = 0.020", uncertainty_margin(POSE, P_BOTH, AHEAD, K, ELL), 0.02)
check("heading-only uncertainty contributes nothing",
      uncertainty_margin(POSE, P_HEADING, AHEAD, K, ELL), 0.0)

T.section("obstacle to the LEFT: y and theta both count")
# J = [0, -1, -0.15]
# var = 1e-4 + 0.0225*2.5e-3 = 1e-4 + 5.625e-5 = 1.5625e-4;  sigma = 0.0125
check("margin = 0.025", uncertainty_margin(POSE, P_BOTH, LEFT, K, ELL), 0.025)
# heading only:  sigma = 0.15 * 0.05 = 0.0075  ->  k*sigma = 0.015
check("heading-only margin = 0.015",
      uncertainty_margin(POSE, P_HEADING, LEFT, K, ELL), 0.015)

T.section("correlation changes the answer (the cross terms are live)")
P_pos = np.array([[1e-4, 0.0, 0.0],
                  [0.0, 1e-4, 4e-4],
                  [0.0, 4e-4, 2.5e-3]])
P_neg = P_pos.copy()
P_neg[1, 2] = P_neg[2, 1] = -4e-4
m_pos = uncertainty_margin(POSE, P_pos, LEFT, K, ELL)
m_neg = uncertainty_margin(POSE, P_neg, LEFT, K, ELL)
m_zero = uncertainty_margin(POSE, P_BOTH, LEFT, K, ELL)
T.info(f"Cov(y,theta) < 0 : {m_neg:.4f}     = 0 : {m_zero:.4f}     > 0 : {m_pos:.4f}")
check("positive correlation inflates, negative deflates",
      m_neg < m_zero < m_pos, True)
check("both P matrices are valid (PSD)",
      min(np.linalg.eigvalsh(P_pos).min(), np.linalg.eigvalsh(P_neg).min()) >= -1e-15,
            True)

T.section("margin grows with k and with uncertainty")
check("k = 3 is 1.5x k = 2",
      uncertainty_margin(POSE, P_BOTH, AHEAD, 3.0, ELL),
            1.5 * uncertainty_margin(POSE, P_BOTH, AHEAD, 2.0, ELL))
check("4x the variance doubles the margin",
      uncertainty_margin(POSE, 4 * P_BOTH, AHEAD, K, ELL),
            2 * uncertainty_margin(POSE, P_BOTH, AHEAD, K, ELL))

T.section("REGRESSION: safety_filter with P = 0 must match P = None exactly")
R, GAMMA, V_MAX, W_MAX = 0.3, 1.0, 1.0, 2.0
u_nom = np.array([0.5, 0.2])
obs = np.array([[0.5, 0.0]])
u_none, i_none = safety_filter(u_nom, POSE, obs, V_MAX, W_MAX, R, ELL, GAMMA)
u_zero, i_zero = safety_filter(u_nom, POSE, obs, V_MAX, W_MAX, R, ELL, GAMMA,
                               P=np.zeros((3, 3)), k=K)
check("same command", u_zero, u_none)
check("same R_eff", i_zero["R_eff"], i_none["R_eff"])

T.section("with real covariance the filter brakes EARLIER")
u_unc, i_unc = safety_filter(u_nom, POSE, obs, V_MAX, W_MAX, R, ELL, GAMMA,
                             P=P_BOTH, k=K)
T.info(f"P = None : R_eff {i_none['R_eff']:.3f}   v* {u_none[0]:.4f}")
T.info(f"P = real : R_eff {i_unc['R_eff']:.3f}   v* {u_unc[0]:.4f}")
check("R_eff inflated by 0.02", i_unc["R_eff"], i_none["R_eff"] + 0.02)
check("v* is smaller", u_unc[0] < u_none[0], True)
# h shrinks by exactly the margin, so the v bound does too: 0.05 - 0.02 = 0.03
check("v* = 0.03 by hand", u_unc[0], 0.03, tol=1e-6)

T.section("margins must NOT accumulate across obstacles")
# three obstacles, all 0.5 m dead ahead-ish. each deserves its own ~0.02 m margin.
# if R_eff accumulates, the third sees R + 0.06 and v* collapses.
far = np.array([[0.5, 0.0], [0.5, 0.001], [0.5, 0.002]])
u_many, i_many = safety_filter(u_nom, POSE, far, V_MAX, W_MAX, R, ELL, GAMMA,
                               P=P_BOTH, k=K)
u_one, i_one = safety_filter(u_nom, POSE, far[:1], V_MAX, W_MAX, R, ELL, GAMMA,
                             P=P_BOTH, k=K)
T.info(f"1 obstacle : R_eff {i_one['R_eff']:.4f}   v* {u_one[0]:.4f}")
T.info(f"3 obstacles: R_eff {i_many['R_eff']:.4f}   v* {u_many[0]:.4f}")
check("R_eff does not grow with obstacle count",
      i_many["R_eff"], i_one["R_eff"], tol=1e-6)

T.done()
