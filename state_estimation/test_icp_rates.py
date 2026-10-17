"""Hand-computable checks for ESEKF.update_icp_rates.

ICP is fused as a rate, not a pose: the two velocity rows share the encoder's
measurement model, and the turn-rate row books its disagreement with the gyro
to the gyro bias, which is the only state that can explain it.
"""

import numpy as np

from state_estimation.ekf import ESEKF
from state_estimation.state import NominalState, IDX

WB_Z = IDX['dwb'].start + 2


def fresh(v=(0.0, 0.0, 0.0), w_b=(0.0, 0.0, 0.0), yaw=0.0, p_scale=1e-2):
    x = NominalState(p=np.zeros(3), v=np.array(v, dtype=float),
                     q=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                     a_b=np.zeros(3), w_b=np.array(w_b, dtype=float))
    P = np.eye(15) * p_scale
    return ESEKF(x, P, (0.05, 0.005, 1e-4, 1e-5),
                 np.eye(3) * 1e-4, np.diag([1e-3, 1e-3, 1e-4]))


R_V = np.diag([1e-4, 1e-4, 1e-4])
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {extra}")
    ok &= bool(cond)


print()
print("1. a forward-velocity measurement moves v toward it")
# prior sigma_v is 0.1 m/s, so keep the innovation inside the chi-squared gate
e = fresh()
accepted = e.update_icp_rates(np.array([0.1, 0.0]), 0.0, np.zeros(3), R_V)
check("accepted", accepted)
check("v_x increased toward 0.1", 0.0 < e.x.v[0] <= 0.1, f"v = {e.x.v[:2].round(4)}")
check("v_y untouched", abs(e.x.v[1]) < 1e-9)
check("P shrank", np.trace(e.P[3:6, 3:6]) < 3e-2)

print()
print("2. a zero innovation changes nothing")
e = fresh(v=(0.4, 0.0, 0.0))
before_v, before_wb = e.x.v.copy(), e.x.w_b.copy()
e.update_icp_rates(np.array([0.4, 0.0]), 0.0, np.zeros(3), R_V)
check("v unchanged", np.allclose(e.x.v, before_v, atol=1e-12))
check("w_b unchanged", np.allclose(e.x.w_b, before_wb, atol=1e-12))

print()
print("3. ICP says turning, gyro says still -> the gap goes to GYRO BIAS")
# h[2] = (w_m - w_b)_z = 0;  measured = +0.1  ->  y = +0.1
# H[2, wb_z] = -1, so the correction must DECREASE w_b to raise h
e = fresh()
e.update_icp_rates(np.array([0.0, 0.0]), 0.1, np.zeros(3), R_V)
check("w_b_z went negative", e.x.w_b[2] < 0, f"w_b_z = {e.x.w_b[2]:+.5f}")
check("the other bias axes are untouched", np.allclose(e.x.w_b[:2], 0.0, atol=1e-12))
check("velocity untouched by a pure yaw-rate innovation",
      np.allclose(e.x.v[:2], 0.0, atol=1e-9), f"v = {e.x.v[:2].round(6)}")

print()
print("4. opposite sign gives the opposite correction")
e = fresh()
e.update_icp_rates(np.array([0.0, 0.0]), -0.1, np.zeros(3), R_V)
check("w_b_z went positive", e.x.w_b[2] > 0, f"w_b_z = {e.x.w_b[2]:+.5f}")

print()
print("5. gyro bias is observable: a biased gyro gets corrected toward the truth")
# the gyro reads +0.1 while ICP sees no rotation, so the bias really is +0.1
e = fresh(w_b=(0.0, 0.0, 0.0))
for _ in range(40):
    e.update_icp_rates(np.array([0.0, 0.0]), 0.0, np.array([0.0, 0.0, 0.1]), R_V)
check("w_b_z converged toward +0.1", 0.05 < e.x.w_b[2] <= 0.11,
      f"w_b_z = {e.x.w_b[2]:+.5f}")

print()
print("6. the NIS gate rejects a wild measurement")
e = fresh(p_scale=1e-6)
check("absurd velocity rejected",
      e.update_icp_rates(np.array([50.0, 0.0]), 0.0, np.zeros(3), R_V) is False)

print()
print("7. heading rotates the velocity rows (robot facing +y)")
# body-frame v_x = 0.1 while facing +y should push WORLD v toward +y
e = fresh(yaw=np.pi / 2)
e.update_icp_rates(np.array([0.1, 0.0]), 0.0, np.zeros(3), R_V)
check("world v_y increased, v_x did not", e.x.v[1] > 0 and abs(e.x.v[0]) < 1e-6,
      f"v = {e.x.v[:2].round(4)}")

print()
print("ALL CHECKS", "PASSED" if ok else "FAILED")
print()
