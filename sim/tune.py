"""Tune Q and R_enc for CONSISTENCY, not for small error.

Target: mean NEES ~= 3 (the pose state dimension). Above that the filter is
overconfident and the CBF margin is built on a lie.
"""
import numpy as np
from sim.scenarios import run, ROOM, nees, pose_errors


def row(label, r):
    n = nees(r)
    e = pose_errors(r)
    pos = np.linalg.norm(e[:, :2], axis=1)
    sig = np.array([np.sqrt(np.trace(i["P3"][:2, :2]) / 2) for i in r["log"]])
    print(f"  {label:26s} NEES mean {n.mean():9.1f}  median {np.median(n):8.1f}"
          f"   pos err {pos.mean()*1000:7.1f} mm   claimed sigma {sig.mean()*1000:6.2f} mm"
          f"   {'OK' if 0.5 < n.mean() < 10 else 'inconsistent'}")


if __name__ == "__main__":
    print()
    print("baseline, no slip")
    row("as shipped", run(ROOM, [2., 5., 0.], [8., 5.]))

    print()
    print("Q sweep (sigma_a), no slip -- Q absorbs unmodelled dynamics")
    for sa in [0.05, 0.2, 0.5, 1.0, 2.0]:
        row(f"sigma_a = {sa:.2f}",
            run(ROOM, [2., 5., 0.], [8., 5.], ekf_kwargs=dict(sigma_a=sa)))

    print()
    print("R_enc sweep, 15% slip -- distrusting odometry lets ICP correct")
    for re_ in [0.01, 0.05, 0.2, 0.5]:
        row(f"r_enc = {re_:.2f} m/s",
            run(ROOM, [2., 5., 0.], [8., 5.], slip=0.15,
                ekf_kwargs=dict(r_enc=re_), max_cycles=3000))

    print()
    print("both, 15% slip")
    for sa, re_ in [(0.05, 0.01), (0.5, 0.05), (1.0, 0.2), (2.0, 0.5)]:
        row(f"sigma_a {sa:.2f}, r_enc {re_:.2f}",
            run(ROOM, [2., 5., 0.], [8., 5.], slip=0.15,
                ekf_kwargs=dict(sigma_a=sa, r_enc=re_), max_cycles=3000))
