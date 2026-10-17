"""Edge cases and robustness sweeps on the full navigation stack."""
import numpy as np
from sim.test_navigate import run, ROOM, OPEN_ROOM, clearances

def line(name, r, extra=""):
    c = clearances(r)
    print(f"  {name:34s} {r['outcome']:8s} cyc {r['cycles']:5d} "
          f"replan {r['replans']:3d} stall {r['stalls']:2d} "
          f"clear {c.min()*1000:7.1f} mm  {extra}")
    return c

print()
print("=" * 86)
print("1. DOORWAY WIDTH   (planner inflation 0.33 m, so needs > 0.66 m of gap)")
for gap in [2.0, 1.2, 0.8, 0.70, 0.60, 0.50]:
    seg = np.vstack([OPEN_ROOM,
                     [[5.0, 1.0, 5.0, 5.0 - gap/2]],
                     [[5.0, 5.0 + gap/2, 5.0, 9.0]]])
    r = run(seg, [2.0, 5.0, 0.0], [8.0, 5.0], max_cycles=1500)
    line(f"gap {gap:.2f} m", r)

print()
print("=" * 86)
print("2. GOAL PLACEMENT")
r = run(ROOM, [2.0, 5.0, 0.0], [8.95, 5.0], max_cycles=1500)
line("goal 0.05 m from the far wall", r)
r = run(ROOM, [2.0, 5.0, 0.0], [12.0, 5.0], max_cycles=600)
line("goal outside the map", r)
r = run(ROOM, [2.0, 5.0, 0.0], [2.0, 5.0], max_cycles=600)
line("goal == start", r)
r = run(ROOM, [2.0, 5.0, 0.0], [5.0, 2.0], max_cycles=1500)
line("goal inside a wall", r)

print()
print("=" * 86)
print("3. START POSE")
r = run(ROOM, [2.0, 5.0, np.pi], [8.0, 5.0], max_cycles=2500)
line("facing 180 deg away from goal", r)
r = run(ROOM, [1.2, 5.0, 0.0], [8.0, 5.0], max_cycles=2500)
line("start 0.2 m from the wall", r)
r = run(ROOM, [4.6, 5.0, 0.0], [8.0, 5.0], max_cycles=1500)
line("start just inside the doorway", r)

print()
print("=" * 86)
print("4. SENSOR DEGRADATION")
for s in [0.0, 0.02, 0.05, 0.10, 0.20]:
    r = run(ROOM, [2.0, 5.0, 0.0], [8.0, 5.0], sigma_range=s, max_cycles=2000)
    est = np.linalg.norm(r["true_final"][:2] - r["est_final"][:2])
    line(f"range noise sigma {s:.2f} m", r, f"est err {est*1000:6.1f} mm")

print()
print("=" * 86)
print("5. WHEEL SLIP")
for sl in [0.0, 0.05, 0.15, 0.30, 0.50]:
    r = run(ROOM, [2.0, 5.0, 0.0], [8.0, 5.0], slip=sl, max_cycles=3000)
    est = np.linalg.norm(r["true_final"][:2] - r["est_final"][:2])
    rej = sum(1 for i in r["log"] if i["gates"][1] is False)
    line(f"slip {sl*100:3.0f}%", r, f"est err {est*1000:6.1f} mm, icp rej {rej}")
