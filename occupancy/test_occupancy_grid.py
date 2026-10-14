from occupancy.occupancy_grid import OccupancyGridMap
import numpy as np

g = OccupancyGridMap(width_m=2.0, height_m=2.0, resolution=0.1,
                     origin_x=0.0, origin_y=0.0,
                     l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0,
                     z_max=12.0)

assert g.W == 20 and g.H == 20
assert g.logodds.shape == (20, 20)
assert g.world_to_cell(0.05, 0.05) == (0, 0)
assert g.world_to_cell(1.55, 1.05) == (15, 10)
assert g.world_to_cell(1.999, 1.999) == (19, 19)
assert g.in_bounds(19, 19) and not g.in_bounds(20, 19) and not g.in_bounds(-1, 0)
assert abs(g.logodds_to_prob(5, 5) - 0.5) < 1e-12

g.update_cell(5, 5, 0.85)
assert abs(g.logodds_to_prob(5, 5) - 0.700) < 1e-3
g.update_cell(5, 5, 0.85)
assert abs(g.logodds_to_prob(5, 5) - 0.845) < 1e-3

for _ in range(50):
    g.update_cell(5, 5, 0.85)
assert g.logodds[5, 5] == 5.0
assert abs(g.logodds_to_prob(5, 5) - 0.9933) < 1e-4

g.update_cell(100, 100, 0.85)   # must not raise

g.update_cell(3, 7, 0.85)
assert abs(g.logodds_to_prob(3, 7) - 0.700) < 1e-3
assert g.logodds[7, 3] == 0.85 and g.logodds[3, 7] == 0.0


def fresh():
    return OccupancyGridMap(width_m=2.0, height_m=2.0, resolution=0.1,
                            origin_x=0.0, origin_y=0.0,
                            l_occ=0.85, l_free=-0.40, l_min=-5.0, l_max=5.0,
                            z_max=12.0, r_min=0.2, clearing_dist=8.0)

POSE = (1.0, 1.0, 0.0)
ok = True

def close(a, b, tol=1e-9):
    return abs(a - b) < tol

# --- single valid hit at 0.5 m straight ahead -> obstacle at cell (15,10)
g = fresh()
g.integrate_scan(POSE, np.array([0.0]), np.array([0.5]))

if not close(g.logodds[10, 15], 0.85):
    print("FAIL hit cell:", g.logodds[10, 15]); ok = False
if not close(g.logodds_to_prob(15, 10), 0.7000, 1e-3):
    print("FAIL hit prob:", g.logodds_to_prob(15, 10)); ok = False
for cx in (10, 11, 12, 13, 14):
    if not close(g.logodds[10, cx], -0.40):
        print(f"FAIL free cell ({cx},10):", g.logodds[10, cx]); ok = False
for cx in (16, 17, 18, 19):
    if g.logodds[10, cx] != 0.0:
        print(f"FAIL behind-obstacle ({cx},10):", g.logodds[10, cx]); ok = False
if g.logodds[5, 5] != 0.0:
    print("FAIL untouched cell:", g.logodds[5, 5]); ok = False

# --- max-range beam: no occupied cell anywhere, free carved through
g2 = fresh()
g2.integrate_scan(POSE, np.array([0.0]), np.array([12.0]))
if (g2.logodds > 0).any():
    print("FAIL phantom obstacle on max-range beam"); ok = False
if not close(g2.logodds[10, 15], -0.40):
    print("FAIL max-range free carve:", g2.logodds[10, 15]); ok = False

# --- saturation
g3 = fresh()
for _ in range(20):
    g3.integrate_scan(POSE, np.array([0.0]), np.array([0.5]))
if g3.logodds[10, 15] != 5.0:
    print("FAIL clamp:", g3.logodds[10, 15]); ok = False
if not close(g3.logodds_to_prob(15, 10), 0.9933, 1e-4):
    print("FAIL clamp prob:", g3.logodds_to_prob(15, 10)); ok = False
if g3.logodds[10, 11] != -5.0:
    print("FAIL free clamp:", g3.logodds[10, 11]); ok = False

# --- discarded beams touch nothing
g4 = fresh()
g4.integrate_scan(POSE, np.array([0.0, 0.0, 0.0]),
                        np.array([0.0, 0.1, float("nan")]))
if g4.logodds.any():
    print("FAIL discarded beams modified the map"); ok = False

print("Section 5.5 checkpoint", "PASSED" if ok else "FAILED")
print("Passed")