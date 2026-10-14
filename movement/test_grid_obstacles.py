import numpy as np
from occupancy.occupancy_grid import OccupancyGridMap
from movement.obstacles import grid_to_obstacles

D_GATE, NS = 0.75, 36

def make(w=10.0, h=10.0, s=0.05, ox=0.0, oy=0.0):
    return OccupancyGridMap(w, h, s, ox, oy, l_occ=0.85, l_free=-0.40,
                            l_min=-5.0, l_max=5.0, z_max=12.0)

def occupy(g, wx, wy, val=5.0):
    cx, cy = g.world_to_cell(wx, wy)
    g.logodds[cy, cx] = val

def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)

ok = True
print()
print("empty grid")
g = make()
ok &= check("no obstacles -> (0,2)", grid_to_obstacles(g, np.array([5.,5.,0.]), D_GATE, NS).shape == (0, 2))

print()
print("single occupied cell, 0.5 m ahead of a robot at (5,5)")
g = make(); occupy(g, 5.5, 5.0)
obs = grid_to_obstacles(g, np.array([5., 5., 0.]), D_GATE, NS)
ok &= check("one point", len(obs) == 1)
# cell containing 5.5 starts at 5.50, centre 5.525;  y cell starts 5.00, centre 5.025
ok &= check(f"at the cell CENTRE (5.525, 5.025), got {obs[0]}",
            np.allclose(obs[0], [5.525, 5.025]))

print()
print("gate: a cell 1.0 m away is outside d_gate = 0.75")
g = make(); occupy(g, 6.0, 5.0)
ok &= check("dropped", len(grid_to_obstacles(g, np.array([5.,5.,0.]), D_GATE, NS)) == 0)

print()
print("[cy,cx] convention: asymmetric placement")
g = make(); occupy(g, 5.3, 5.0)          # 0.3 m in +x only
obs = grid_to_obstacles(g, np.array([5., 5., 0.]), D_GATE, NS)
ok &= check(f"found in +x, not +y: {obs[0]}", obs[0][0] > 5.2 and abs(obs[0][1] - 5.025) < 1e-9)

print()
print("domination: wall of cells, one per wedge")
g = make()
for wy in np.arange(4.4, 5.65, 0.05):
    occupy(g, 5.4, wy)
obs = grid_to_obstacles(g, np.array([5., 5., 0.]), D_GATE, NS)
print(f"        {int(np.sum(g.logodds >= 0.85))} occupied cells -> {len(obs)} points")
ok &= check("at most one per wedge", len(obs) <= NS)
ok &= check("all inside the gate",
            np.all(np.linalg.norm(obs - np.array([5., 5.]), axis=1) < D_GATE + 1e-9))

print()
print("map edge: robot at (0.2, 0.2) must not wrap to the far corner")
g = make(); occupy(g, 9.9, 9.9)          # diagonally opposite
obs = grid_to_obstacles(g, np.array([0.2, 0.2, 0.0]), D_GATE, NS)
ok &= check("nothing found (no negative-index wrap)", len(obs) == 0)
occupy(g, 0.4, 0.2)
obs = grid_to_obstacles(g, np.array([0.2, 0.2, 0.0]), D_GATE, NS)
ok &= check(f"the real near cell IS found: {obs}", len(obs) == 1 and abs(obs[0][0] - 0.425) < 1e-9)

print()
print("negative origin")
g = make(ox=-5.0, oy=-5.0); occupy(g, 0.4, 0.0)
obs = grid_to_obstacles(g, np.array([0., 0., 0.]), D_GATE, NS)
ok &= check(f"origin offset handled: {obs[0]}", np.allclose(obs[0], [0.425, 0.025]))

print()
print("heading rotates the wedge grid, so WHICH cells win changes --")
print("but every survivor must still be a real wall cell inside the gate")
g = make()
for wy in np.arange(4.6, 5.45, 0.05):
    occupy(g, 5.4, wy)
for th in [0.0, 0.5, 1.3, -2.0, np.pi]:
    obs = grid_to_obstacles(g, np.array([5., 5., th]), D_GATE, NS)
    on_wall = np.allclose(obs[:, 0], 5.425)
    inside = np.all(np.linalg.norm(obs - np.array([5., 5.]), axis=1) < D_GATE + 1e-9)
    ok &= check(f"theta = {th:5.2f}: {len(obs):2d} pts, all on the wall and in range",
                on_wall and inside)
# the looseness this costs is bounded by the in-wedge range spread
spread = 0.425 * (1 / np.cos(np.pi / NS) - 1)
print(f"        worst-case extra slack from wedge rotation: {spread*1000:.1f} mm")

print()
print("ALL CHECKS", "PASSED" if ok else "FAILED")
print()
