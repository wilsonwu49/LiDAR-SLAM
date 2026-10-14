import numpy as np
from icp.costs import point_to_point_cost, point_to_line_cost

x = np.linspace(-1, 1, 100)
dst = np.column_stack((x, np.ones(100)))         # wall along y=1.0
src = np.column_stack((x + 0.05, np.ones(100)))  # same wall, slid +0.05 in x (tangential)
normals = np.tile(np.array([0.0, 1.0]), (100, 1)) # every normal points straight up

phi, t = 0.0, np.array([0.0, 0.0])

pl = point_to_line_cost(src, dst, normals, phi, t)
pp = point_to_point_cost(src, dst, phi, t)
print(f"point_to_line_cost = {pl:.6f}  (expect ~0 -- tangential slide is free)")
print(f"point_to_point_cost = {pp:.6f}  (expect clearly > 0)")