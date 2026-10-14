from occupancy.bresenham import bresenham_cells


def ascii_line(cells, w, h):
    grid = [["." for _ in range(w)] for _ in range(h)]
    for (cx, cy) in cells:
        if 0 <= cx < w and 0 <= cy < h:
            grid[cy][cx] = "#"
    return "\n".join("".join(row) for row in reversed(grid))


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      got  {got}")
        print(f"      want {want}")
    return ok


results = []

results.append(check(
    "shallow (0,0)->(5,2)",
    bresenham_cells(0, 0, 5, 2),
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]))

results.append(check(
    "diagonal (0,0)->(3,3)",
    bresenham_cells(0, 0, 3, 3),
    [(0, 0), (1, 1), (2, 2), (3, 3)]))

results.append(check(
    "steep (0,0)->(2,5)",
    bresenham_cells(0, 0, 2, 5),
    [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)]))

results.append(check(
    "single point",
    bresenham_cells(3, 3, 3, 3),
    [(3, 3)]))

fwd = bresenham_cells(0, 0, 5, 2)
rev = bresenham_cells(5, 2, 0, 0)
results.append(check("reversal symmetry", rev, list(reversed(fwd))))

results.append(check(
    "horizontal length n+1",
    len(bresenham_cells(0, 0, 7, 0)),
    8))

results.append(check(
    "vertical length n+1",
    len(bresenham_cells(0, 0, 0, 6)),
    7))

quadrants = [(6, 3), (-6, 3), (6, -3), (-6, -3), (3, 6), (-3, 6), (3, -6), (-3, -6)]
all_octants_ok = True
for (ex, ey) in quadrants:
    ray = bresenham_cells(0, 0, ex, ey)
    if ray[0] != (0, 0) or ray[-1] != (ex, ey):
        print(f"FAIL  octant endpoints (0,0)->({ex},{ey}): {ray[0]} .. {ray[-1]}")
        all_octants_ok = False
    if len(set(ray)) != len(ray):
        print(f"FAIL  octant repeats (0,0)->({ex},{ey})")
        all_octants_ok = False
    for (ax, ay), (bx, by) in zip(ray, ray[1:]):
        if max(abs(bx - ax), abs(by - ay)) != 1:
            print(f"FAIL  octant connectivity (0,0)->({ex},{ey}): {(ax,ay)} -> {(bx,by)}")
            all_octants_ok = False
results.append(check("all 8 octants: endpoints, no repeats, 8-connected", all_octants_ok, True))

print()
print("visual check, (0,0)->(9,4):")
print(ascii_line(bresenham_cells(0, 0, 9, 4), 10, 5))
print()
print("visual check, (0,0)->(4,9):")
print(ascii_line(bresenham_cells(0, 0, 4, 9), 5, 10))
print()
print("Section 4 checkpoint", "PASSED" if all(results) else "FAILED")