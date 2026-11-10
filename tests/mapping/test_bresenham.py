from lidar_nav.mapping.bresenham import bresenham_cells
from tests.testkit import Suite

T = Suite("Map: Bresenham ray tracing")


def same(label, got, want):
    T.check(label, got == want, "" if got == want else f"got {got}, want {want}")


T.section("known lines")
same("shallow (0,0) to (5,2)", bresenham_cells(0, 0, 5, 2), [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)])
same("diagonal (0,0) to (3,3)", bresenham_cells(0, 0, 3, 3), [(0, 0), (1, 1), (2, 2), (3, 3)])
same("steep (0,0) to (2,5)", bresenham_cells(0, 0, 2, 5), [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)])
same("single point", bresenham_cells(3, 3, 3, 3), [(3, 3)])

T.section("properties")
fwd = bresenham_cells(0, 0, 5, 2)
same("reversed line visits the same cells backwards", bresenham_cells(5, 2, 0, 0), list(reversed(fwd)))
same("horizontal line of length 7 has 8 cells", len(bresenham_cells(0, 0, 7, 0)), 8)
same("vertical line of length 6 has 7 cells", len(bresenham_cells(0, 0, 0, 6)), 7)

T.section("all eight directions")
for ex, ey in [(6, 3), (-6, 3), (6, -3), (-6, -3), (3, 6), (-3, 6), (3, -6), (-3, -6)]:
    ray = bresenham_cells(0, 0, ex, ey)
    gaps = [(a, b) for a, b in zip(ray, ray[1:]) if max(abs(b[0] - a[0]), abs(b[1] - a[1])) != 1]
    T.check(f"to ({ex:+d},{ey:+d}): right endpoints, no repeats, no gaps",
            ray[0] == (0, 0) and ray[-1] == (ex, ey) and len(set(ray)) == len(ray) and not gaps)

T.done()
