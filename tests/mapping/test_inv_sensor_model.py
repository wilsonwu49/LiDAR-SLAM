from lidar_nav.mapping.inv_sensor_model import classify_beam
from tests.testkit import Suite

T = Suite("Map: beam classification")

Z_MAX, R_MIN, CLEAR = 12.0, 0.2, 8.0
cases = [
    ("normal hit", 3.2, ("valid_hit", 3.2)),
    ("hit just inside max range", 11.99, ("valid_hit", 11.99)),
    ("hit exactly at minimum range", 0.2, ("valid_hit", 0.2)),
    ("max range means nothing was hit", 12.0, ("no_return", 8.0)),
    ("beyond max range", 15.0, ("no_return", 8.0)),
    ("zero range is discarded", 0.0, None),
    ("below minimum range is discarded", 0.1, None),
    ("negative range is discarded", -1.0, None),
    ("nan is discarded", float("nan"), None),
    ("inf is discarded", float("inf"), None),
    ("-inf is discarded", float("-inf"), None),
]
for label, z, want in cases:
    got = classify_beam(z, Z_MAX, R_MIN, CLEAR)
    T.check(label, got == want, "" if got == want else f"got {got}, want {want}")

T.done()
