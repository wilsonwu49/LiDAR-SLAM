from occupancy.inv_sensor_model import classify_beam

Z_MAX, R_MIN, CLEAR = 12.0, 0.2, 8.0

cases = [
    (3.2,             ("valid_hit", 3.2)),
    (11.99,           ("valid_hit", 11.99)),
    (0.2,             ("valid_hit", 0.2)),
    (12.0,            ("no_return", 8.0)),
    (15.0,            ("no_return", 8.0)),
    (0.0,             None),
    (0.1,             None),
    (-1.0,            None),
    (float("nan"),    None),
    (float("inf"),    None),
    (float("-inf"),   None),
]

ok = True
for z, want in cases:
    got = classify_beam(z, Z_MAX, R_MIN, CLEAR)
    if got != want:
        print(f"FAIL  z={z}: got {got}, want {want}")
        ok = False
print("classify_beam", "PASSED" if ok else "FAILED")