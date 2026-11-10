"""Final fixed-vs-adaptive comparison on the two-pinch route, with confidence intervals.

    python -m experiments.final_run            # 30 seeds per config
    python -m experiments.final_run --smoke    # 3 seeds, to check it runs
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import numpy as np

from experiments.two_pinch import trial

CONFIGS = [("fixed", 0.25, 0.0), ("fixed", 0.30, 0.0), ("fixed", 0.35, 0.0),
           ("adaptive", 0.25, 0.5), ("adaptive", 0.25, 1.0)]
REFERENCE = (0.35, 0.0)


def run_one(job):
    kind, R, k, seed = job
    t = trial(R, k, seed, max_cycles=2500)
    t["kind"] = kind
    return t


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def rate_ci(x):
    x = np.asarray(x, float)
    p = x.mean()
    return p, 1.96 * np.sqrt(p * (1 - p) / len(x))


def label(R, k):
    return f"fixed R={R:.2f}" if k == 0 else f"adaptive k={k:.1f}"


def main():
    n_seeds = 3 if "--smoke" in sys.argv else 30
    jobs = [(kind, R, k, s) for kind, R, k in CONFIGS for s in range(1, n_seeds + 1)]
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            print(f"  {i}/{len(jobs)}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    print()
    print(f"{'config':>16} {'late clearance (mm)':>24} {'early clearance (mm)':>24} {'success':>16}")
    for _, R, k in CONFIGS:
        s = [r for r in rows if r["R"] == R and r["k"] == k]
        ml, hl, _ = mean_ci([r["clear_late"] * 1000 for r in s])
        me, he, _ = mean_ci([r["clear_early"] * 1000 for r in s])
        p, hp = rate_ci([r["reached"] for r in s])
        print(f"{label(R, k):>16} {ml:14.1f} ± {hl:5.1f} {me:16.1f} ± {he:5.1f} "
              f"{p * 100:9.0f}% ± {hp * 100:3.0f}")

    print()
    print(f"paired difference in late clearance vs {label(*REFERENCE)}, same seed:")
    ref = {r["seed"]: r for r in rows if (r["R"], r["k"]) == REFERENCE}
    for _, R, k in CONFIGS:
        if (R, k) == REFERENCE:
            continue
        d = [(r["clear_late"] - ref[r["seed"]]["clear_late"]) * 1000
             for r in rows if r["R"] == R and r["k"] == k and r["seed"] in ref]
        m, h, n = mean_ci(d)
        verdict = "no detectable difference" if abs(m) < h else ("better" if m > 0 else "worse")
        print(f"  {label(R, k):>16}: {m:+7.1f} ± {h:5.1f} mm  (n={n})  {verdict}")

    keys = ("R", "k", "seed", "clear_early", "clear_late", "margin_early",
            "margin_late", "reached", "cycles")
    os.makedirs("experiments/results", exist_ok=True)
    np.savez("experiments/results/final_results.npz",
             kind=np.array([r["kind"] for r in rows]),
             **{key: np.array([r.get(key, np.nan) for r in rows], float) for key in keys})
    print("\nwrote experiments/results/final_results.npz")


if __name__ == "__main__":
    main()
