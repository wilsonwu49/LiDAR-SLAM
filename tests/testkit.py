"""Shared test reporting so every suite prints the same way and run_tests.py can read it.

    T = Suite("ICP: point-to-line solver")
    T.section("single step on a known misalignment")
    T.check("cost drops by 1000x", after < before * 1e-3, f"{before:.2e} -> {after:.2e}")
    T.near("recovers the rotation", phi, 0.03, tol=1e-6)
    T.done()
"""

import atexit
import os
import sys
import time

import numpy as np

QUIET = os.environ.get("TESTKIT_QUIET") == "1"
WIDTH = 58


def fmt(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        v = float(a)
        return f"{v:.4g}" if v == v else "nan"
    return "[" + ", ".join(f"{v:.4g}" for v in a.ravel()[:6]) + (", ..." if a.size > 6 else "") + "]"


class Suite:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = []
        self.t0 = time.perf_counter()
        self.closed = False
        atexit.register(self._at_exit)
        if not QUIET:
            print(f"\n{name}\n{'=' * len(name)}")

    def section(self, title):
        if not QUIET:
            print(f"\n  {title}")

    def info(self, *args, **_):
        if not QUIET:
            print("        " + " ".join(str(a) for a in args))

    def check(self, label, cond, detail=""):
        ok = bool(cond)
        if ok:
            self.passed += 1
        else:
            self.failed.append(label)
        if ok and QUIET:
            return ok
        line = f"    {'ok  ' if ok else 'FAIL'}  {label}"
        if detail:
            line = f"{line:<{WIDTH}}  {detail}"
        print(line)
        return ok

    def near(self, label, got, want, tol, unit=""):
        g = np.asarray(got, dtype=float)
        w = np.asarray(want, dtype=float)
        ok = g.shape == w.shape and bool(np.allclose(g, w, atol=tol))
        return self.check(label, ok, f"got {fmt(g)}{unit}  want {fmt(w)}{unit}")

    def fail(self, label, detail=""):
        return self.check(label, False, detail)

    def _summary(self):
        n = self.passed + len(self.failed)
        dt = time.perf_counter() - self.t0
        if QUIET:
            print(f"@@TESTKIT {self.passed} {len(self.failed)} {dt:.3f}", flush=True)
        else:
            verdict = "all passed" if not self.failed else f"{len(self.failed)} FAILED"
            print(f"\n  {self.passed}/{n} checks, {verdict}  ({dt:.1f} s)\n")

    def done(self):
        if self.closed:
            return
        self.closed = True
        self._summary()
        if self.failed:
            sys.exit(1)

    def _at_exit(self):
        if not self.closed:
            self.closed = True
            self._summary()
