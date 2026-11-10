"""Run every test suite and print one summary table.

Run from the project folder:

    python tests/run_tests.py              all fast suites
    python tests/run_tests.py --all        include the full-navigation simulations (slow)
    python tests/run_tests.py icp map      only suites whose name contains "icp" or "map"
    python tests/run_tests.py -v           show every suite's full output

One suite on its own:  python -m tests.planning.test_astar
"""

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the project folder
STAGES = [("estimation", "estimation"), ("scan_matching", "scan matching"), ("mapping", "mapping"),
          ("planning", "planning"), ("control", "control"), ("full_system", "full system")]
SLOW = {"tests.full_system.test_navigate"}


def discover():
    found = []
    for pkg, stage in STAGES:
        for f in sorted((ROOT / "tests" / pkg).glob("test_*.py")):
            found.append((stage, f"tests.{pkg}.{f.stem}"))
    return found


def title_of(module):
    text = (ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
    m = re.search(r'Suite\("([^"]+)"\)', text)
    return m.group(1) if m else module


def run(module, verbose):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if not verbose:
        env["TESTKIT_QUIET"] = "1"
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", module], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    m = re.search(r"@@TESTKIT (\d+) (\d+) [\d.]+", p.stdout)
    passed, failed = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    crashed = p.returncode != 0 and failed == 0
    return dict(module=module, passed=passed, failed=failed, crashed=crashed, seconds=dt,
                out=p.stdout.replace(m.group(0), "").rstrip() if m else p.stdout.rstrip(),
                err=p.stderr.rstrip())


def main():
    args = sys.argv[1:]
    verbose = "-v" in args
    include_slow = "--all" in args
    filters = [a.lower() for a in args if not a.startswith("-")]

    suites = [(s, m) for s, m in discover()
              if (include_slow or m not in SLOW)
              and (not filters or any(f in m.lower() or f in s or f in title_of(m).lower() for f in filters))]
    if not suites:
        print("no suites match")
        return 1

    t0 = time.perf_counter()
    workers = 1 if verbose else max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda sm: run(sm[1], verbose), suites))

    if verbose:
        for r in results:
            print(r["out"])
            if r["err"]:
                print(r["err"])

    print()
    print(f"  {'stage':<14} {'suite':<52} {'checks':>8} {'time':>7}")
    print(f"  {'-' * 14} {'-' * 52} {'-' * 8} {'-' * 7}")
    last_stage = None
    for (stage, module), r in zip(suites, results):
        n = r["passed"] + r["failed"]
        status = "CRASHED" if r["crashed"] else ("FAIL" if r["failed"] else "ok")
        checks = "-" if r["crashed"] else f"{r['passed']}/{n}"
        shown = stage if stage != last_stage else ""
        last_stage = stage
        print(f"  {shown:<14} {title_of(module)[:52]:<52} {checks:>8} {r['seconds']:6.1f}s  {status}")

    total = sum(r["passed"] + r["failed"] for r in results)
    bad = [r for r in results if r["failed"] or r["crashed"]]
    print()
    print(f"  {len(results)} suites, {total} checks, "
          f"{'all passed' if not bad else f'{len(bad)} suite(s) with problems'}"
          f"  ({time.perf_counter() - t0:.1f} s)")
    if not include_slow and not filters:
        print("  (full-navigation simulations skipped; add --all to include them)")

    for r in bad:
        print(f"\n  ---- {r['module']} ----")
        if r["out"].strip():
            print(r["out"])
        if r["crashed"] and r["err"]:
            print("\n".join(r["err"].splitlines()[-12:]))
    print()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
