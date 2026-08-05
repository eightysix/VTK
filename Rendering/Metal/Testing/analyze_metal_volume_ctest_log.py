#!/usr/bin/env python3
"""Parse a RenderingVolumeCxx-Metal ctest run into a comprehensive report.

Volume suite mirror of analyze_metal_ctest_log.py: same report structure, but
scoped to the generic Rendering/Volume multi-backend suite run against Metal.

Run the suite once, then interpret the results from the saved artifacts:

    ctest --test-dir build_macos_metal -R "RenderingVolumeCxx-Metal" -j 8 \
        > /tmp/metal_volume_suite_run.log 2>&1
    python3 Rendering/Metal/Testing/analyze_metal_volume_ctest_log.py

Default paths mirror that invocation (ctest console log, LastTest.log, and
TESTING_STATE.md). Override with --runlog / --lastlog / --state.

Report: pass/fail/crash counts and names, per-test TIGHT_VALID ImageError
buckets (near-miss / mid / gross), the below-threshold pick-check failures,
and a regression check against a file of names expected to pass
(--passing, one test name per line; optional).
"""

import argparse
import re
import sys
from collections import defaultdict

THRESHOLD = 0.05
BUCKETS = [("near-miss", 0.05, 0.1), ("mid", 0.1, 0.5), ("gross", 0.5, None)]
TEST_PREFIX = "RenderingVolumeCxx-Metal"


def parse_ctest_console(path):
    fails, aborts = set(), set()
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\t\d+ - VTK::" + TEST_PREFIX + r"-(\S+) \(([^)]+)\)", line)
            if m:
                (aborts if m.group(2) == "Subprocess aborted" else fails).add(m.group(1))
    return fails, aborts


def parse_lastlog(path):
    with open(path) as fh:
        log = fh.read()
    names = set(re.findall(r"Testing: VTK::" + TEST_PREFIX + r"-(\S+)", log))
    blocks = re.split(r"\n(?=\d+/\d+ Testing: )", log)
    errors = {}
    for b in blocks:
        m = re.match(r"\d+/\d+ Testing: VTK::" + TEST_PREFIX + r"-(\S+)", b)
        if not m:
            continue
        err = re.search(r"ImageError[^>]*>([\d.eE+-]+)", b)
        if err:
            errors[m.group(1)] = float(err.group(1))
    return names, errors


def bucket(name, err):
    for label, lo, hi in BUCKETS:
        if err >= lo and (hi is None or err < hi):
            return label
    return "below-threshold"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runlog", default="/tmp/metal_volume_suite_run.log",
                    help="ctest console log (list of failed tests)")
    ap.add_argument("--lastlog",
                    default="build_macos_metal/Testing/Temporary/LastTest.log",
                    help="LastTest.log from the same run")
    ap.add_argument("--passing", default=None,
                    help="optional file of test names expected to pass "
                         "(one per line) for the regression check")
    args = ap.parse_args(argv)

    try:
        fails, aborts = parse_ctest_console(args.runlog)
    except FileNotFoundError:
        sys.exit(f"missing --runlog: {args.runlog}")
    try:
        names, errors = parse_lastlog(args.lastlog)
    except FileNotFoundError:
        sys.exit(f"missing --lastlog: {args.lastlog}")

    passes = names - fails - aborts
    print(f"PASS {len(passes)}  FAIL {len(fails)}  ABORT {len(aborts)}  "
          f"(total {len(names)})")

    print("\nABORTS:")
    for n in sorted(aborts):
        print(f"  {n}")

    over = sorted((n for n in fails if n in errors and errors[n] >= THRESHOLD),
                  key=lambda n: errors[n])
    bybucket = defaultdict(list)
    for n in over:
        bybucket[bucket(n, errors[n])].append(n)
    below = sorted(n for n in fails if n in errors and errors[n] < THRESHOLD)
    noerr = sorted(fails - set(errors) - aborts)

    print(f"\nIMAGE-COMPARE FAILS (TIGHT_VALID >= {THRESHOLD}): {len(over)}")
    for label, _, _ in BUCKETS:
        lst = bybucket[label]
        print(f"  {label} [{len(lst)}]: " +
              ", ".join(f"{n} {errors[n]:.4f}" for n in lst) or "none")
    print(f"  below-threshold [{len(below)}]: " +
          ", ".join(f"{n} {errors[n]:.4f}" for n in below) or "none")
    print(f"NON-IMAGE FAILS [no ImageError metric] ({len(noerr)}): "
          + ", ".join(noerr))

    if args.passing:
        try:
            passing_named = {l.strip() for l in open(args.passing) if l.strip()}
        except FileNotFoundError:
            passing_named = set()
        regressed = sorted((fails | aborts) & passing_named)
        print("\nREGRESSION CHECK vs passing list (%s):" % args.passing)
        print("  " + (", ".join(regressed) if regressed else "none"))


if __name__ == "__main__":
    main()
