#!/usr/bin/env python3
"""Parse a RenderingCoreCxx-Metal ctest run into a comprehensive report.

Run the suite once, then interpret the results from the saved artifacts:

    ctest --test-dir build_macos_metal -R "RenderingCoreCxx-Metal" -j 8 \
        > /tmp/metal_suite_run.log 2>&1
    python3 Rendering/Metal/Testing/analyze_metal_ctest_log.py

Default paths mirror that invocation (ctest console log, LastTest.log, and
TESTING_STATE.md). Override with --runlog / --lastlog / --state.

Report: pass/fail/crash counts and names, per-test TIGHT_VALID ImageError
buckets (near-miss / mid / gross), the below-threshold pick-check failures,
and a regression check against tests TESTING_STATE.md names as passing/failing.
"""

import argparse
import re
import sys
from collections import defaultdict

THRESHOLD = 0.05
BUCKETS = [("near-miss", 0.05, 0.1), ("mid", 0.1, 0.5), ("gross", 0.5, None)]


def parse_ctest_console(path):
    fails, aborts = set(), set()
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\t\d+ - VTK::RenderingCoreCxx-Metal-(\S+) \(([^)]+)\)", line)
            if m:
                (aborts if m.group(2) == "Subprocess aborted" else fails).add(m.group(1))
    return fails, aborts


def parse_lastlog(path):
    with open(path) as fh:
        log = fh.read()
    names = set(re.findall(r"Testing: VTK::RenderingCoreCxx-Metal-(\S+)", log))
    blocks = re.split(r"\n(?=\d+/\d+ Testing: )", log)
    errors = {}
    for b in blocks:
        m = re.match(r"\d+/\d+ Testing: VTK::RenderingCoreCxx-Metal-(\S+)", b)
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
    ap.add_argument("--runlog", default="/tmp/metal_suite_run.log",
                    help="ctest console log (list of failed tests)")
    ap.add_argument("--lastlog",
                    default="build_macos_metal/Testing/Temporary/LastTest.log",
                    help="LastTest.log from the same run")
    ap.add_argument("--state",
                    default="Rendering/Metal/Testing/TESTING_STATE.md",
                    help="TESTING_STATE.md used for the regression check")
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

    if args.state:
        try:
            doc = open(args.state).read()
        except FileNotFoundError:
            doc = None
        if doc is not None:
            passing_named = {"TestAreaSelections", "TestHardwareSelector",
                             "TestAxesActor", "TestReadPixels",
                             "TestSelectVisiblePoints", "TestWorldPointPicker",
                             "TestRemoveActors",
                             "TestCompositePolyDataMapperBlockTextures",
                             "TestCompositePolyDataMapperOverrideLUT",
                             "TestCompositePolyDataMapperOverrideScalarArray",
                             "TestCompositePolyDataMapperNaNPartial",
                             "TestTextureInterpolateScalars",
                             "TestTranslucentLUTTextureAlphaBlending",
                             "TestTranslucentLUTTextureDepthPeeling",
                             "TestOpacityMSAA", "TestActor2D",
                             "TestTexturedBackground", "TestStereoBackgroundLeft",
                             "TestStereoBackgroundRight",
                             "TestGlyph3DMapperIndexing", "TestGlyph3DMapperTreeIndexing",
                             "TestWireframe", "TestSurfacePlusEdges",
                             "TestTexturedCylinder", "TestEdgeThickness",
                             "TestNActorsOneMapper", "TestNActorsNMappersOneInput",
                             "TestNViewportsOneActor", "TestNViewportsNActorsOneMapper",
                             "TestNViewportsNActorsNMappersOneInput",
                             "TestNViewportsNActorsNMappersNInputs",
                             "TestImageAndAnnotations", "TestActor2DTextures",
                             "TestBackfaceCulling",
                             "TestLineRenderingTranslucent",
                             "TestGlyph3DMapperCompositeDisplayAttributeInheritance",
                             "TestMixedGeometryCellScalars",
                             "TestTransformCoordinateUseDouble",
                             "TestResetCameraScreenSpace",
                             "TestPolyDataMapperClipPlanes",
                             "TestRenderLinesAsTubes",
                             "TestRenderLinesAsTubesOrthoCamera"}
            regressed = sorted((fails | aborts) & passing_named)
            print("\nREGRESSION CHECK vs TESTING_STATE.md passing cluster:")
            print("  " + (", ".join(regressed) if regressed else "none"))


if __name__ == "__main__":
    main()
