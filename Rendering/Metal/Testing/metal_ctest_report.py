#!/usr/bin/env python3
"""Run a Metal ctest image-comparison suite, analyze it, and export the failures.

Unified replacement for the former analyze_metal_ctest_log.py (report) and
export_image_compare.sh (ctest run + failing-render export).

Supports both the Rendering/Core suite (RenderingCoreCxx-Metal, the default)
and the Rendering/Volume suite (RenderingVolumeCxx-Metal); pick one with
--prefix.

Typical use:

    python3 Rendering/Metal/Testing/metal_ctest_report.py                # run core suite, report + export
    python3 Rendering/Metal/Testing/metal_ctest_report.py -p RenderingVolumeCxx-Metal
    python3 Rendering/Metal/Testing/metal_ctest_report.py --no-run        # analyze/export last run only

The script re-runs the ctest suite (unless --no-run), prints a pass/fail/crash
report with per-test TIGHT_VALID ImageError buckets (near-miss / mid / gross /
below-threshold), a regression check against a list of names expected to pass
(--passing, or a built-in list for the core suite), then exports the failing
renders (Metal + baseline + diff PNGs) plus a per-test metric manifest into a
review folder (default <build>/Testing/ImageCompareReview, or
...ImageCompareReviewVolume for the volume suite).

Important: this script runs the already-built test binaries in <build>; it
does not compile anything. After changing Metal rendering source (or tests),
rebuild the tree with the tests-enabled build first (per AGENTS.md):

    ./macos_metal_build.sh --resume --tests

otherwise the suite would exercise stale binaries.

"Image-compare" failures are tests that rendered and produced a
vtkRegressionTestImage comparison (reported as
"Failed Image Test ( <Test>.png ) : <metric>" in the ctest log, threshold
0.05 on the TIGHT_VALID metric). Tests that crash or fail a non-image check
(pick/selection) render no output and are only listed in the summary.

Options:
  -b, --build-dir DIR   Metal build tree (default: <repo>/build_macos_metal)
  -o, --out DIR         review output dir (default: <build>/Testing/ImageCompareReview
                        or .../ImageCompareReviewVolume for the volume suite)
  -p, --prefix PREFIX   ctest test prefix (default: RenderingCoreCxx-Metal;
                        use RenderingVolumeCxx-Metal for the volume suite)
  -d, --baseline-dir DIR
                        baseline PNG dir (default derived from --prefix:
                        <build>/ExternalData/Rendering/Core/... or
                        <build>/ExternalData/Rendering/Volume/...)
  -j, --jobs N          ctest parallelism (default: 8)
  -n, --no-run          skip the ctest run; analyze/export last run's artifacts
  -r, --runlog FILE     ctest console log (default: /tmp/metal_suite_run.log
                        or /tmp/metal_volume_suite_run.log for the volume suite)
  -l, --lastlog FILE    LastTest.log (default: <build>/Testing/Temporary/LastTest.log)
  --passing FILE        optional file of test names expected to pass (one per
                        line) for the regression check; the core suite falls
                        back to a built-in list
  -h, --help            show this help

Requires: a GUI login session (tests render to the WindowServer), a --tests
Metal build, ctest, and python3 (stdlib only; no external packages).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

THRESHOLD = 0.05
BUCKETS = [("near-miss", 0.05, 0.1), ("mid", 0.1, 0.5), ("gross", 0.5, None)]
BUCKET_LABEL = {
    "below-threshold": "below-threshold",
    "near-miss": "near-miss (0.05-0.1)",
    "mid": "mid (0.1-0.5)",
    "gross": "gross (>=0.5)",
}
DEFAULT_PREFIX = "RenderingCoreCxx-Metal"

NAME_TO_KEY = {
    "TestImage": "metal",
    "ValidImage": "baseline",
    "DifferenceImage": "diff",
}

CORE_PASSING = {
    "TestAreaSelections", "TestHardwareSelector",
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
    "TestRenderLinesAsTubesOrthoCamera",
}


def suite_runlog(prefix):
    return "/tmp/metal_volume_suite_run.log" if "Volume" in prefix \
        else "/tmp/metal_suite_run.log"


def parse_ctest_console(path, prefix):
    """Failed/aborted test names from a ctest console log.

    Handles both the modern per-test lines ("Test #123: VTK::<prefix>-<name> ...
    ***Failed" / "... Subprocess aborted***") and the legacy summary section
    ("\t<num> - VTK::<prefix>-<name> (Failed)").
    """
    esc = re.escape(prefix)
    inline = re.compile(r"Test #\d+: VTK::" + esc + r"-(\S+)")
    legacy = re.compile(r"\t\d+ - VTK::" + esc + r"-(\S+) \(([^)]+)\)")
    fails, aborts = set(), set()
    with open(path, errors="replace") as fh:
        for line in fh:
            m = legacy.search(line)
            if m:
                (aborts if m.group(2) == "Subprocess aborted" else fails).add(m.group(1))
                continue
            m = inline.search(line)
            if m:
                name = m.group(1)
                if "Subprocess aborted" in line:
                    aborts.add(name)
                elif "Failed" in line:
                    fails.add(name)
    return fails, aborts


def parse_lastlog(path, prefix):
    """All test names, image-compare failure metrics, and comparison image paths.

    An image-compare failure is identified by the
    "Failed Image Test ( <Test>.png ) : <metric>" line vtkTesting emits only
    when an image comparison fails (the ImageError DartMeasurement is also
    printed for passing comparisons and non-image failures, so it is not used
    as the discriminator). Returns (names, imgfails, imgs) where imgfails maps
    test name -> TIGHT_VALID metric and imgs maps test name -> {metal, baseline,
    diff} paths when present in the log.
    """
    fail_img = re.compile(r"Failed Image Test \( ([A-Za-z0-9_]+\.png) \) : "
                          r"([\d.eE+-]+)")
    with open(path, errors="replace") as fh:
        log = fh.read()
    esc = re.escape(prefix)
    names = set(re.findall(r"Testing: VTK::" + esc + r"-(\S+)", log))
    blocks = re.split(r"\n(?=\d+/\d+ Testing: )", log)
    imgfails = {}
    imgs = {}
    for b in blocks:
        m = re.match(r"\d+/\d+ Testing: VTK::" + esc + r"-(\S+)", b)
        if not m:
            continue
        name = m.group(1)
        err = fail_img.search(b)
        if err:
            imgfails[name] = float(err.group(2))
        im = {}
        for dm in re.finditer(
                r'<DartMeasurementFile name="(\w+)"[^>]*>([^<]+)</DartMeasurementFile>', b):
            key = NAME_TO_KEY.get(dm.group(1))
            if key:
                im[key] = dm.group(2)
        if im:
            imgs[name] = im
    return names, imgfails, imgs


def parse_failed_list(path, prefix):
    """Test names from ctest's LastTestsFailed.log (all failures incl. aborts)."""
    out = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(line.split(":", 1)[-1].replace("VTK::" + prefix + "-", ""))
    return out


def bucket(err):
    if err < THRESHOLD:
        return "below-threshold"
    for label, lo, hi in BUCKETS:
        if err >= lo and (hi is None or err < hi):
            return label
    return "gross"


def run_ctest(build_dir, prefix, jobs, runlog):
    with open(runlog, "w") as fh:
        proc = subprocess.Popen(
            ["ctest", "--test-dir", build_dir, "-R", prefix, "-j", str(jobs)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout or []:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=("Run a Metal ctest image-comparison suite, analyze it, "
                     "and export the failures."))
    ap.add_argument("-p", "--prefix", default=DEFAULT_PREFIX,
                    help="ctest test prefix (default: %(default)s; "
                         "use RenderingVolumeCxx-Metal for the volume suite)")
    ap.add_argument("-b", "--build-dir", default=None,
                    help="Metal build tree (default: <repo>/build_macos_metal)")
    ap.add_argument("-o", "--out", default=None,
                    help="review output dir (default: "
                         "<build>/Testing/ImageCompareReview[Volume])")
    ap.add_argument("-d", "--baseline-dir", default=None,
                    help="baseline PNG dir (default derived from --prefix)")
    ap.add_argument("-j", "--jobs", type=int, default=8,
                    help="ctest parallelism (default: %(default)s)")
    ap.add_argument("-n", "--no-run", action="store_true",
                    help="skip the ctest run; analyze/export last run's artifacts")
    ap.add_argument("-r", "--runlog", default=None,
                    help="ctest console log (default: /tmp/metal_suite_run.log "
                         "or /tmp/metal_volume_suite_run.log for the volume "
                         "suite)")
    ap.add_argument("-l", "--lastlog", default=None,
                    help="LastTest.log (default: "
                         "<build>/Testing/Temporary/LastTest.log)")
    ap.add_argument("--passing", default=None,
                    help="optional file of test names expected to pass "
                         "(one per line) for the regression check; the core "
                         "suite falls back to a built-in list")
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    build_dir = os.path.abspath(args.build_dir or
                                os.path.join(repo_root, "build_macos_metal"))
    if not os.path.isdir(build_dir):
        sys.exit(f"build dir not found: {build_dir}")

    volume = "Volume" in args.prefix
    review = "ImageCompareReviewVolume" if volume else "ImageCompareReview"
    out_dir = os.path.abspath(args.out or
                              os.path.join(build_dir, "Testing", review))
    os.makedirs(out_dir, exist_ok=True)
    baseline_rel = ("ExternalData/Rendering/Volume/Testing/Data/Baseline"
                    if volume else
                    "ExternalData/Rendering/Core/Testing/Data/Baseline")
    baseline_dir = os.path.abspath(args.baseline_dir or
                                   os.path.join(build_dir, baseline_rel))

    runlog = os.path.abspath(args.runlog or suite_runlog(args.prefix))
    lastlog = os.path.abspath(args.lastlog or os.path.join(
        build_dir, "Testing", "Temporary", "LastTest.log"))
    failed_list = os.path.join(build_dir, "Testing", "Temporary",
                               "LastTestsFailed.log")

    if not args.no_run:
        print(f"== ctest --test-dir {build_dir} -R {args.prefix} -j {args.jobs}")
        run_ctest(build_dir, args.prefix, args.jobs, runlog)
    else:
        print("== --no-run: analyzing/exporting the last run's artifacts")

    fails, aborts = set(), set()
    runlog_had = False
    if os.path.isfile(runlog):
        fails, aborts = parse_ctest_console(runlog, args.prefix)
        runlog_had = bool(fails or aborts)
    elif args.no_run:
        print(f"WARNING: --runlog not found: {runlog}")

    if not os.path.isfile(lastlog):
        sys.exit(f"missing --lastlog: {lastlog}")
    names, imgfails, imgs = parse_lastlog(lastlog, args.prefix)
    if not names:
        sys.exit(f"no '{args.prefix}' tests found in {lastlog}; "
                 "run the suite first")

    failed_names = set()
    if os.path.isfile(failed_list):
        failed_names = parse_failed_list(failed_list, args.prefix)
    if failed_names:
        aborts &= failed_names
        if not runlog_had:
            aborts = set()
        fails = failed_names - aborts
    else:
        fails |= aborts
        aborts = set()
        if fails:
            print("NOTE: LastTestsFailed.log not found; using the ctest log "
                  "(crash/abort split unavailable)")
    failed_names = (failed_names | fails | aborts) & names
    non_image = sorted(failed_names - set(imgfails))

    passes = names - fails - aborts
    print(f"PASS {len(passes)}  FAIL {len(fails)}  ABORT {len(aborts)}  "
          f"(total {len(names)})")

    print("\nABORTS:")
    for n in sorted(aborts):
        print(f"  {n}")

    over = sorted((n for n in imgfails if imgfails[n] >= THRESHOLD),
                  key=lambda n: imgfails[n])
    below = sorted((n for n in imgfails if imgfails[n] < THRESHOLD),
                   key=lambda n: imgfails[n])
    bybucket = defaultdict(list)
    for n in over:
        bybucket[bucket(imgfails[n])].append(n)

    print(f"\nIMAGE-COMPARE FAILS (TIGHT_VALID >= {THRESHOLD}): {len(over)}")
    for label, _, _ in BUCKETS:
        lst = bybucket[label]
        print(f"  {label} [{len(lst)}]: " +
              ", ".join(f"{n} {imgfails[n]:.4f}" for n in lst) or "none")
    print(f"  below-threshold [{len(below)}]: " +
          ", ".join(f"{n} {imgfails[n]:.4f}" for n in below) or "none")
    print(f"NON-IMAGE FAILS [no Failed Image Test] ({len(non_image)}): "
          + ", ".join(non_image))

    passing_named = None
    source = None
    if args.passing:
        try:
            passing_named = {l.strip() for l in open(args.passing) if l.strip()}
            source = args.passing
        except FileNotFoundError:
            passing_named = set()
            source = args.passing
    elif not volume:
        passing_named = CORE_PASSING
        source = "built-in list"
    if passing_named is not None:
        regressed = sorted((fails | aborts) & passing_named)
        print("\nREGRESSION CHECK vs passing list (%s):" % source)
        print("  " + (", ".join(regressed) if regressed else "none"))

    exportable = sorted(set(imgfails) & failed_names) if failed_names \
        else sorted(imgfails)
    temp_dir = os.path.join(build_dir, "Testing", "Temporary")
    meta = {}
    for name in exportable:
        im = imgs.get(name, {})
        meta[name] = {
            "metric": imgfails[name],
            "metal": im.get("metal", os.path.join(temp_dir, name + ".png")),
            "diff": im.get("diff", os.path.join(temp_dir, name + ".diff.png")),
            "baseline": im.get("baseline", os.path.join(baseline_dir, name + ".png")),
        }

    for old in os.listdir(out_dir):
        p = os.path.join(out_dir, old)
        if os.path.isfile(p):
            os.remove(p)
    missing = []
    for name, f in sorted(meta.items()):
        for key, suffix in (("metal", ".metal.png"),
                            ("baseline", ".baseline.png"),
                            ("diff", ".diff.png")):
            src = f.get(key)
            if src and os.path.isfile(src):
                shutil.copyfile(src, os.path.join(out_dir, name + suffix))
            else:
                missing.append((name, key))

    rows = sorted(meta.items(), key=lambda kv: kv[1]["metric"])
    with open(os.path.join(out_dir, "manifest.txt"), "w") as fh:
        fh.write("%-60s %12s\n" % ("Test", "ImageError"))
        fh.write("-" * 74 + "\n")
        for name, f in rows:
            fh.write("%-60s %12.6f\n" % (name, f["metric"]))
        fh.write("-" * 74 + "\n")
        fh.write("image-compare failures: %d\n" % len(rows))

    counts = Counter(bucket(f["metric"]) for f in meta.values())

    print("\nexport")
    print("  total failed tests: %d" % len(failed_names))
    print("  image-compare failures exported: %d" % len(exportable))
    for short in ("near-miss", "mid", "gross", "below-threshold"):
        if counts[short]:
            print("    %-22s %d" % (BUCKET_LABEL[short], counts[short]))
    if missing:
        print("  WARNING: missing source images (not copied):")
        for name, key in missing:
            print("    %s (%s)" % (name, key))
    if non_image:
        print("  non-image failures (crash / pick-check, no render exported): "
              "%d" % len(non_image))
        for t in non_image:
            print("    %s" % t)
    print("\nreview at: %s" % out_dir)


if __name__ == "__main__":
    main()
