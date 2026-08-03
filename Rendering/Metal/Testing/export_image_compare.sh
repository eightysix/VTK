#!/bin/bash
#
# Re-runs the generic Metal image-comparison suite and exports the failing
# renders (Metal + baseline + diff PNGs) plus a per-test metric manifest into a
# review folder, so failures can be inspected side by side.
#
# "Image-compare" failures are tests that rendered and produced a
# vtkRegressionTestImage comparison (reported as
# "Failed Image Test ( <Test>.png ) : <metric>" in the ctest log, threshold
# 0.05 on the TIGHT_VALID metric). Tests that crash or fail a non-image check
# (pick/selection) render no output and are only listed in the summary.
#
# Usage:
#   ./export_image_compare.sh              # run suite, export + manifest
#   ./export_image_compare.sh --no-run     # export from the last run's artifacts
#   ./export_image_compare.sh --out /tmp/compare --jobs 4
#
# Options:
#   -b, --build-dir DIR   Metal build tree (default: <repo>/build_macos_metal)
#   -o, --out DIR         review output dir (default: <build>/Testing/ImageCompareReview)
#   -j, --jobs N          ctest parallelism (default: 8)
#   -n, --no-run          skip the ctest run; export from the last run's artifacts
#   -h, --help            show this help
#
# Requires: a GUI login session (tests render to the WindowServer), a --tests
# Metal build, ctest, and python3 (stdlib only; no external packages).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build_macos_metal"
OUT_DIR=""
JOBS=8
RUN=1

usage() {
  grep '^#' "$0" | sed '1d;s/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    -b|--build-dir) BUILD_DIR="$2"; shift 2 ;;
    -o|--out) OUT_DIR="$2"; shift 2 ;;
    -j|--jobs) JOBS="$2"; shift 2 ;;
    -n|--no-run) RUN=0; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ ! -d "$BUILD_DIR" ]; then
  echo "build dir not found: $BUILD_DIR" >&2
  exit 1
fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
OUT_DIR="${OUT_DIR:-${BUILD_DIR}/Testing/ImageCompareReview}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

TEST_PREFIX="RenderingCoreCxx-Metal"
LOG="${BUILD_DIR}/Testing/Temporary/LastTest.log"
FAILED_LIST="${BUILD_DIR}/Testing/Temporary/LastTestsFailed.log"

if [ "$RUN" -eq 1 ]; then
  echo "== ctest -R ${TEST_PREFIX} -j ${JOBS}"
  ctest --test-dir "$BUILD_DIR" -R "$TEST_PREFIX" -j "$JOBS" || true
else
  echo "== --no-run: exporting from the last run's artifacts"
fi

echo "== exporting image-compare failures to ${OUT_DIR}"
python3 - "$BUILD_DIR" "$OUT_DIR" "$LOG" "$FAILED_LIST" "$TEST_PREFIX" <<'PYEOF'
import os
import re
import shutil
import sys

build_dir, out_dir, log, failed_list, test_prefix = sys.argv[1:6]

fail_img = re.compile(r'Failed Image Test \( ([A-Za-z0-9_]+\.png) \) : ([\d.eE+-]+)')
test_line = re.compile(r'^[0-9]+/[0-9]+ Test: (\S+)')
measure = re.compile(r'<DartMeasurementFile name="(\w+)"[^>]*>([^<]+)</DartMeasurementFile>')
name_map = {
    'TestImage': 'metal',
    'ValidImage': 'baseline',
    'DifferenceImage': 'diff',
}
failures = {}
current = None
if os.path.exists(log):
    with open(log, errors='replace') as f:
        for line in f:
            t = test_line.match(line)
            if t:
                full = t.group(1)
                current = full[full.rfind('-') + 1:]
                failures.setdefault(current, {})
                continue
            m = fail_img.search(line)
            if m and current:
                failures[current]['metric'] = float(m.group(2))
                continue
            for mm in measure.finditer(line):
                key = name_map.get(mm.group(1))
                if key and current:
                    failures[current][key] = mm.group(2)
failures = {n: f for n, f in failures.items() if 'metric' in f}

if not failures and os.path.exists(log):
    if os.path.getsize(log) == 0:
        print("WARNING: log is empty: %s" % log)
    else:
        with open(log, errors='replace') as f:
            found = any(test_prefix in line for line in f)
        if not found:
            print("WARNING: %s contains no '%s' tests; run the suite first" % (log, test_prefix))

temp_dir = os.path.join(build_dir, 'Testing', 'Temporary')
base_dir = os.path.join(build_dir, 'ExternalData', 'Rendering', 'Core', 'Testing', 'Data', 'Baseline')
for name, f in failures.items():
    f.setdefault('metal', os.path.join(temp_dir, name + '.png'))
    f.setdefault('diff', os.path.join(temp_dir, name + '.diff.png'))
    f.setdefault('baseline', os.path.join(base_dir, name + '.png'))

failed_tests = []
if os.path.exists(failed_list):
    with open(failed_list, errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                failed_tests.append(line.split(':', 1)[-1].replace('VTK::RenderingCoreCxx-Metal-', ''))

for old in os.listdir(out_dir):
    os.remove(os.path.join(out_dir, old))
missing = []
for name, f in sorted(failures.items()):
    for key, suffix in (('metal', '.metal.png'), ('baseline', '.baseline.png'), ('diff', '.diff.png')):
        src = f.get(key)
        if src and os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, name + suffix))
        else:
            missing.append((name, key))

rows = sorted(failures.items(), key=lambda kv: kv[1].get('metric', float('inf')))
with open(os.path.join(out_dir, 'manifest.txt'), 'w') as fh:
    fh.write("%-60s %12s\n" % ('Test', 'ImageError'))
    fh.write('-' * 74 + '\n')
    for name, f in rows:
        m = f.get('metric')
        fh.write("%-60s %12s\n" % (name, 'n/a' if m is None else '%.6f' % m))
    fh.write('-' * 74 + '\n')
    fh.write("image-compare failures: %d\n" % len(rows))

def bucket(m):
    if m is None:
        return None
    if m < 0.05:
        return 'below-threshold'
    if m < 0.1:
        return 'near-miss (0.05-0.1)'
    if m < 0.5:
        return 'mid (0.1-0.5)'
    return 'gross (>=0.5)'

from collections import Counter
counts = Counter(bucket(f.get('metric')) for f in failures.values())

non_image = [t for t in failed_tests if t not in failures]

print()
print("summary")
print("  total failed tests: %d" % len(failed_tests))
print("  image-compare failures exported: %d" % len(failures))
for b in ('near-miss (0.05-0.1)', 'mid (0.1-0.5)', 'gross (>=0.5)', 'below-threshold'):
    if counts[b]:
        print("    %-22s %d" % (b, counts[b]))
if missing:
    print("  WARNING: missing source images (not copied):")
    for name, key in missing:
        print("    %s (%s)" % (name, key))
if non_image:
    print("  non-image failures (crash / pick-check, no render exported): %d" % len(non_image))
    for t in sorted(non_image):
        print("    %s" % t)
print()
print("review at: %s" % out_dir)
PYEOF
