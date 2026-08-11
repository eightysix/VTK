#!/bin/bash
# =============================================================================
# run_pixel_diff_suite.sh
#
# Metal vs OpenGL volume-raycast pixel-diff reproduction suite.
#
# Runs the contained camera-inside volume test family on BOTH backends and
# diffs the two backends' rendered frames pixel-by-pixel, reporting the same
# metrics the findings docs use:
#
#     diff px   = pixels with any-channel |delta| >= 1  (any difference)
#     |d|>=2    = pixels with max-channel |delta| >= 2
#     |d|>=5    = pixels with max-channel |delta| >= 5  (the "big" deltas)
#     max d     = max over pixels/channels of |delta|
#
# -----------------------------------------------------------------------------
# WHY A CHECKERBOARD DUMMY BASELINE (NOT A BLACK ONE)
# -----------------------------------------------------------------------------
# Every test here ends in vtkRegressionTestImage (via vtkTesting::Test /
# InteractorEventLoop). To make a test "fail", the -V baseline image must be
# present and must NOT match the render. A SOLID BLACK dummy is NOT reliable:
# VTK's image compare has a pass threshold, and a near-uniform (dark) render
# reports ImageError ~ 0 against a flat black image, so the test silently
# PASSES, no image is dumped, and the harness path is the wrong one.
#
#   VolumeRayCastBackendComparisonFindingsUpdate65.md §4.5:
#     "VTK image-compare pass threshold gotcha: a near-uniform render passes
#      against a flat/black dummy baseline (ImageError 0) ... All captures in
#      this update use a 16-px red/green checkerboard dummy so any render fails
#      and dumps."
#
# A 16-px red/green checkerboard differs from every real render by a huge
# ImageError, so the regression ALWAYS fails and VTK always writes the dump.
# -----------------------------------------------------------------------------
# WHY THE DUMMY MUST BE REGENERATED PER INVOCATION
# -----------------------------------------------------------------------------
# On regression failure vtkTesting writes the actual render to
# `$TMPDIR/<basename of -V>` (Testing/Rendering/vtkTesting.cxx:904,
# testImageFileName = tmpDir + "/" + validName). It does NOT write to the -V
# path, so a stale dummy at -V is fine, but a stale DUMP at -T is a trap:
# always use a fresh per-run tmp dir (this script does).
# -----------------------------------------------------------------------------
# FRAME ALIGNMENT (why the comparison is valid)
# -----------------------------------------------------------------------------
# vtkTestingInteractor::Start() runs the W2IF regression BEFORE the test's
# InteractorEventLoop returns, so the last frame on the front buffer is the
# W2IF-perturbed camera copy (30.0000008 deg, the CPU-side float32 view-angle
# round trip of vtkWindowToImageFilter, documented in update 19). Both backends
# receive the SAME perturbed camera, so the raw-capture diff is a same-camera,
# frame-aligned, deterministic backend diff. Verified in update 79:
#   - Metal raw == Metal w2if   0 px
#   - GL raw     == GL w2if     0 px
#   - Metal raw vs GL raw  == harness (178 px, max d 8, for this build)
# -----------------------------------------------------------------------------
# PREREQUISITES
# -----------------------------------------------------------------------------
#   - Build with tests enabled:
#         ./macos_metal_build.sh --resume --tests
#     (produces build_macos_metal/bin/vtkRenderingVolumeCxxTests)
#   - python3 with PIL and numpy (used for the dummy + diff).
#   - MUST be run with BASH, not zsh. zsh does not word-split unquoted
#     variables, so `env $envstr` would pass the whole string as one argument
#     and every matrix run would silently render the DEFAULT config (the
#     update-67 incident: "zsh word-splitting of $envs silently ran a garbage
#     matrix"). This script is #!/bin/bash and refuses to run under zsh.
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   ./run_pixel_diff_suite.sh                     # full suite, 1 run/backend
#   ./run_pixel_diff_suite.sh --quick             # reference + NoJitter + FlatTF
#   RUNS=2 ./run_pixel_diff_suite.sh              # 2 runs/backend + md5 check
#   ONLY=NoJitter ./run_pixel_diff_suite.sh       # single test (substring match)
#   BUILD_DIR=/path ./run_pixel_diff_suite.sh     # custom build
#   WORK=/tmp/out ./run_pixel_diff_suite.sh       # custom output dir
#
# Exit codes:
#   0 = all runs produced captures and diffs (metrics are informational)
#   1 = usage error / missing binary / missing capture / diff input mismatch
#   These tests are EXPECTED to "fail" (return nonzero) against the dummy
#   baseline -- that is the point. Nonzero from the test binary is normal.
# =============================================================================

set -u

if [ -n "${ZSH_VERSION:-}" ]; then
  echo "FATAL: run with bash, not zsh (zsh does not word-split env strings; see header)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_macos_metal}"
BIN="$BUILD_DIR/bin/vtkRenderingVolumeCxxTests"
EXT="$BUILD_DIR/ExternalData/Testing"
WORK="${WORK:-/tmp/bc/pixdiff}"
DUMMY="$WORK/dummy.png"
RUNS="${RUNS:-1}"
ONLY="${ONLY:-}"

# Per-test tmp dirs keep the fail-dump (which lands at $TMPDIR/<dummy name>)
# from being clobbered by the next run (vtkTesting.cxx:904).
TMPROOT="$WORK/tmp"
SUMMARY="$WORK/summary.txt"

mkdir -p "$WORK" "$TMPROOT"
: > "$SUMMARY"

if [ ! -x "$BIN" ]; then
  echo "FATAL: test binary not found: $BIN" >&2
  echo "       Build with tests first: ./macos_metal_build.sh --resume --tests" >&2
  exit 1
fi

echo "== config =="
echo "  binary     $BIN"
echo "  data root  $EXT"
echo "  work dir   $WORK"
echo "  runs/backend $RUNS"

# -----------------------------------------------------------------------------
# 16-px red/green checkerboard dummy (512x512). Any real render differs from
# this by a huge ImageError, so the regression always fails and always dumps.
# -----------------------------------------------------------------------------
python3 - "$DUMMY" <<'PY'
import sys
from PIL import Image
n, c = 512, 16
img = Image.new('RGB', (n, n))
px = img.load()
for y in range(n):
    for x in range(n):
        px[x, y] = (255, 0, 0) if ((x // c) + (y // c)) % 2 == 0 else (0, 255, 0)
img.save(sys.argv[1])
PY

# -----------------------------------------------------------------------------
# Test family.
# -----------------------------------------------------------------------------
# RAW_CAPTURE tests: VTK_STEP_RAW_CAPTURE=<file> reads the front buffer after
# the harness finishes, so the capture is the W2IF-perturbed frame, frame
# aligned on both backends (update 79). These are the three tests with the hook.
BASE_RAW=(
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform"          # reference (jitter on)
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter"  # jitter disabled
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF"    # step-TF matrix below
)

# No raw-capture hook: compared via the -V/-T fail-dump (the dumped image IS the
# W2IF render, equivalent to raw on each backend, update 79 §1).
BASE_DUMP=(
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideNoJitter"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFineStep"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFlatTF"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformLinear"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIP"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearest"
  "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearPlaneTiny"
)

# StepTF matrix (name|env). VTK_STEP_WHEEL=1 replicates the reference log's
# single MouseWheelForwardEvent (Dolly(1.21) + ResetCameraClippingRange);
# required for the volume to be in frame at all (update 67 §1).
# m3a/b/c = mode-3 (constant color, opacity linear ramp) ramp-endpoint ladder.
STEPTF_CFG=(
  "m0|VTK_STEP_MODE=0 VTK_STEP_WHEEL=1"
  "m1|VTK_STEP_MODE=1 VTK_STEP_WHEEL=1"
  "m2|VTK_STEP_MODE=2 VTK_STEP_WHEEL=1"
  "m3a|VTK_STEP_MODE=3 VTK_STEP_RAMP_MAX=0.005 VTK_STEP_WHEEL=1"
  "m3b|VTK_STEP_MODE=3 VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1"
  "m3c|VTK_STEP_MODE=3 VTK_STEP_RAMP_MAX=0.1 VTK_STEP_WHEEL=1"
  "m4|VTK_STEP_MODE=4 VTK_STEP_WHEEL=1"
  "m5|VTK_STEP_MODE=5 VTK_STEP_WHEEL=1"
  "B|VTK_STEP_MODE=3 VTK_STEP_CONSTANT=2000 VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1"
  "D256|VTK_STEP_MODE=3 VTK_STEP_DIMS=256 VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1"
  "D64|VTK_STEP_MODE=3 VTK_STEP_DIMS=0 VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1"
  "E|VTK_STEP_MODE=3 VTK_CAMERA_AXIS=z VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1"
  "m0lin|VTK_STEP_MODE=0 VTK_STEP_LINEAR=1 VTK_STEP_WHEEL=1"
  "m2lin|VTK_STEP_MODE=2 VTK_STEP_LINEAR=1 VTK_STEP_WHEEL=1"
)

# -----------------------------------------------------------------------------
# run_backend <test> <envstr> <backend> <capture-out>
#   Runs one test on one backend. The capture-out is where the backend's frame
#   must end up afterwards: either the VTK_STEP_RAW_CAPTURE path or the
#   fail-dump path ($TMPDIR/dummy.png). The test binary is expected to return
#   nonzero (regression fail against the checkerboard) -- that is normal.
# -----------------------------------------------------------------------------
run_backend() {
  local test="$1" envstr="$2" be="$3" out="$4"
  local tmp="$TMPROOT/${test}_${be}"
  mkdir -p "$tmp"
  local envs=()
  if [ -n "$envstr" ]; then
    # shellcheck disable=SC2206 # intentional word-split of the env list
    envs=($envstr)
  fi

  # Raw-capture tests read the front buffer AFTER the harness, so the capture
  # lands at the exact file we give it (W2IF-perturbed frame, frame aligned).
  # Non-hooked tests dump to $tmp/dummy.png (basename of -V) on regression fail.
  if [[ " ${BASE_RAW[*]} " == *" $test "* ]]; then
    env "${envs[@]+"${envs[@]}"}" VTK_STEP_RAW_CAPTURE="$out" \
      "$BIN" "$test" \
      --vtk-factory-prefer "RenderingBackend=$be" \
      -D "$EXT" -T "$tmp" -V "$DUMMY" > "$tmp/run.log" 2>&1 || true
  else
    env "${envs[@]+"${envs[@]}"}" \
      "$BIN" "$test" \
      --vtk-factory-prefer "RenderingBackend=$be" \
      -D "$EXT" -T "$tmp" -V "$DUMMY" > "$tmp/run.log" 2>&1 || true
    cp "$tmp/$(basename "$DUMMY")" "$out" 2>/dev/null || true
  fi

  if [ ! -s "$out" ]; then
    echo "MISSING capture: $test $be -> $out (see $tmp/run.log)" >&2
    return 1
  fi
  return 0
}

# -----------------------------------------------------------------------------
# diff_images <gl> <mt> <label>
#   Emits one metric line and appends it to the summary.
# -----------------------------------------------------------------------------
diff_images() {
  python3 - "$1" "$2" "$3" "$SUMMARY" <<'PY'
import sys
from PIL import Image
import numpy as np
gl = np.array(Image.open(sys.argv[1]).convert('RGB')).astype(int)
mt = np.array(Image.open(sys.argv[2]).convert('RGB')).astype(int)
label = sys.argv[3]
if gl.shape != mt.shape:
    print(f'ERROR: size mismatch {gl.shape} vs {mt.shape}: {label}', file=sys.stderr)
    sys.exit(2)
d = mt - gl
md = np.abs(d).max(axis=2)
line = (f'{label}: diff={int((md>=1).sum()):7d} '
        f'|d|>=2={int((md>=2).sum()):6d} |d|>=5={int((md>=5).sum()):5d} '
        f'max_d={int(md.max())}')
print(line)
with open(sys.argv[4], 'a') as f:
    f.write(line + '\n')
PY
}

# -----------------------------------------------------------------------------
# Suite driver.
# -----------------------------------------------------------------------------
fail=0
run_case() {
  local test="$1" envstr="$2" tag="$3"
  local gl="$WORK/gl_${tag}.png" mt="$WORK/mt_${tag}.png"
  local gl1="$WORK/gl_${tag}_r1.png" mt1="$WORK/mt_${tag}_r1.png"
  local r
  for r in $(seq 1 "$RUNS"); do
    if ! run_backend "$test" "$envstr" OpenGL "$gl"; then fail=1; return; fi
    if ! run_backend "$test" "$envstr" Metal  "$mt"; then fail=1; return; fi
    if [ "$r" -eq 1 ]; then cp "$gl" "$gl1"; cp "$mt" "$mt1"; fi
  done
  # Determinism check (optional, RUNS=2): each backend must be byte-identical
  # across runs, else the diff is not backend-to-backend (update 69 §2 protocol).
  if [ "$RUNS" -gt 1 ]; then
    if ! cmp -s "$gl" "$gl1" || ! cmp -s "$mt" "$mt1"; then
      echo "NON-DETERMINISTIC: $tag (rerun mismatch)" >&2
      fail=1
    fi
  fi
  diff_images "$gl" "$mt" "$tag"
}

# --quick: reference + NoJitter + FlatTF only (the fast acceptance gate).
if [ "${1:-}" = "--quick" ]; then
  run_case "${BASE_RAW[0]}" "" "Reference"
  run_case "${BASE_RAW[1]}" "" "NoJitter"
  run_case "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFlatTF" "" "FlatTF"
  echo "--- summary ($WORK/summary.txt) ---"
  cat "$SUMMARY"
  exit "$fail"
fi

tag_for() {
  local t="$1" tag="${t#TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform}"
  [ -z "$tag" ] && tag="Reference"
  echo "$tag"
}

for t in "${BASE_RAW[@]}"; do
  tag="$(tag_for "$t")"
  if [ -n "$ONLY" ] && [[ "$tag" != *"$ONLY"* ]]; then continue; fi
  run_case "$t" "" "$tag"
done

for t in "${BASE_DUMP[@]}"; do
  tag="$(tag_for "$t")"
  if [ -n "$ONLY" ] && [[ "$tag" != *"$ONLY"* ]]; then continue; fi
  run_case "$t" "" "$tag"
done

for cfg in "${STEPTF_CFG[@]}"; do
  name="${cfg%%|*}"
  envstr="${cfg#*|}"
  t="${BASE_RAW[2]}" # StepTF test
  if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && [[ "StepTF" != *"$ONLY"* ]] &&
     [[ "$t" != *"$ONLY"* ]]; then
    continue
  fi
  run_case "$t" "$envstr" "StepTF_$name"
done

echo "--- summary ($WORK/summary.txt) ---"
cat "$SUMMARY"
exit "$fail"
