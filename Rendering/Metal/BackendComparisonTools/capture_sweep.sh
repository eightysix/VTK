#!/bin/bash
# Fixed-sample-distance sweep for ...CamOutsideFixedStep, both backends.
#
# Runs TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep
# with VTK_FIXED_SAMPLE_DISTANCE over a step-size ladder (auto-adjust off) and
# captures each backend's render + log under $OUT/sweep/, reporting the PROBE
# fixedSD/autoAdjust line and OpenGL GL_SAMPLING engagement count.
#
# Usage:
#   ./capture_sweep.sh [step ...]
#   BUILD_DIR=/path/to/build ./capture_sweep.sh
#   Env: BUILD_DIR (default <repo-root>/build_macos_metal), OUT (default /tmp/bc/caprun)
#   Positional args override the default step ladder.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_macos_metal}"
OUT="${OUT:-/tmp/bc/caprun}"
BIN="$BUILD_DIR/bin/vtkRenderingVolumeCxxTests"
EXT="$BUILD_DIR/ExternalData/Testing"

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(0.0675 0.135 0.27 0.5 1.0 2.0 4.0)
fi

B="$OUT/dummy_baseline.png"
mkdir -p "$OUT/sweep"
for sd in "${STEPS[@]}"; do
  for BE in OpenGL Metal; do
    rm -rf "$OUT/tmp"; mkdir -p "$OUT/tmp"
    VTK_FIXED_SAMPLE_DISTANCE=$sd "$BIN" \
      TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep \
      --vtk-factory-prefer "RenderingBackend=$BE" \
      -D "$EXT" -T "$OUT/tmp" -V "$B" > "$OUT/sweep/sd${sd}_${BE}.log" 2>&1 || true
    cp "$OUT/tmp/dummy_baseline.png" "$OUT/sweep/sd${sd}_${BE}.png" 2>/dev/null || true
    PROBE=$(grep -o "PROBE fixedSD=[0-9.]* autoAdjust=[0-9]*" "$OUT/sweep/sd${sd}_${BE}.log" | head -1)
    GLENG=$(grep -c "GL_SAMPLING" "$OUT/sweep/sd${sd}_${BE}.log" || true)
    echo "sd$sd $BE: $PROBE GL_SAMPLING=$GLENG"
  done
done
