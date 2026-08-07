#!/bin/bash
# Capture both backends (OpenGL vs Metal) for a set of ray-cast test variants.
#
# For each variant runs vtkRenderingVolumeCxxTests twice (RenderingBackend=OpenGL
# and RenderingBackend=Metal), writing the rendered image and stderr log under
# $OUT/<variant>/{OpenGL,Metal}.{log,png}, then reports the OpenGL GL_SAMPLING
# engagement count (Metal runs produce none).
#
# Usage:
#   ./capture_variants.sh [variant ...]
#   BUILD_DIR=/path/to/build ./capture_variants.sh
#   Env: BUILD_DIR (default <repo-root>/build_macos_metal), OUT (default /tmp/bc/caprun)
#   Positional args override the default variant set (single variants, e.g.
#   TestGPURayCastCameraInsideTransformationNoShade).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_macos_metal}"
OUT="${OUT:-/tmp/bc/caprun}"
BIN="$BUILD_DIR/bin/vtkRenderingVolumeCxxTests"
EXT="$BUILD_DIR/ExternalData/Testing"

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(TestGPURayCastCameraInsideTransformation
   TestGPURayCastCameraInsideTransformationConstGradOp
   TestGPURayCastCameraInsideTransformationNoGradOp
   TestGPURayCastCameraInsideTransformationNoShade
   TestGPURayCastCameraInsideTransformationNoShadeConstGradOp
   TestGPURayCastCameraInsideTransformationNoShadeLinGradOp
   TestGPURayCastCameraInsideTransformationNoShadeNoGradOp
   TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform
   TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside
   TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearPlaneTiny
   TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFineStep
   TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearest
   TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIP
   TestGPURayCastCameraInsideTransformationSampleDist0_5
   TestGPURayCastCameraInsideTransformationSampleDist0_25)
fi

B="$OUT/dummy_baseline.png"
for V in "${VARIANTS[@]}"; do
  mkdir -p "$OUT/$V"
  for BE in OpenGL Metal; do
    rm -rf "$OUT/tmp"; mkdir -p "$OUT/tmp"
    "$BIN" "$V" --vtk-factory-prefer "RenderingBackend=$BE" \
      -D "$EXT" -T "$OUT/tmp" -V "$B" > "$OUT/$V/${BE}.log" 2>&1 || true
    if [ -f "$OUT/tmp/dummy_baseline.png" ]; then
      cp "$OUT/tmp/dummy_baseline.png" "$OUT/$V/${BE}.png"
    else
      echo "MISSING: $V $BE"
    fi
  done
  GLENG=$(grep -c "GL_SAMPLING" "$OUT/$V/OpenGL.log" || true)
  echo "$V: GL_SAMPLING=$GLENG"
done
