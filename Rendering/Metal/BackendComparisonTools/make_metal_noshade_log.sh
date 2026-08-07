#!/bin/bash
# Regenerate metal_noshade.log (per-sample SAMPLE + GRADOP dump) from a Metal
# NoShade test run.
#
# Same prerequisites and os_log plumbing as make_metal3_log.sh, but runs the
# NoShade variant so the gradient-opacity block (GRADOP lines) fires instead of
# the shading block (LIGHT lines). See
# Rendering/Metal/VolumeRayCastBackendComparisonFindings.md section 5.
#
# Usage: ./make_metal_noshade_log.sh [build_dir]
#   build_dir defaults to <repo-root>/build_macos_metal (run from anywhere).
# Output: $OUT (default /tmp/bc/metal_noshade.log, override via OUT env var)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/build_macos_metal}"
WORK="${WORK:-/tmp/bc}"
OUT="${OUT:-$WORK/metal_noshade.log}"

B="${B:-$WORK/TestGPURayCastCameraInsideTransformationNoShade.png}"
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$B')"

MTL_LOG_LEVEL=MTLLogLevelDebug \
MTL_LOG_BUFFER_SIZE=16777216 \
MTL_LOG_TO_STDERR=1 \
  "$BUILD_DIR/bin/vtkRenderingVolumeCxxTests" TestGPURayCastCameraInsideTransformationNoShade \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$BUILD_DIR/ExternalData/Testing" -T "$BUILD_DIR/Testing/Temporary" \
    -V "$B" \
    2> "$OUT"

wc -l "$OUT"
