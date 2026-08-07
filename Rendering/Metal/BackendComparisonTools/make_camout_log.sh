#!/bin/bash
# Regenerate a per-sample MARCH/SAMPLE log for the camera-outside fixed-step
# scene on Metal (NoShadeNoGradOpNoTransformCamOutsideFixedStep).
#
# Same prerequisites and os_log plumbing as make_metal3_log.sh; the scene is the
# camera-outside fixed-step isolate used by
# VolumeRayCastBackendComparisonFindingsUpdate.md (step swept via
# VTK_FIXED_SAMPLE_DISTANCE; debugMarchGate pxOkCamOut pins the ring/border px).
#
# Usage: ./make_camout_log.sh [build_dir] [step] [out.log]
#   build_dir defaults to <repo-root>/build_macos_metal (run from anywhere).
#   step defaults to 0.0675; out defaults to /tmp/bc/camout_sd<step>.log
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/build_macos_metal}"
STEP="${2:-0.0675}"
OUT="${3:-/tmp/bc/camout_sd${STEP}.log}"

B="${B:-/tmp/bc/TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep.png}"
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$B')"

VTK_FIXED_SAMPLE_DISTANCE="$STEP" \
MTL_LOG_LEVEL=MTLLogLevelDebug \
MTL_LOG_BUFFER_SIZE=134217728 \
MTL_LOG_TO_STDERR=1 \
  "$BUILD_DIR/bin/vtkRenderingVolumeCxxTests" \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$BUILD_DIR/ExternalData/Testing" -T "$BUILD_DIR/Testing/Temporary" \
    -V "$B" \
    2> "$OUT"

wc -l "$OUT"
