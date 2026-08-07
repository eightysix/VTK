#!/bin/bash
# Regenerate metal3.log (per-sample march dump) from a Metal test run.
#
# Prerequisites:
#   - Build with tests enabled so the volume shaders compile with shader
#     logging: ./macos_metal_build.sh --resume --tests
#     (VTK_BUILD_TESTING=ON -> VTK_METAL_ENABLE_LOGGING is defined and
#     MTLCompileOptions.enableLogging is set; see
#     Rendering/Metal/Testing/Cxx/TestMetalVolumeShaderLog.cxx).
#   - The per-sample call sites (MARCH/SAMPLE/LIGHT/LIGHT2) are gated to a few
#     pixels by debugMarchGate in Rendering/Metal/Shaders/MetalShaders.metal
#     (includes the (256,256) camera-inside pixel used by the comparison docs).
#   - Shader os_log is buffered per command buffer and forwarded to stderr via
#     MTL_LOG_TO_STDERR=1; MTL_LOG_BUFFER_SIZE must be large enough to hold a
#     full frame (~0.5 MB for the 469-sample pixel gate), else messages are
#     silently dropped.
#
# Usage: ./make_metal3_log.sh [build_dir]
#   build_dir defaults to <repo-root>/build_macos_metal (run from anywhere).
# Output: $OUT (default /tmp/bc/metal3.log, override via OUT env var)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/build_macos_metal}"
WORK="${WORK:-/tmp/bc}"
OUT="${OUT:-$WORK/metal3.log}"

# dummy baseline so the run exercises the full (fail-and-dump) path
B="${B:-$WORK/TestGPURayCastCameraInsideTransformation.png}"
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$B')"

MTL_LOG_LEVEL=MTLLogLevelDebug \
MTL_LOG_BUFFER_SIZE=16777216 \
MTL_LOG_TO_STDERR=1 \
  "$BUILD_DIR/bin/vtkRenderingVolumeCxxTests" TestGPURayCastCameraInsideTransformation \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$BUILD_DIR/ExternalData/Testing" -T "$BUILD_DIR/Testing/Temporary" \
    -V "$B" \
    2> "$OUT"

wc -l "$OUT"
