#!/bin/bash
#
# Build VTK with Metal support for iOS.
#
# Usage:
#   ./ios_metal_build.sh              # iOS build (fresh)
#   ./ios_metal_build.sh --resume     # iOS build (resume existing)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_ios_metal"
RESUME=0

for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
  esac
done

MIN_VERSION="16.0"
PLATFORM="iPhoneOS"

if [ "$RESUME" -eq 0 ]; then
  echo "Step 1: Removing build folder..."
  rm -rf "${BUILD_DIR}"
else
  echo "Step 1: Skipping clean (--resume), reusing existing build folder..."
fi

echo "Step 2: Running CMake..."
CMAKE_CMD=(cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_SYSTEM_NAME:STRING=iOS \
    -DCMAKE_SYSTEM_PROCESSOR:STRING=arm64 \
    -DCMAKE_OSX_ARCHITECTURES:STRING=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING="${MIN_VERSION}" \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DVTK_BUILD_EXAMPLES:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DVTK_ENABLE_WRAPPING:BOOL=OFF \
    -DVTK_MODULE_ENABLE_VTK_RenderingLICOpenGL2:STRING=DONT_WANT \
    -DVTK_MODULE_ENABLE_VTK_RenderingWebGPU:STRING=DONT_WANT \
    -DVTK_MODULE_ENABLE_VTK_RenderingMetal:STRING=YES \
    -DAPPLE_IOS:BOOL=ON \
    -DHAVE_GETENTROPY:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH="${BUILD_DIR}/install")

"${CMAKE_CMD[@]}"

echo "Step 3: Building..."
cmake --build "${BUILD_DIR}" -j"$(sysctl -n hw.ncpu)"

echo "Step 4: Installing..."
cmake --install "${BUILD_DIR}"

echo "Step 5: Creating framework (shallow bundle for iOS)..."
FRAMEWORK_DIR="${BUILD_DIR}/frameworks/vtk.framework"
mkdir -p "${FRAMEWORK_DIR}/Headers"

VTK_INCLUDE_DIR=$(find "${BUILD_DIR}/install/include" -maxdepth 1 -type d -name "vtk-*" | head -1)
if [ -z "${VTK_INCLUDE_DIR}" ]; then
  echo "Error: Could not find VTK include directory in ${BUILD_DIR}/install/include/"
  exit 1
fi
cp -r "${VTK_INCLUDE_DIR}/" "${FRAMEWORK_DIR}/Headers/"

echo "Step 5a: Merging static libraries..."
LIBTOOL_INPUT=$(find "${BUILD_DIR}/install/lib" -name "*.a" | tr '\n' ' ')
libtool -static -o "${FRAMEWORK_DIR}/vtk" ${LIBTOOL_INPUT}

echo "Step 5b: Creating Info.plist..."
cat > "${FRAMEWORK_DIR}/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>vtk</string>
    <key>CFBundleIdentifier</key>
    <string>org.vtk.metal</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>VTK</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>9.6</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>${PLATFORM}</string>
    </array>
    <key>MinimumOSVersion</key>
    <string>${MIN_VERSION}</string>
</dict>
</plist>
PLIST

echo "Done! Framework created at ${FRAMEWORK_DIR}"
