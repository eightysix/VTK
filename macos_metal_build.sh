#!/bin/bash
#
# Build VTK with Metal support for macOS.
#
# Usage:
#   ./macos_metal_build.sh              # macOS build (fresh)
#   ./macos_metal_build.sh --resume     # macOS build (resume existing)
#   ./macos_metal_build.sh --tests      # also enable the module test suites
#   ./macos_metal_build.sh --no-tests   # disable test suites (overrides --tests)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_macos_metal"
RESUME=0
TESTS=0

for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
    --tests) TESTS=1 ;;
    --no-tests) TESTS=0 ;;
  esac
done

MIN_VERSION="14.0"
PLATFORM="MacOSX"

if [ "$RESUME" -eq 0 ]; then
  echo "Step 1: Removing build folder..."
  rm -rf "${BUILD_DIR}"
else
  echo "Step 1: Skipping clean (--resume), reusing existing build folder..."
fi

echo "Step 2: Running CMake..."
CMAKE_CMD=(cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_OSX_ARCHITECTURES:STRING=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING="${MIN_VERSION}" \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DVTK_BUILD_EXAMPLES:BOOL=OFF \
    -DVTK_BUILD_TESTING:STRING=$(if [ "$TESTS" -eq 1 ]; then echo ON; else echo OFF; fi) \
    -DBUILD_TESTING:BOOL=$(if [ "$TESTS" -eq 1 ]; then echo ON; else echo OFF; fi) \
    -DVTK_ENABLE_WRAPPING:BOOL=OFF \
    -DVTK_MODULE_ENABLE_VTK_RenderingLICOpenGL2:STRING=DONT_WANT \
    -DVTK_MODULE_ENABLE_VTK_RenderingWebGPU:STRING=DONT_WANT \
    -DVTK_MODULE_ENABLE_VTK_RenderingMetal:STRING=YES \
    -DVTK_MODULE_ENABLE_VTK_vtkDICOM:STRING=YES \
    -DCMAKE_INSTALL_PREFIX:PATH="${BUILD_DIR}/install" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON)

"${CMAKE_CMD[@]}"

echo "Step 2a: Pointing clangd at this build..."
"${SCRIPT_DIR}/clangd_switch.sh" macos

echo "Step 3: Building..."
cmake --build "${BUILD_DIR}" -j"$(sysctl -n hw.ncpu)"

echo "Step 4: Installing..."
cmake --install "${BUILD_DIR}"

echo "Step 5: Creating framework (versioned bundle)..."
FRAMEWORK_DIR="${BUILD_DIR}/frameworks/vtk.framework"
VERSION_DIR="${FRAMEWORK_DIR}/Versions/A"
mkdir -p "${VERSION_DIR}/Resources"
mkdir -p "${VERSION_DIR}/Headers"

VTK_INCLUDE_DIR=$(find "${BUILD_DIR}/install/include" -maxdepth 1 -type d -name "vtk-*" | head -1)
if [ -z "${VTK_INCLUDE_DIR}" ]; then
  echo "Error: Could not find VTK include directory in ${BUILD_DIR}/install/include/"
  exit 1
fi
cp -r "${VTK_INCLUDE_DIR}/" "${VERSION_DIR}/Headers/"

echo "Step 5a: Merging static libraries..."
LIBTOOL_INPUT=$(find "${BUILD_DIR}/install/lib" -name "*.a" | tr '\n' ' ')
libtool -static -o "${VERSION_DIR}/vtk" ${LIBTOOL_INPUT}

echo "Step 5b: Creating Info.plist..."
cat > "${VERSION_DIR}/Resources/Info.plist" <<PLIST
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
    <key>LSMinimumSystemVersion</key>
    <string>${MIN_VERSION}</string>
</dict>
</plist>
PLIST

echo "Step 5c: Creating symlinks..."
ln -sfn A "${FRAMEWORK_DIR}/Versions/Current"
ln -sfn Versions/Current/vtk "${FRAMEWORK_DIR}/vtk"
ln -sfn Versions/Current/Resources "${FRAMEWORK_DIR}/Resources"
ln -sfn Versions/Current/Headers "${FRAMEWORK_DIR}/Headers"

echo "Done! Framework created at ${FRAMEWORK_DIR}"
