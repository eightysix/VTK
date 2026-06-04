#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DAWN_INSTALL_DIR="${SCRIPT_DIR}/.gitlab/dawn"

echo "Step 1: Removing build folder..."
rm -rf "${BUILD_DIR}"

echo "Step 2: Running CMake..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_OSX_ARCHITECTURES:STRING=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=14.0 \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DVTK_ENABLE_WEBGPU:BOOL=ON \
    -DDawn_DIR:PATH="${DAWN_INSTALL_DIR}/lib/cmake/Dawn" \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DVTK_BUILD_EXAMPLES:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DVTK_MODULE_ENABLE_VTK_RenderingWebGPU:STRING=YES \
    -DCMAKE_INSTALL_PREFIX:PATH="${BUILD_DIR}/install"

echo "Step 3: Building..."
cmake --build "${BUILD_DIR}" -j"$(sysctl -n hw.ncpu)"

echo "Step 4: Installing..."
cmake --install "${BUILD_DIR}"

echo "Step 5: Creating framework..."
FRAMEWORK_DIR="${BUILD_DIR}/frameworks/vtk.framework"
mkdir -p "${FRAMEWORK_DIR}/Headers"

VTK_INCLUDE_DIR=$(find "${BUILD_DIR}/install/include" -maxdepth 1 -type d -name "vtk-*" | head -1)
if [ -z "${VTK_INCLUDE_DIR}" ]; then
  echo "Error: Could not find VTK include directory in ${BUILD_DIR}/install/include/"
  exit 1
fi
cp -r "${VTK_INCLUDE_DIR}/" "${FRAMEWORK_DIR}/Headers/"

echo "Step 5a: Copying Dawn headers..."
cp -r "${DAWN_INSTALL_DIR}/include/webgpu" "${FRAMEWORK_DIR}/Headers/"
cp -r "${DAWN_INSTALL_DIR}/include/dawn" "${FRAMEWORK_DIR}/Headers/"

echo "Step 5b: Merging static libraries..."
LIBTOOL_INPUT=$(find "${BUILD_DIR}/install/lib" -name "*.a" | tr '\n' ' ')
libtool -static -o "${FRAMEWORK_DIR}/vtk" ${LIBTOOL_INPUT} "${DAWN_INSTALL_DIR}/lib/libwebgpu_dawn.a"

echo "Step 5c: Copying Info.plist..."
cat > "${FRAMEWORK_DIR}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>vtk</string>
    <key>CFBundleIdentifier</key>
    <string>org.vtk.webgpu</string>
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
        <string>MacOSX</string>
    </array>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
</dict>
</plist>
PLIST

echo "Done! Framework created at ${FRAMEWORK_DIR}"
