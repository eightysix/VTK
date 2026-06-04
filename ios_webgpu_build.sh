#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_ios"
DAWN_INSTALL_DIR="/Users/macair/Public/VTK-Source/ci-utilities/install/dawn-v20251002.162335-ios-arm64"

if [ "$1" != "--resume" ]; then
  echo "Step 1: Removing build folder..."
  rm -rf "${BUILD_DIR}"
else
  echo "Step 1: Skipping clean (--resume), reusing existing build folder..."
fi

echo "Step 2: Running CMake..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_SYSTEM_NAME:STRING=iOS \
    -DCMAKE_SYSTEM_PROCESSOR:STRING=arm64 \
    -DCMAKE_OSX_ARCHITECTURES:STRING=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=16.0 \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DAPPLE_IOS:BOOL=ON \
    -DVTK_ENABLE_WEBGPU:BOOL=ON \
    -DDawn_DIR:PATH="${DAWN_INSTALL_DIR}/lib/cmake/Dawn" \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DVTK_BUILD_EXAMPLES:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DVTK_ENABLE_WRAPPING:BOOL=OFF \
    -DHAVE_GETENTROPY:BOOL=OFF \
    -DVTK_MODULE_ENABLE_VTK_RenderingWebGPU:STRING=YES \
    -DCMAKE_INSTALL_PREFIX:PATH="${BUILD_DIR}/install"

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

echo "Step 5a: Copying Dawn headers..."
cp -r "${DAWN_INSTALL_DIR}/include/webgpu" "${VERSION_DIR}/Headers/"
cp -r "${DAWN_INSTALL_DIR}/include/dawn" "${VERSION_DIR}/Headers/"

echo "Step 5b: Merging static libraries..."
LIBTOOL_INPUT=$(find "${BUILD_DIR}/install/lib" -name "*.a" | tr '\n' ' ')
libtool -static -o "${VERSION_DIR}/vtk" ${LIBTOOL_INPUT} "${DAWN_INSTALL_DIR}/lib/libwebgpu_dawn.a"

echo "Step 5c: Creating Info.plist..."
cat > "${VERSION_DIR}/Resources/Info.plist" <<'PLIST'
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
        <string>iPhoneOS</string>
    </array>
    <key>MinimumOSVersion</key>
    <string>15.0</string>
</dict>
</plist>
PLIST

echo "Step 5d: Creating symlinks..."
ln -sfn A "${FRAMEWORK_DIR}/Versions/Current"
ln -sfn Versions/Current/vtk "${FRAMEWORK_DIR}/vtk"
ln -sfn Versions/Current/Resources "${FRAMEWORK_DIR}/Resources"
ln -sfn Versions/Current/Headers "${FRAMEWORK_DIR}/Headers"

echo "Done! Framework created at ${FRAMEWORK_DIR}"
