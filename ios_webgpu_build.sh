#!/bin/bash
#
# Build VTK with WebGPU support for iOS or Mac Catalyst.
#
# Usage:
#   ./ios_webgpu_build.sh              # iOS build (fresh)
#   ./ios_webgpu_build.sh --resume     # iOS build (resume existing)
#   ./ios_webgpu_build.sh --catalyst   # Catalyst build (fresh)
#
# --catalyst: Build for Mac Catalyst instead of iOS.
#   Differences from iOS:
#   - Needs a Catalyst-compatible Dawn library. The macOS Dawn at
#     .gitlab/dawn/ links against Cocoa/IOKit which are unavailable
#     on Catalyst. Build Dawn from source (ci-utilities/dawn/src/)
#     with -DCMAKE_CXX_FLAGS="-target arm64-apple-ios17.0-macabi"
#     and point DAWN_INSTALL_DIR below to the install prefix.
#   - Uses CMAKE_OSX_SYSROOT=iphonesimulator and the macabi target
#     triple (arm64-apple-ios-macabi).
#   - Does NOT set APPLE_IOS or HAVE_GETENTROPY (macOS has getentropy).
#   - Info.plist uses MacOSX platform and LSMinimumSystemVersion.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_ios"
CATALYST=0
RESUME=0

for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
    --catalyst) CATALYST=1 ;;
  esac
done

if [ "$CATALYST" -eq 1 ]; then
  echo "Building for Mac Catalyst..."
  # Point to your Catalyst-compatible Dawn install. The macOS Dawn at
  # .gitlab/dawn will NOT work here (links Cocoa/IOKit). See comments above.
  DAWN_INSTALL_DIR="${SCRIPT_DIR}/.gitlab/dawn"
  BUILD_DIR="${SCRIPT_DIR}/build_catalyst"
  MIN_VERSION="17.0"
  PLATFORM="MacOSX"
  OS_KEY="LSMinimumSystemVersion"
else
  echo "Building for iOS..."
  DAWN_INSTALL_DIR="/Users/macair/Public/VTK-Source/ci-utilities/install/dawn-v20251002.162335-ios-arm64"
  BUILD_DIR="${SCRIPT_DIR}/build_ios"
  MIN_VERSION="16.0"
  PLATFORM="iPhoneOS"
  OS_KEY="MinimumOSVersion"
fi

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
    -DVTK_ENABLE_WEBGPU:BOOL=ON \
    -DDawn_DIR:PATH="${DAWN_INSTALL_DIR}/lib/cmake/Dawn" \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DVTK_BUILD_EXAMPLES:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DVTK_ENABLE_WRAPPING:BOOL=OFF \
    -DVTK_MODULE_ENABLE_VTK_RenderingLICOpenGL2:STRING=DONT_WANT \
    -DVTK_MODULE_ENABLE_VTK_RenderingWebGPU:STRING=YES \
    -DCMAKE_INSTALL_PREFIX:PATH="${BUILD_DIR}/install")

if [ "$CATALYST" -eq 1 ]; then
  # Catalyst uses the iOS Simulator SDK with the macabi target triple.
  CMAKE_CMD+=(
    -DCMAKE_OSX_SYSROOT:STRING=iphonesimulator
    -DCMAKE_C_FLAGS:STRING="-target arm64-apple-ios${MIN_VERSION}-macabi"
    -DCMAKE_CXX_FLAGS:STRING="-target arm64-apple-ios${MIN_VERSION}-macabi"
  )
else
  # iOS-specific overrides for CMake 4.x compatibility and
  # functions missing in the iOS SDK (getentropy, system()).
  CMAKE_CMD+=(
    -DAPPLE_IOS:BOOL=ON
    -DHAVE_GETENTROPY:BOOL=OFF
  )
fi

"${CMAKE_CMD[@]}"

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
        <string>${PLATFORM}</string>
    </array>
    <key>${OS_KEY}</key>
    <string>${MIN_VERSION}</string>
</dict>
</plist>
PLIST

echo "Step 5d: Creating symlinks..."
ln -sfn A "${FRAMEWORK_DIR}/Versions/Current"
ln -sfn Versions/Current/vtk "${FRAMEWORK_DIR}/vtk"
ln -sfn Versions/Current/Resources "${FRAMEWORK_DIR}/Resources"
ln -sfn Versions/Current/Headers "${FRAMEWORK_DIR}/Headers"

echo "Done! Framework created at ${FRAMEWORK_DIR}"
