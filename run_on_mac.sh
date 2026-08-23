#!/bin/bash

set -e

SCHEME="test-vtk-metal-mac"
PROJECT_DIR="Examples/GUI/iOSMetal"

show_usage() {
    echo "Usage: $0 [debug|release]"
    echo "  debug   - Build Debug configuration (default)"
    echo "  release - Build Release configuration"
    exit 1
}

CONFIG="Debug"
if [[ "$1" == "release" ]]; then
    CONFIG="Release"
elif [[ "$1" == "debug" ]]; then
    CONFIG="Debug"
elif [[ -n "$1" ]]; then
    echo "Unknown option: $1"
    show_usage
fi

echo "Building $SCHEME ($CONFIG) for Mac..."

xcodebuild -project "$PROJECT_DIR/test-vtk-metal.xcodeproj" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -destination 'platform=macOS' \
    build

DERIVED_DATA_PATH=$(find ~/Library/Developer/Xcode/DerivedData/test-vtk-metal-* -maxdepth 0 -type d 2>/dev/null | head -1)
if [[ -z "$DERIVED_DATA_PATH" ]]; then
    echo "Error: Could not find DerivedData for test-vtk-metal"
    exit 1
fi

MAC_APP="$DERIVED_DATA_PATH/Build/Products/${CONFIG}/${SCHEME}.app"

if [[ ! -d "$MAC_APP" ]]; then
    echo "Error: Built app not found at $MAC_APP"
    exit 1
fi

echo "Launching app (Ctrl+C to quit)..."
exec "$MAC_APP/Contents/MacOS/$SCHEME"
