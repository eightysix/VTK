#!/bin/bash

set -e

SCHEME="test-vtk-metal"
PROJECT_NAME="test-vtk-metal"
PROJECT_DIR="Examples/GUI/iOSMetal"
OUTPUT_DIR="build"

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

echo "Building $SCHEME ($CONFIG) for Mac (Designed for iPad)..."

xcodebuild -project "$PROJECT_DIR/$PROJECT_NAME.xcodeproj" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -destination 'generic/platform=iOS' \
    build

DERIVED_DATA_PATH=$(find ~/Library/Developer/Xcode/DerivedData/${PROJECT_NAME}-* -maxdepth 0 -type d 2>/dev/null | head -1)
if [[ -z "$DERIVED_DATA_PATH" ]]; then
    echo "Error: Could not find DerivedData for $PROJECT_NAME"
    exit 1
fi

IOS_APP="$DERIVED_DATA_PATH/Build/Products/${CONFIG}-iphoneos/${SCHEME}.app"

if [[ ! -d "$IOS_APP" ]]; then
    echo "Error: Built app not found at $IOS_APP"
    exit 1
fi

echo "Creating wrapper structure in project folder..."
rm -rf "$OUTPUT_DIR/${SCHEME}.app"
mkdir -p "$OUTPUT_DIR/${SCHEME}.app/Wrapper"
cp -R "$IOS_APP" "$OUTPUT_DIR/${SCHEME}.app/Wrapper/"
ln -sf "Wrapper/${SCHEME}.app" "$OUTPUT_DIR/${SCHEME}.app/WrappedBundle"

echo "Wrapper created at: $PWD/$OUTPUT_DIR/${SCHEME}.app"
echo "Launching app..."
open "$OUTPUT_DIR/${SCHEME}.app"

echo "Done!"
