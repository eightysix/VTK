#!/bin/bash
#
# Point .clangd's compilation database at the macOS or iOS build.
#
# clangd only reads one compile_commands.json, so this script symlinks the
# chosen build's compile_commands.json into .clangd-db/, which .clangd points at.
#
# Usage:
#   ./clangd_switch.sh            # use the macOS build (default)
#   ./clangd_switch.sh macos      # use build_macos_metal
#   ./clangd_switch.sh ios        # use build_ios_metal
#
# Run the corresponding *_metal_build.sh first so compile_commands.json exists.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="${SCRIPT_DIR}/.clangd-db"
TARGET="${1:-macos}"

case "$TARGET" in
  macos|ios) ;;
  *)
    echo "Usage: $0 [macos|ios]" >&2
    exit 1
    ;;
esac

BUILD_DIR="${SCRIPT_DIR}/build_${TARGET}_metal"
COMPILE_COMMANDS="${BUILD_DIR}/compile_commands.json"

if [ ! -f "${COMPILE_COMMANDS}" ]; then
  echo "Error: ${COMPILE_COMMANDS} not found." >&2
  echo "Run ./${TARGET}_metal_build.sh (or ./${TARGET}_metal_build.sh --resume) first." >&2
  exit 1
fi

mkdir -p "${DB_DIR}"
ln -sfn "${COMPILE_COMMANDS}" "${DB_DIR}/compile_commands.json"

echo "clangd now using ${COMPILE_COMMANDS}"
