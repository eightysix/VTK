#!/bin/bash
# Build the probe7b GPU benchmark harness.
# Usage: ./build_probe.sh
# Output: ./probe7b (next to this script)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clang -fobjc-arc -framework Metal -framework Foundation "$DIR/probe7b.m" -o "$DIR/probe7b"
echo "built $DIR/probe7b"
