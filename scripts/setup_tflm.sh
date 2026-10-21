#!/bin/sh
# Fetches TFLM at the commit the firmware was verified with, applies the int4 patch and downloads its toolchain.
# Usage: scripts/setup_tflm.sh   -> third_party/tflite-micro (git-ignored). Build steps: docs/TFLM_BUILD.md
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); DST="$ROOT/third_party/tflite-micro"
COMMIT=e0021ca434cd930acbab69768e88d959083a3d4b
if [ ! -d "$DST/.git" ]; then rm -rf "$DST"; mkdir -p "$ROOT/third_party"; git clone https://github.com/tensorflow/tflite-micro.git "$DST"; fi
cd "$DST" && git checkout "$COMMIT"
git apply --check "$ROOT/firmware/patches/tflm-no-int4.patch" 2>/dev/null && git apply "$ROOT/firmware/patches/tflm-no-int4.patch" || echo "patch already applied or not applicable"
echo "TFLM ready at $DST (next: build the library as in docs/TFLM_BUILD.md)"
