#!/bin/sh
# Builds the two TFLM libraries the firmware uses, with the flags it was measured with (see docs/VERIFICATION.md):
#   release            -> firmware/app/build_app.sh (error strings stripped, kernels at -Os)
#   release_with_logs  -> firmware/bird/build_bird_qemu.sh (QEMU tests)
# Usage: scripts/build_tflm.sh [TFLM_ROOT]   (default third_party/tflite-micro; the first run downloads GCC and CMSIS)
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); TFLM=$(cd "${1:-$ROOT/third_party/tflite-micro}" && pwd)
MAKE=$(command -v gmake || command -v make)
cd "$TFLM"
for bt in release release_with_logs; do
  "$MAKE" -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp \
    OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=$bt KERNEL_OPTIMIZATION_LEVEL=-Os THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL=-Os \
    OPTIMIZE_KERNELS_FOR=KERNELS_OPTIMIZED_FOR_SIZE microlite
done
ls -la gen/cortex_m_generic_cortex-m4+fp_*_cmsis_nn_gcc/lib/libtensorflow-microlite.a
