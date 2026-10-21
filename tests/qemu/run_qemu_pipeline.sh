#!/bin/sh
# End-to-end test on an emulated Cortex-M4F: one known 3 s PCM window -> C front-end -> TFLM int8 backbone -> C tail,
# compared with the logits of the host pipeline and of PyTorch stored in the package (test/pipe_vec.cc). Prints PASS or FAIL.
# Usage: run_qemu_pipeline.sh TFLM_ROOT [PYTHON] [MODEL_PACKAGE]   (PYTHON needs numpy + ai_edge_litert; qemu-system-arm installed)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); B="$ROOT/firmware/bird"; OUT="$HERE/out"; mkdir -p "$OUT"
TFLM=$(cd "$1" && pwd); PY=${2:-python3}; PKG=$(cd "${3:-$B/model}" && pwd); MODEL=$(ls "$PKG"/*.tflite | head -n 1)
case "$(uname)" in Darwin) EXT=dylib;; *) EXT=so;; esac
cc -O2 -shared -fPIC -I"$B" -I"$PKG/generated" "$B/bird_tail.c" -lm -o "$OUT/libtail.$EXT"
PIPE_VEC="$PKG/test/pipe_vec.cc" OUTSUFFIX=_ci sh "$B/build_bird_qemu.sh" "$TFLM" "$MODEL" \
  "$PKG/test/specs_one.npz" "$OUT/libtail.$EXT" "$PKG/generated" "$PY"
qemu-system-arm -M mps2-an386 -cpu cortex-m4 -display none -serial none -monitor none \
  -semihosting-config enable=on,target=native -kernel "$B/out_bird_ci/fw.elf" 2>&1 | tee "$OUT/qemu.txt"   # some QEMU builds print semihosting output on stderr
tr -d '\r' < "$OUT/qemu.txt" | grep -q '^PASS'
