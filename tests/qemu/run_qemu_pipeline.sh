#!/bin/sh
# End-to-end test on an emulated Cortex-M4F: one known 3 s PCM window -> C front-end -> TFLM int8 backbone -> C tail,
# compared with the logits of the host pipeline and of PyTorch stored in vectors/pipe_vec.cc. Prints PASS or FAIL.
# Usage: run_qemu_pipeline.sh TFLM_ROOT [PYTHON]   (PYTHON needs numpy + ai_edge_litert; qemu-system-arm must be installed)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); B="$ROOT/firmware/bird"; OUT="$HERE/out"; mkdir -p "$OUT"
TFLM=$(cd "$1" && pwd); PY=${2:-python3}
case "$(uname)" in Darwin) EXT=dylib;; *) EXT=so;; esac
cc -O2 -shared -fPIC -I"$B" -I"$B/generated" "$B/bird_tail.c" -lm -o "$OUT/libtail.$EXT"
PIPE_VEC="$HERE/vectors/pipe_vec.cc" OUTSUFFIX=_ci sh "$B/build_bird_qemu.sh" "$TFLM" "$B/model/bird_backbone_int8.tflite" \
  "$HERE/vectors/specs_one.npz" "$OUT/libtail.$EXT" "$B/generated" "$PY"
qemu-system-arm -M mps2-an386 -cpu cortex-m4 -display none -serial none -monitor none \
  -semihosting-config enable=on,target=native -kernel "$B/out_bird_ci/fw.elf" 2>&1 | tee "$OUT/qemu.txt"   # some QEMU builds print semihosting output on stderr
grep -q '^PASS' "$OUT/qemu.txt"
