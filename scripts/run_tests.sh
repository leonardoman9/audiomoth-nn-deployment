#!/bin/sh
# Runs every test that does not need a board, for one model package (what CI runs, see .github/workflows/ci.yml):
# schedule logic, FatFS, firmware and diagnostic builds with the flash budget, simulator with sanitizers (2 days,
# recordings, 40 fault scenarios) and, if the package has test vectors, the pipeline on an emulated Cortex-M4F.
# Usage: scripts/run_tests.sh TFLM_ROOT [MODEL_PACKAGE] [AUDIO.raw] [WORKDIR]
#   TFLM_ROOT     built with HOST=1 scripts/build_tflm.sh; MODEL_PACKAGE default firmware/bird/model
#   AUDIO.raw     32 kHz int16 for the simulator (default: 60 s of synthetic audio); WORKDIR default /tmp/wren_tests
# Needs clang/cc, python3 + numpy, mtools; qemu-system-arm + ai_edge_litert for the QEMU step (skipped without them).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TFLM=$(cd "$1" && pwd); PKG=$(cd "${2:-$ROOT/firmware/bird/model}" && pwd); WORK=${4:-/tmp/wren_tests}
if [ -n "$3" ]; then AUDIO=$(cd "$(dirname "$3")" && pwd)/$(basename "$3"); fi
cd "$ROOT"
export MODEL_PKG="$PKG" ASAN_OPTIONS=detect_leaks=0 MTOOLS_SKIP_CHECK=1
SAN="-g -fsanitize=address,undefined -fno-sanitize-recover=undefined"
rm -rf "$WORK"; mkdir -p "$WORK"
AUDIO=${AUDIO:-$WORK/audio.raw}; [ -f "$AUDIO" ] || python3 tests/sim/make_test_audio.py "$AUDIO" 60
step() { echo; echo "=== $*"; }
# run a step, keep its output in a file, show the end; fails the script when the step fails (plain sh has no
# pipefail: "cmd | tail" would hide cmd's exit status)
run() { out="$WORK/step.log"; if "$@" > "$out" 2>&1; then tail -n "${KEEP:-3}" "$out"; else cat "$out"; echo "FAILED: $*"; exit 1; fi; }

step "schedule and log logic (sanitizers)"
cc $SAN -Ifirmware/app -I"$PKG/generated" firmware/app/schedule.c tests/schedule/test_schedule.c -lm -o "$WORK/test_schedule"
KEEP=2 run python3 tests/schedule/test_schedule.py "$WORK/test_schedule"
grep -q "mismatches 0" "$WORK/step.log"

step "FatFS on a FAT32 image (sanitizers)"
mformat -C -F -T 262144 -i "$WORK/card.img" ::
KEEP=8 run env FS_CFLAGS="$SAN" sh tests/fstest/run_fstest.sh "$WORK/card.img" "$PKG"

step "firmware and diagnostic builds, flash budget"
KEEP=2 run sh firmware/app/build_app.sh "$TFLM" "$PKG"
size=$(sed -n 's/^flash image: *\([0-9]*\) bytes.*/\1/p' "$WORK/step.log")
[ -n "$size" ] && [ "$size" -le 245760 ] || { echo "FAILED: flash budget ($size)"; exit 1; }
KEEP=1 run env EXTRA_CF=-DBIRD_DIAG_RECORD sh firmware/app/build_app.sh "$TFLM" "$PKG"

step "simulator (sanitizers): 2 days, recordings, fault scenarios"
KEEP=1 run env SAN=1 OUTSUFFIX=_tests sh tests/sim/build_sim.sh "$TFLM" "$PKG"
SIM=tests/sim/out_tests/bird_logger_sim
KEEP=1 run python3 tests/sim/run_scenario.py $SIM "$AUDIO" "$WORK/days" 2 "2026-10-21 00:00:05"
KEEP=4 run python3 tests/sim/check_long_run.py "$WORK/days" 2026-10-21 2
KEEP=1 run python3 tests/sim/check_recording.py "$AUDIO" "$WORK/days/sd"
KEEP=1 run python3 tests/sim/run_faults.py $SIM "$AUDIO" "$WORK/faults"
rm -rf "$WORK/days" "$WORK/faults"   # ~20 MB per recording

if [ -f "$PKG/test/pipe_vec.cc" ] && command -v qemu-system-arm >/dev/null && python3 -c "import ai_edge_litert" 2>/dev/null; then
  step "pipeline on an emulated Cortex-M4F (QEMU)"
  KEEP=2 run sh tests/qemu/run_qemu_pipeline.sh "$TFLM" python3 "$PKG"
else
  step "QEMU pipeline skipped (no test vectors in the package, or no qemu-system-arm / ai_edge_litert)"
fi
echo; echo "ALL TESTS PASSED for $PKG"
