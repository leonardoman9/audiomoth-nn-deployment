#!/bin/sh
# Host simulator of the bird logger: main_app.c + schedule.c + the REAL front-end, TFLM backbone,
# tail and DigitalFilter, on a software model of the AudioMoth (hal_sim.c).
# Usage: [SAN=1] [OUTSUFFIX=_x] build_sim.sh TFLM_ROOT MODEL.tflite   (SAN=1: AddressSanitizer + UBSan on our code)
set -e
# The host uses 64-bit pointers, so the interpreter needs more arena than the 96 KiB of the 32-bit device.
TFLM=$(cd "$1" && pwd); MODEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); APP="$ROOT/firmware/app"; REPO="$ROOT/vendor/audiomoth"
BIRD="$ROOT/firmware/bird"; OUT="$HERE/out${OUTSUFFIX:-}"; mkdir -p "$OUT"
# vptr is off: TFLM is built without RTTI, so its typeinfo does not exist (Linux ld fails on it)
CF="-O1"; LF=""; [ -n "$SAN" ] && { CF="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize=vptr -fno-sanitize-recover=undefined"; LF="-fsanitize=address,undefined"; }
D="$TFLM/tensorflow/lite/micro/tools/make/downloads"
# host library (scripts/build_tflm.sh with HOST=1): gen/<os>_<arch>_release_with_logs_cmsis_nn_gcc
LIB=$(ls "$TFLM"/gen/*_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a 2>/dev/null | grep -v cortex_m | head -n 1)
[ -n "$LIB" ] || { echo "no host TFLM library in $TFLM/gen (run HOST=1 scripts/build_tflm.sh)"; exit 1; }
TF="-I$TFLM -I$D/flatbuffers/include -I$D/gemmlowp -I$D/ruy -I$D/cmsis_nn -I$D/cmsis_nn/Include -I$D/cmsis/CMSIS/Core/Include"
python3 - "$MODEL" "$OUT/bird_model.cc" <<'PY'
import sys
d = open(sys.argv[1], "rb").read()
open(sys.argv[2], "w").write('#include <stdint.h>\nextern "C" const unsigned char g_bird_model[%d] __attribute__((aligned(16))) = {%s};\n' % (len(d), ",".join(map(str, d))))
PY
# digitalfilter.c is the real firmware code; it only needs the standard headers on the host.
SIMDEF="-DBirdPipeline_Begin=sim_BirdPipeline_Begin -DBirdPipeline_Push=sim_BirdPipeline_Push -DBirdPipeline_Finish=sim_BirdPipeline_Finish"
clang $CF -Wall -Wno-unused-function -I"$HERE" -I"$REPO/inc" -I"$APP" -I"$BIRD" -c "$REPO/src/digitalfilter.c" -o "$OUT/digitalfilter.o" 2>&1 | grep -E "error" || true
clang $CF -Wall $EXTRA_CF -I"$HERE" -I"$REPO/inc" -I"$APP" -I"$BIRD" $SIMDEF -c "$APP/main_app.c" -o "$OUT/main_app.o"
clang $CF -I"$APP" -c "$APP/schedule.c" -o "$OUT/schedule.o"
clang $CF -I"$HERE" -I"$BIRD" -c "$HERE/hal_sim.c" -o "$OUT/hal_sim.o"
clang $CF -fno-math-errno -I"$BIRD" -I"$BIRD/generated" -c "$BIRD/bird_frontend.c" -o "$OUT/bird_frontend.o"
clang $CF -I"$BIRD" -I"$BIRD/generated" -c "$BIRD/bird_tail.c" -o "$OUT/bird_tail.o"
for f in "$BIRD/nn_bird.cc" "$BIRD/bird_pipeline.cc" "$OUT/bird_model.cc"; do
  clang++ -std=c++17 $CF -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DNDEBUG -DBIRD_ARENA_BYTES=196608 -I"$HERE" -I"$BIRD" $TF -c "$f" -o "$OUT/$(basename "$f" .cc).o"
done
clang++ $LF "$OUT"/*.o "$LIB" -lm -o "$OUT/bird_logger_sim"
echo "built $OUT/bird_logger_sim"
