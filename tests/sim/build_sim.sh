#!/bin/sh
# Host simulator of the bird logger: main_app.c + schedule.c + the REAL front-end, TFLM backbone,
# tail and DigitalFilter, on a software model of the AudioMoth (hal_sim.c).
# Usage: build_sim.sh TFLM_ROOT MODEL.tflite
set -e
# The host uses 64-bit pointers, so the interpreter needs more arena than the 96 KiB of the 32-bit device.
TFLM=$(cd "$1" && pwd); MODEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); APP="$ROOT/firmware/app"; REPO="$ROOT/vendor/audiomoth"
BIRD="$ROOT/firmware/bird"; OUT="$HERE/out"; mkdir -p "$OUT"
D="$TFLM/tensorflow/lite/micro/tools/make/downloads"
LIB="$TFLM/gen/osx_arm64_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
TF="-I$TFLM -I$D/flatbuffers/include -I$D/gemmlowp -I$D/ruy -I$D/cmsis_nn -I$D/cmsis_nn/Include -I$D/cmsis/CMSIS/Core/Include"
python3 - "$MODEL" "$OUT/bird_model.cc" <<'PY'
import sys
d = open(sys.argv[1], "rb").read()
open(sys.argv[2], "w").write('#include <stdint.h>\nextern "C" const unsigned char g_bird_model[%d] __attribute__((aligned(16))) = {%s};\n' % (len(d), ",".join(map(str, d))))
PY
# digitalfilter.c is the real firmware code; it only needs the standard headers on the host.
SIMDEF="-DBirdPipeline_Begin=sim_BirdPipeline_Begin -DBirdPipeline_Push=sim_BirdPipeline_Push -DBirdPipeline_Finish=sim_BirdPipeline_Finish"
clang -O1 -Wall -Wno-unused-function -I"$HERE" -I"$REPO/inc" -I"$APP" -I"$BIRD" -c "$REPO/src/digitalfilter.c" -o "$OUT/digitalfilter.o" 2>&1 | grep -E "error" || true
clang -O1 -Wall $EXTRA_CF -I"$HERE" -I"$REPO/inc" -I"$APP" -I"$BIRD" $SIMDEF -c "$APP/main_app.c" -o "$OUT/main_app.o"
clang -O1 -I"$APP" -c "$APP/schedule.c" -o "$OUT/schedule.o"
clang -O1 -I"$HERE" -I"$BIRD" -c "$HERE/hal_sim.c" -o "$OUT/hal_sim.o"
clang -O1 -fno-math-errno -I"$BIRD" -I"$BIRD/generated" -c "$BIRD/bird_frontend.c" -o "$OUT/bird_frontend.o"
clang -O1 -I"$BIRD" -I"$BIRD/generated" -c "$BIRD/bird_tail.c" -o "$OUT/bird_tail.o"
for f in "$BIRD/nn_bird.cc" "$BIRD/bird_pipeline.cc" "$OUT/bird_model.cc"; do
  clang++ -std=c++17 -O1 -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DNDEBUG -DBIRD_ARENA_BYTES=196608 -I"$HERE" -I"$BIRD" $TF -c "$f" -o "$OUT/$(basename "$f" .cc).o"
done
clang++ "$OUT"/*.o "$LIB" -lm -o "$OUT/bird_logger_sim"
echo "built $OUT/bird_logger_sim"
