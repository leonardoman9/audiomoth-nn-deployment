#!/bin/sh
# Bird classifier (TFLM backbone + C tail) on emulated Cortex-M4F.
# Env: PIPE_VEC (builds the PCM->logits program with that vector file), LIBVARIANT (release_with_logs | release), OUTSUFFIX.  Usage: build_bird_qemu.sh TFLM_ROOT MODEL.tflite SPECS.npz TAIL_LIB TAIL_HEADERS_DIR [PY_TF] [ARENA_BYTES]
set -e
TFLM=$(cd "$1" && pwd); MODEL=$2; SPECS=$3; TAILLIB=$4; TAILH=$5; PY=${6:-python3}; ARENA=${7:-163840}
HERE=$(cd "$(dirname "$0")" && pwd); P=$(cd "$HERE/../.." && pwd); OUT="$HERE/out_bird${OUTSUFFIX:-}"; mkdir -p "$OUT"
GCC="$TFLM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi"
D="$TFLM/tensorflow/lite/micro/tools/make/downloads"
LIB="$TFLM/gen/cortex_m_generic_cortex-m4+fp_${LIBVARIANT:-release_with_logs}_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
ARCH="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
INC="-I$HERE/../qemu -I$HERE -I$TAILH -I$TFLM -I$D/flatbuffers/include -I$D/gemmlowp -I$D/ruy -I$D/cmsis_nn -I$D/cmsis_nn/Include -I$D/cmsis/CMSIS/Core/Include"
CXXF="$EXTRA_CXXF -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -ffunction-sections -fdata-sections -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DNDEBUG -DBIRD_ARENA_BYTES=$ARENA"
# The library variant "release" strips error strings; the same define must reach our own code.
case "${LIBVARIANT:-release_with_logs}" in release) CXXF="$CXXF -DTF_LITE_STRIP_ERROR_STRINGS";; esac
"$PY" "$HERE/gen_bird_fw_data.py" "$MODEL" "$SPECS" "$TAILLIB" "$OUT/gen"
if [ -n "$PIPE_VEC" ]; then rm -f "$OUT/gen/bird_vec.cc"; cp "$PIPE_VEC" "$OUT/gen/pipe_vec.cc"; MAINSRC="$HERE/qemu_pipeline_main.cc"; EXTRA="$HERE/bird_pipeline.cc"; $GCC-gcc $ARCH -Os -fno-math-errno -ffunction-sections -fdata-sections -I"$HERE" -I"$HERE/generated" -c "$HERE/bird_frontend.c" -o "$OUT/bird_frontend.o"; else MAINSRC="$HERE/qemu_bird_main.cc"; EXTRA=""; fi
sed "s/LENGTH = 64K/LENGTH = 256K/" "$P/link.ld" > "$OUT/link.ld"
$GCC-gcc $ARCH -Os -c "$P/startup.c" -o "$OUT/startup.o"
$GCC-gcc $ARCH -Os -ffunction-sections -fdata-sections -I"$HERE" -I"$TAILH" -c "$HERE/bird_tail.c" -o "$OUT/bird_tail.o"
for f in "$HERE/nn_bird.cc" "$MAINSRC" $EXTRA "$OUT"/gen/*.cc; do
  $GCC-g++ $ARCH -Os $CXXF $INC -c "$f" -o "$OUT/$(basename "$f" .cc).o"
done
$GCC-g++ $ARCH -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -Wl,-Map,"$OUT/fw.map" -T "$OUT/link.ld" \
  "$OUT"/*.o "$LIB" -lm -o "$OUT/fw.elf"
$GCC-size -A "$OUT/fw.elf" | grep -E "^\.(text|data|bss|extsram|testdata)"
