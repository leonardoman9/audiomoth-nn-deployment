#!/bin/sh
# Cross-build the TFLM parity test for Cortex-M4F and (optionally) run in QEMU.
# Usage: [RAM_KB=32] build.sh TFLM_ROOT WRENNET_ROOT [ARENA_BYTES]
set -e
TFLM=$(cd "$1" && pwd); WREN=$(cd "$2" && pwd); ARENA=${3:-28672}
HERE=$(cd "$(dirname "$0")" && pwd); OUT="$HERE/out"; mkdir -p "$OUT"
GCC="$TFLM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi"
D="$TFLM/tensorflow/lite/micro/tools/make/downloads"
LIB="$TFLM/gen/cortex_m_generic_cortex-m4+fp_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
sed "s/LENGTH = 32K/LENGTH = ${RAM_KB:-32}K/" "$HERE/link.ld" > "$OUT/link.ld"
python3 "$HERE/gen_data.py" "$WREN" "$OUT/gen"
ARCH="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
INC="-I$OUT/gen -I$TFLM -I$D/flatbuffers/include -I$D/gemmlowp -I$D/ruy -I$D/cmsis_nn -I$D/cmsis_nn/Include -I$D/cmsis/CMSIS/Core/Include"
# Model/vector arrays are separate translation units so they can be sized apart.
for n in backbone_model streaming_model backbone_in backbone_out seq_feat seq_hidden seq_logits; do
  printf '#include <cstdint>\n#include <cstdalign>\n#include "%s.inc"\n' "$n" > "$OUT/gen/$n.cc"
  $GCC-g++ $ARCH -Os -std=c++17 -fno-exceptions -fno-rtti -c "$OUT/gen/$n.cc" -I"$OUT/gen" -o "$OUT/$n.o"
done
$GCC-gcc $ARCH -Os -c "$HERE/startup.c" -o "$OUT/startup.o"
$GCC-g++ $ARCH -Os -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -ffunction-sections -fdata-sections \
  -DTF_LITE_STATIC_MEMORY -DNDEBUG -DCMSIS_NN -DARENA_BYTES=$ARENA ${EXTRA_DEFS:-} $INC -c "$HERE/main.cc" -o "$OUT/main.o"
$GCC-g++ $ARCH -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -Wl,-Map,"$OUT/fw.map" \
  -T "$OUT/link.ld" "$OUT/startup.o" "$OUT/main.o" "$OUT"/*_model.o "$OUT"/backbone_in.o "$OUT"/backbone_out.o \
  "$OUT"/seq_*.o "$LIB" -lm -o "$OUT/fw.elf"
$GCC-size -A "$OUT/fw.elf" | grep -E "^\.(text|data|bss|testdata)"
