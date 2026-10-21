#!/bin/sh
# Runs the same nn_real.cc + selftest.cc as the AudioMoth firmware under QEMU.
# Usage: build_qemu.sh TFLM_ROOT WRENNET_ROOT
set -e
TFLM=$(cd "$1" && pwd); WREN=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd); P=$(cd "$HERE/.." && pwd); OUT="$HERE/out_qemu"; mkdir -p "$OUT"
GCC="$TFLM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi"
D="$TFLM/tensorflow/lite/micro/tools/make/downloads"
LIB="$TFLM/gen/cortex_m_generic_cortex-m4+fp_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
ARCH="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
TFINC="-I$HERE/qemu -I$HERE -I$TFLM -I$D/flatbuffers/include -I$D/gemmlowp -I$D/ruy -I$D/cmsis_nn -I$D/cmsis_nn/Include -I$D/cmsis/CMSIS/Core/Include"
CXXF="-std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -ffunction-sections -fdata-sections -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DNDEBUG"
python3 "$HERE/gen_fw_data.py" "$WREN" "$OUT/gen"
sed "s/LENGTH = 64K/LENGTH = 96K/" "$P/link.ld" > "$OUT/link.ld"
$GCC-gcc $ARCH -Os -c "$P/startup.c" -o "$OUT/startup.o"
for f in "$HERE/nn_real.cc" "$HERE/selftest.cc" "$HERE/qemu/qemu_main.cc" "$OUT"/gen/*.cc; do
  $GCC-g++ $ARCH -Os $CXXF $TFINC -c "$f" -o "$OUT/$(basename "$f" .cc).o"
done
$GCC-g++ $ARCH -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -T "$OUT/link.ld" \
  "$OUT"/*.o "$LIB" -lm -o "$OUT/fw.elf"
$GCC-size -A "$OUT/fw.elf" | grep -E "^\.(text|data|bss|extsram)"
