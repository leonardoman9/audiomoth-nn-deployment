#!/bin/sh
# Builds the AudioMoth self-test firmware with real TFLM + CMSIS-NN.
# Usage: build_fw.sh TFLM_ROOT WRENNET_ROOT   (TFLM built as in ../README.md)
set -e
TFLM=$(cd "$1" && pwd); WREN=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); OUT="$HERE/out"; mkdir -p "$OUT/obj"
GCC="$TFLM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi"
D="$TFLM/tensorflow/lite/micro/tools/make/downloads"
LIB="$TFLM/gen/cortex_m_generic_cortex-m4+fp_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
ARCH="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
AMINC="-I$REPO/cmsis -I$REPO/device/inc -I$REPO/emlib/inc -I$REPO/emusb/inc -I$REPO/drivers/inc -I$REPO/fatfs/inc -I$REPO/gps/inc -I$REPO/inc"
DEFS="-DEFM32WG380F256=1 -DARM_MATH_CM4=1 -DNDEBUG"
TFINC="-I$TFLM -I$D/flatbuffers/include -I$D/gemmlowp -I$D/ruy -I$D/cmsis_nn -I$D/cmsis_nn/Include -I$D/cmsis/CMSIS/Core/Include"
CXXF="-std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -ffunction-sections -fdata-sections -DTF_LITE_STATIC_MEMORY -DCMSIS_NN"

python3 "$HERE/gen_fw_data.py" "$WREN" "$OUT/gen"

EXCL="em_crypto em_aes em_can em_lcd em_pcnt em_lesense em_qspi em_csen em_vdac em_idac"
CSRC="$REPO/device/src/system_efm32wg.c $REPO/src/audiomoth.c $REPO/drivers/src/dmactrl.c $REPO/emusb/src/*.c"
for f in $REPO/emlib/src/*.c; do
  n=$(basename "$f" .c); case " $EXCL " in *" $n "*) continue;; esac; CSRC="$CSRC $f"
done
for f in $CSRC; do
  $GCC-gcc $ARCH -Os -std=gnu99 -ffunction-sections -fdata-sections $DEFS $AMINC -c "$f" -o "$OUT/obj/$(basename "$f" .c).o"
done
$GCC-gcc $ARCH -x assembler-with-cpp -c "$REPO/device/src/startup_efm32wg.s" -o "$OUT/obj/startup.o"
for f in "$HERE/nn_real.cc" "$HERE/main.cc" "$OUT"/gen/*.cc; do
  $GCC-g++ $ARCH -Os $CXXF $DEFS $AMINC $TFINC -c "$f" -o "$OUT/obj/$(basename "$f" .cc).o"
done
$GCC-g++ $ARCH --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -Wl,-Map,"$OUT/fw.map" \
  -T "$HERE/audiomoth_tflm.ld" "$OUT"/obj/*.o "$LIB" -lm -Wl,--start-group -lgcc -lc -lnosys -lstdc++ -Wl,--end-group -o "$OUT/audiomoth_tflm.axf"
$GCC-objcopy -O binary "$OUT/audiomoth_tflm.axf" "$OUT/audiomoth_tflm.bin"
$GCC-objcopy -O ihex "$OUT/audiomoth_tflm.axf" "$OUT/audiomoth_tflm.hex"
$GCC-size "$OUT/audiomoth_tflm.axf"
ls -l "$OUT/audiomoth_tflm.bin"
