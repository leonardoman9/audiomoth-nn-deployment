#!/bin/sh
# Builds the WrenNet bird logger for the AudioMoth (EFM32WG380F256) and prints the flash/RAM it needs.
# Usage: [EXTRA_CF=-DBIRD_DIAG_RECORD] build_app.sh TFLM_ROOT MODEL.tflite   (the diagnostic build records WAV files, no network)
#   TFLM_ROOT: TFLM built as in docs/TFLM_BUILD.md with BUILD_TYPE=release (see the "size" variant) and the
#   int4 patch in firmware/patches/. The hardware libraries (emlib, device, emusb, drivers, fatfs) come from this repo.
set -e
TFLM=$(cd "$1" && pwd); MODEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
HERE=$(cd "$(dirname "$0")" && pwd); P=$(cd "$HERE/.." && pwd); ROOT=$(cd "$P/.." && pwd); REPO="$ROOT/vendor/audiomoth"
BIRD="$P/bird"; OUT="$HERE/out"; mkdir -p "$OUT/obj"
GCC="$TFLM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi"
D="$TFLM/tensorflow/lite/micro/tools/make/downloads"
LIB="$TFLM/gen/cortex_m_generic_cortex-m4+fp_release_cmsis_nn_gcc/lib/libtensorflow-microlite.a"
ARCH="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
# fatfs_conf/ holds copies of the FatFS headers next to our ffconf.h: ff.h includes "ffconf.h" from its own directory.
HW="-I$HERE/fatfs_conf -I$REPO/cmsis -I$REPO/device/inc -I$REPO/emlib/inc -I$REPO/emusb/inc -I$REPO/drivers/inc -I$REPO/fatfs/inc -I$REPO/gps/inc -I$REPO/inc"
DEFS="-DEFM32WG380F256=1 -DARM_MATH_CM4=1 -DNDEBUG"
TF="-I$TFLM -I$D/flatbuffers/include -I$D/gemmlowp -I$D/ruy -I$D/cmsis_nn -I$D/cmsis_nn/Include -I$D/cmsis/CMSIS/Core/Include"
CXXF="-std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -ffunction-sections -fdata-sections -DTF_LITE_STATIC_MEMORY -DCMSIS_NN -DTF_LITE_STRIP_ERROR_STRINGS -DBIRD_ARENA_BYTES=98304"
CF="-Os -std=gnu99 -ffunction-sections -fdata-sections $EXTRA_CF"

# model as a C array in flash
python3 - "$MODEL" "$OUT/bird_model.cc" <<'PY'
import sys
d = open(sys.argv[1], "rb").read()
open(sys.argv[2], "w").write('#include <stdint.h>\nextern "C" const unsigned char g_bird_model[%d] __attribute__((aligned(16))) = {%s};\n' % (len(d), ",".join(map(str, d))))
PY

# digitalfilter.c keeps a 4 KB Hamming window that only the frequency trigger uses (never called here): shrink it.
sed 's/^#define MAXIMUM_HAMMING_WINDOW_LENGTH .*/#define MAXIMUM_HAMMING_WINDOW_LENGTH 16/' "$REPO/src/digitalfilter.c" > "$OUT/digitalfilter.c"
SRCS="$REPO/device/src/system_efm32wg.c $REPO/src/audiomoth.c $OUT/digitalfilter.c $REPO/src/biquad.c $REPO/src/butterworth.c \
  $REPO/drivers/src/dmactrl.c $REPO/drivers/src/microsd.c $REPO/fatfs/src/ff.c $REPO/fatfs/src/diskio.c"
if grep -q "define FF_USE_LFN[[:space:]]*[12]" "$HERE/fatfs_conf/ffconf.h"; then SRCS="$SRCS $REPO/fatfs/src/ffunicode.c"; fi
for f in $REPO/emlib/src/em_cmu.c $REPO/emlib/src/em_system.c $REPO/emlib/src/em_emu.c $REPO/emlib/src/em_gpio.c $REPO/emlib/src/em_adc.c \
  $REPO/emlib/src/em_dma.c $REPO/emlib/src/em_ebi.c $REPO/emlib/src/em_burtc.c $REPO/emlib/src/em_rtc.c $REPO/emlib/src/em_wdog.c \
  $REPO/emlib/src/em_prs.c $REPO/emlib/src/em_timer.c $REPO/emlib/src/em_opamp.c $REPO/emlib/src/em_dac.c $REPO/emlib/src/em_vcmp.c \
  $REPO/emlib/src/em_usart.c $REPO/emlib/src/em_msc.c $REPO/emlib/src/em_int.c $REPO/emlib/src/em_core.c $REPO/emlib/src/em_acmp.c \
  $REPO/emlib/src/em_i2c.c $REPO/emlib/src/em_leuart.c $REPO/emlib/src/em_letimer.c $REPO/emlib/src/em_rmu.c; do
  [ -f "$f" ] && SRCS="$SRCS $f"
done
for f in $SRCS "$HERE/main_app.c" "$HERE/schedule.c"; do
  $GCC-gcc $ARCH $CF $DEFS $HW -I"$BIRD" -I"$HERE" -c "$f" -o "$OUT/obj/$(basename "$f" .c).o"
done
$GCC-gcc $ARCH -O2 -std=gnu99 -ffunction-sections -fdata-sections -fno-math-errno -I"$BIRD" -I"$BIRD/generated" -c "$BIRD/bird_frontend.c" -o "$OUT/obj/bird_frontend.o"
$GCC-gcc $ARCH $CF -DBIRD_TAIL_FLOAT_WEIGHTS_UNUSED $DEFS -I"$BIRD" -I"$BIRD/generated" -c "$BIRD/bird_tail.c" -o "$OUT/obj/bird_tail.o"
$GCC-gcc $ARCH -x assembler-with-cpp -c "$REPO/device/src/startup_efm32wg.s" -o "$OUT/obj/startup.o"
for f in "$BIRD/nn_bird.cc" "$BIRD/bird_pipeline.cc" "$OUT/bird_model.cc"; do
  $GCC-g++ $ARCH -Os $CXXF $DEFS $HW $TF -I"$BIRD" -I"$BIRD/generated" -c "$f" -o "$OUT/obj/$(basename "$f" .cc).o"
done
$GCC-g++ $ARCH --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -Wl,-Map,"$OUT/app.map" \
  -T "$P/audiomoth_tflm.ld" "$OUT"/obj/*.o "$LIB" -lm -Wl,--start-group -lgcc -lc -lnosys -lstdc++ -Wl,--end-group -o "$OUT/bird_logger.axf"
$GCC-objcopy -O binary "$OUT/bird_logger.axf" "$OUT/bird_logger.bin"
$GCC-size "$OUT/bird_logger.axf"
echo "flash image: $(wc -c < "$OUT/bird_logger.bin") bytes (limit 245760 = 0x3C000)"
