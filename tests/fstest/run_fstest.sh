#!/bin/sh
# Runs the app's SD-card call sequences on the real FatFS (trimmed ffconf.h) over a FAT32 image, then reads it back with mtools.
# Usage: run_fstest.sh CARD.img [MODEL_PACKAGE]   (a FAT32 image, e.g. made with mformat; it is modified)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); V="$ROOT/vendor/audiomoth"; OUT="$HERE/out"; mkdir -p "$OUT"
IMG=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); export MTOOLS_SKIP_CHECK=1
GEN="$(cd "${2:-$ROOT/firmware/bird/model}" && pwd)/generated"
python3 "$HERE/extract_fs_funcs.py" "$V/src/audiomoth.c" "$OUT/fs_audiomoth_funcs.inc"
cc -O1 -w -I"$ROOT/firmware/app/fatfs_conf" -I"$V/fatfs/inc" -I"$OUT" -I"$GEN" "$V/fatfs/src/ff.c" "$HERE/fs_shim.c" "$HERE/diskio_img.c" \
  "$HERE/fs_test.c" "$ROOT/firmware/app/schedule.c" -lm -o "$OUT/fs_test"
printf '2026-10-21 06:58:30\n' > "$OUT/CONFIG.TXT"; mdel -i "$IMG" ::LOG.TXT 2>/dev/null || true; mcopy -o -i "$IMG" "$OUT/CONFIG.TXT" ::CONFIG.TXT
SD_IMAGE="$IMG" "$OUT/fs_test"
echo "CONFIG.TXT: $(mtype -i "$IMG" ::CONFIG.TXT)"; echo "LOG.TXT lines: $(mtype -i "$IMG" ::LOG.TXT | wc -l)"
