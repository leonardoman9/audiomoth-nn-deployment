# WrenNet on AudioMoth

Firmware for the **AudioMoth Dev** (EFM32WG380F256, Cortex-M4F at 48 MHz). It records from the microphone and classifies
bird song with the **WrenNet** network (8 species + "non-bird"). All computation runs on the board:

```
microphone -> ADC 384 kHz -> decimation to 32 kHz -> C front-end (STFT + 64 learned filters, dB, normalisation)
           -> int8 CNN backbone (TFLM + CMSIS-NN) -> GRU + attention + FC in C -> 9 logits -> LOG.TXT on microSD
```

Every day at **07:00** (board clock) it records the whole **5-minute** window to the microSD card, then classifies the
recording in 3 s windows and appends one line per window to `LOG.TXT`. Every second of the window is classified, and
the audio stays on the card so the results can be checked by ear or re-analysed later. The rest of the time the board
sleeps (EM4). It does not use USB, GPS or the configuration app.

> **Status.** Verified in emulation only: host simulator, Cortex-M4 in QEMU, FatFS on a FAT32 image.
> **It has not yet run on a real board.** Timing and energy figures are estimates. Full results in
> [docs/VERIFICATION.md](docs/VERIFICATION.md).

| | |
|---|---|
| Flash | 233,476 B of 245,760 (~12 KB margin) |
| Internal RAM | .data 192 B + .bss 16,160 B, stack ~2.7 KB |
| External SRAM | 258,512 B of 262,144 (96 KiB TFLM arena, front-end, 1.5 s recording ring) |
| Model | int8 backbone 94.7 KB, 151 operators; tail (GRU 64) in C with int8 weights |
| Accuracy (8 species, 161 windows) | float 76.4 %, device pipeline 75.8 % |

Classes, in logit order: Bubo bubo, Certhia familiaris, Apus apus, Certhia brachydactyla, Emberiza cia,
Lophophanes cristatus, Periparus ater, Poecile montanus, non-bird.

## Requirements

**Hardware**
- AudioMoth **Dev** with external SRAM and a microSD slot. The USB microphone versions (≥ AM_VERSION_4) have neither SD
  nor external SRAM.
- microSD card formatted **FAT32**. FatFS is trimmed: no exFAT, 8.3 file names only.
- To program the board: the AudioMoth Flash App (bootloader) or J-Link/SWD.

**Software (macOS or Linux)**
- `git`, `curl`, **GNU make ≥ 4**. On macOS: `brew install make`, which installs `gmake`; the system `make` is too old.
- `python3` with `numpy`.
- You do **not** need to install an ARM toolchain: TFLM downloads one (GCC 14.3.1). GNU Arm 10.3-2021.10 also works and
  gives the same size (see VERIFICATION.md).
- For the tests only: `cc`/`clang`, `qemu-system-arm`, `mtools`. Comparisons against the model also need a Python
  environment with `ai_edge_litert`, `torch`, `torchaudio`, `soundfile`.

The model that goes into the firmware is in this repository, as a **model package** in `firmware/bird/model/`
(int8 backbone `bird_backbone_int8.tflite`, sha256 `6f77fa86…`, from the WrenNet repository, plus the generated C
headers and the class list; see "Other models" below). The PyTorch checkpoint
(`best_distillation_model.pt`) and `bird_reference.npz` from WrenNet are only needed for the comparisons against PyTorch.

## Build

```sh
scripts/setup_tflm.sh    # fetches tflite-micro at the verified commit (e0021ca) and applies firmware/patches/tflm-no-int4.patch
scripts/build_tflm.sh    # builds TFLM for Cortex-M4F; the first run downloads GCC and CMSIS (a few minutes)
firmware/app/build_app.sh third_party/tflite-micro firmware/bird/model
```

The output is `firmware/app/out/bird_logger.bin`, together with `bird_logger.axf`. The script prints the size:
`flash image: 233476 bytes (limit 245760)`. The flags in `build_tflm.sh` (kernels at `-Os`) are required: without them the
firmware exceeds 244 KB.

The **diagnostic firmware** records WAV files instead of classifying. Use it to check the microphone on a real board:

```sh
EXTRA_CF=-DBIRD_DIAG_RECORD firmware/app/build_app.sh third_party/tflite-micro firmware/bird/model
```

### Other models

The firmware takes the model from a **model package** folder: the int8 backbone, the generated C headers (learned
filterbank, tail weights, `model_config.h`) and `model_config.json`. The layout and the generator of each file are in
[firmware/bird/model/README.md](firmware/bird/model/README.md). The architecture is fixed (WrenNet: 32 kHz, 3 s
windows); the number of classes, their names, the "nothing of interest" class, the daily window, the step between classified
windows (hop; 1.5 s = half overlap, twice the windows and the classification time) and the confidence threshold come
from the package:

```sh
python3 firmware/bird/gen_model_config.py path/to/package --names "Species A,Species B,background" \
    --window-start 05:30 --window-minutes 10 --hop-seconds 1.5 --confidence 60
firmware/app/build_app.sh third_party/tflite-micro path/to/package
```

Each class costs ~88 B of flash, so up to **~180 classes** fit; beyond that the link fails with "FLASH memory
overflowed". `tests/make_test_package.py` makes packages with any number of classes (meaningless weights) to test the
firmware; CI runs every test with the default model, with 3- and 40-class packages and with a 1.5 s hop.

## Using the board

1. Flash the `.bin` with the bootloader; the application starts at `0x4000`.
2. Create `CONFIG.TXT` on the microSD card to set the clock (see below).
3. Insert the card and power on. At boot the clock is set and `CONFIG.TXT` is rewritten as `APPLIED <time>`, so it is not
   applied again.
4. From 07:00 to 07:05 the board records to `20261021/070000.WAV` (one folder per day). Then it classifies the
   recording, which takes several minutes (estimate: 6–8 s per 3 s window, to be measured on the board), and writes one
   line per window to `LOG.TXT`. The time on each line is the time of that window in the recording:

   ```
   2026-10-21 07:00:09,Bubo bubo,69,Y,241,-87,-15,-19,-65,-56,-37,-39,-79
   time, most likely class, probability %, Y if >= 50 %, 9 logits x100
   ```

Start time, duration, the step between windows (hop) and the confidence threshold come from the model package
(`gen_model_config.py`, see "Other models").

To update the firmware on a deployed board, flash it when `STATE.TXT` says `DONE` (or on a fresh card): a pending
session left by an older firmware version is not resumed.

### What is on the card

| file | content |
|---|---|
| `CONFIG.TXT` | the clock to set (you write it; see below); `APPLIED ...` once used |
| `YYYYMMDD/HHMMSS.WAV` | the recording of each session: 32 kHz, mono, 16 bit, ~19 MB for 5 minutes (~3.5 GB in 6 months) |
| `LOG.TXT` | one line per classified window |
| `SESSIONS.TXT` | one line per session: start, recording, seconds recorded, samples lost to SD stalls, windows, classification seconds |
| `STATE.TXT` | progress of the current session (`REC`, `CLS`, `DONE`), a fixed 96-byte record rewritten in place: after a reset or a power cut the board resumes from here |
| `REPLAY.PCM` | optional, for the replay test (below) |

If the card cannot keep up, or a write fails (the SD driver gives up on a card busy for more than 500 ms; a full
card), the lost audio is written as silence and counted in `SESSIONS.TXT`, so the timeline of the recording stays exact.
After a reset or a power cut the recording keeps what was saved up to ~2.6 s before it, and the next boot classifies it
(and any unfinished classification) even without a clock: every reset other than a wake-up from sleep clears the
clock (that is how the AudioMoth library works), so a new `CONFIG.TXT` is needed afterwards for the next sessions.
The board wakes 2 s before the window to create the file, so the recording starts exactly on the window's first second.

### CONFIG.TXT (setting the clock)

The board has no clock until it reads one from the card. The file is not in this repository: you create it.

- **Where:** in the **root** of the microSD card (not in a folder), named `CONFIG.TXT` (FAT is case-insensitive, so
  `config.txt` works too). The card must be FAT32.
- **Format:** plain ASCII text, one line, `YYYY-MM-DD hh:mm:ss`, 24-hour clock:

  ```
  2026-10-21 06:58:30
  ```

  Spaces, tabs and a line ending (`\n` or `\r\n`) before or after the time are fine; anything else is rejected: other
  characters after the seconds, `T` instead of the space, missing seconds, impossible dates such as `2026-02-30`, years
  outside 2020–2100.
- **Time zone:** the firmware uses the time exactly as written, without time zones or daylight saving. The daily window
  is 07:00–07:05 *of that clock*: write local time if you want 07:00 local, but it will not follow summer/winter time
  changes. If you write UTC, the window is 07:00 UTC.
- **When it is read:** only at power-up when the board has no valid time: the first power-up, or after the batteries
  were removed. The time is applied at the moment of boot, so write a time a few seconds ahead and power on when it
  is reached. Accuracy is that of your timing (seconds).
- **After it is read:** the file is overwritten with `APPLIED 2026-10-21 06:58:30`, so the same time is not applied again
  at the next boot. To set the clock again, write a new time into the file and remove power so the board loses its time.
- **If it is missing or invalid:** the board blinks red 3 times and retries every 60 s; it does nothing else until it
  finds a valid file.

If power is lost, the time is lost too and a new `CONFIG.TXT` is needed. Clock drift is ±0.9 s per day.

**Replay mode.** If the card holds `REPLAY.PCM`, the firmware classifies that window instead of the microphone and writes
a `REPLAY,...` line to the log. The file contains one known 3 s window and is prepared with `tests/sim/run_replay.py`.
It checks that the computation on the board matches the one on the computer.

**First test on a real board:** follow [docs/FIRST_BOARD_TEST.md](docs/FIRST_BOARD_TEST.md) (LED codes, diagnostic
recording, level analysis, replay comparison, timing and power), using the tools in `tools/board/`.

## Verifying without a board

Everything below also runs on every push in GitHub Actions (`.github/workflows/ci.yml`). The simulator needs the host
TFLM library: `HOST=1 scripts/build_tflm.sh`.

```sh
T=third_party/tflite-micro; M=firmware/bird/model    # or another model package; the Python tests read MODEL_PKG

# schedule, log, recording file names, WAV header, STATE.TXT and SESSIONS.TXT logic: 3,228 cases checked against Python
cc -Ifirmware/app -I$M/generated firmware/app/schedule.c tests/schedule/test_schedule.c -lm -o /tmp/ts && python3 tests/schedule/test_schedule.py /tmp/ts

# simulator: main_app.c and the real pipeline on a software model of the board, over several simulated days
tests/sim/build_sim.sh $T $M
python3 tests/sim/make_test_audio.py audio.raw 60      # synthetic audio; or make_audio.py from real recordings
python3 tests/sim/run_scenario.py tests/sim/out/bird_logger_sim audio.raw /tmp/work 2
python3 tests/sim/check_long_run.py /tmp/work 2026-10-21 2   # every window of every day, sessions, recordings
python3 tests/sim/check_recording.py audio.raw /tmp/work/sd  # the recordings are the microphone input
python3 tests/sim/verify_log.py /path/to/WrenNet best_distillation_model.pt /tmp/work    # comparison with PyTorch

# fault injection: no SD, no SRAM, failed writes, SD stalls and slow cards, reset and power cut during the recording
# and during the classification (and 25 cut times), a full card, a window that always crashes, an unreadable pending
# recording, a late start, bad CONFIG.TXT / STATE.TXT / REPLAY.PCM, silence, little time left: 44 scenarios.
# Everything at once (what CI runs): scripts/run_tests.sh $T $M
SAN=1 OUTSUFFIX=_san tests/sim/build_sim.sh $T $M      # with AddressSanitizer + UBSan
python3 tests/sim/run_faults.py tests/sim/out_san/bird_logger_sim audio.raw /tmp/faults

# file system: the real FatFS on a FAT32 image (for example: mformat -C -F -T 262144 -i card.img ::)
tests/fstest/run_fstest.sh card.img $M
```

The full pipeline on an emulated Cortex-M4F (PCM to logits, compared with the host pipeline and PyTorch):
`tests/qemu/run_qemu_pipeline.sh $T python3 $M` (needs `qemu-system-arm`, numpy and `ai_edge_litert`). The instruction and
memory profile uses `firmware/bird/build_bird_qemu.sh` and `tools/memcount.c`. Procedure and results are in
[docs/VERIFICATION.md](docs/VERIFICATION.md).

## Layout

```
firmware/app      application: main_app.c (hardware), schedule.c (schedule and log), fatfs_conf/, build_app.sh
firmware/bird     front-end, TFLM backbone, GRU/attention tail, QEMU and generator scripts
firmware/bird/model  default model package (int8 backbone, generated headers, class list, test vectors)
firmware/patches  TFLM patch (removes the int4 kernels, saves flash)
vendor/audiomoth  AudioMoth hardware library (emlib, FatFS, drivers, audiomoth.c) from AudioMoth-Firmware-Basic
tests/            sim (simulator, fault injection, long runs), fstest (FatFS), schedule, qemu (pipeline test + vectors)
tools/            board/ (first-board-test kit), table generator, QEMU instruction-count plugin, energy parser
scripts/          setup_tflm.sh, build_tflm.sh
docs/             VERIFICATION.md (all results), FIRST_BOARD_TEST.md, TFLM_BUILD.md, datasheets
benchmarks/       Raspberry Pi benchmarks (earlier work; audio-model-int8.tflite, 41 MB, is not in the repo)
```

## Known limitations

- After any reset that is not a wake-up from sleep (watchdog, brown-out, power cut) the library clears the clock, and
  sessions stop until a new `CONFIG.TXT` (pending work is still finished; files written without a clock show the year
  2098 on a computer). Keeping the clock across a watchdog reset
  would need a change in the AudioMoth library, to be tested on the board.
- Never run on hardware. Still to be tested: enabling the external SRAM, the SD driver (SPI) and its write speed while
  recording, real microphone levels, timing (estimate: ~6–8 s of compute per window, so ~10–13 minutes awake for a
  5-minute window) and energy. The estimates can be off by about 2×.
- The model was trained on clips centred on calls; here fixed 3 s windows are used, so accuracy in the field may be lower.
  There is no false-positive mitigation yet.

## License

MIT, see [LICENSE](LICENSE). The library in `vendor/audiomoth` is by Open Acoustic Devices.
