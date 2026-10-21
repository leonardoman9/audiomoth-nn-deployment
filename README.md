# WrenNet on AudioMoth

Firmware for the **AudioMoth Dev** (EFM32WG380F256, Cortex-M4F at 48 MHz). It records from the microphone and classifies
bird song with the **WrenNet** network (8 species + "non-bird"). All computation runs on the board:

```
microphone -> ADC 384 kHz -> decimation to 32 kHz -> C front-end (STFT + 64 learned filters, dB, normalisation)
           -> int8 CNN backbone (TFLM + CMSIS-NN) -> GRU + attention + FC in C -> 9 logits -> LOG.TXT on microSD
```

Every day at **07:00** (board clock) it classifies about **5 minutes** of audio in 3 s windows and appends one line per
window to `LOG.TXT`. The rest of the time the board sleeps (EM4). It does not use USB, GPS or the configuration app.

> **Status.** Verified in emulation only: host simulator, Cortex-M4 in QEMU, FatFS on a FAT32 image.
> **It has not yet run on a real board.** Timing and energy figures are estimates. Full results in
> [docs/VERIFICATION.md](docs/VERIFICATION.md).

| | |
|---|---|
| Flash | 230,124 B of 245,760 (~15.6 KB margin) |
| Internal RAM | .data 192 B + .bss 14,112 B, stack ~2.7 KB |
| External SRAM | ~160 KB (96 KiB TFLM arena + buffers) |
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

**From the WrenNet repository** (`docs/deployment/bird/`)
- `bird_backbone_int8.tflite`: the model that goes into the firmware.
- `bird_reference.npz` and the checkpoint `best_distillation_model.pt`: only needed for the comparison tests against PyTorch.

## Build

```sh
scripts/setup_tflm.sh    # fetches tflite-micro at the verified commit (e0021ca) and applies firmware/patches/tflm-no-int4.patch
scripts/build_tflm.sh    # builds TFLM for Cortex-M4F; the first run downloads GCC and CMSIS (a few minutes)
firmware/app/build_app.sh third_party/tflite-micro /path/to/WrenNet/docs/deployment/bird/bird_backbone_int8.tflite
```

The output is `firmware/app/out/bird_logger.bin`, together with `bird_logger.axf`. The script prints the size:
`flash image: 230124 bytes (limit 245760)`. The flags in `build_tflm.sh` (kernels at `-Os`) are required: without them the
firmware exceeds 244 KB.

The **diagnostic firmware** records WAV files instead of classifying. Use it to check the microphone on a real board:

```sh
EXTRA_CF=-DBIRD_DIAG_RECORD firmware/app/build_app.sh third_party/tflite-micro .../bird_backbone_int8.tflite
```

## Using the board

1. Flash the `.bin` with the bootloader; the application starts at `0x4000`.
2. On the microSD card, create `CONFIG.TXT` with a single line holding the current time: `2026-10-21 06:58:30`.
3. Insert the card and power on. At boot the clock is set and `CONFIG.TXT` is rewritten as `APPLIED <time>`, so it is not
   applied again.
4. From 07:00 to 07:05 the firmware writes one line about every 6 s to `LOG.TXT`:

   ```
   2026-10-21 07:00:09,Bubo bubo,69,Y,241,-87,-15,-19,-65,-56,-37,-39,-79
   time, most likely class, probability %, Y if >= 50 %, 9 logits x100
   ```

If power is lost, the time is lost too and a new `CONFIG.TXT` is needed. Clock drift is ±0.9 s per day.
Start time and duration are set in `firmware/app/schedule.h` (`SCHED_WINDOW_*`).

**Replay mode.** If the card holds `REPLAY.PCM`, the firmware classifies that window instead of the microphone and writes
a `REPLAY,...` line to the log. The file contains one known 3 s window and is prepared with `tests/sim/run_replay.py`.
It checks that the computation on the board matches the one on the computer.

**First test on a real board**, in this order:
1. diagnostic firmware → recorded WAVs: check microphone levels and noise;
2. normal firmware with `REPLAY.PCM` → compare the logits with the computer's;
3. full session at 07:00 → measure timing and power.

## Verifying without a board

```sh
T=third_party/tflite-micro; M=/path/to/WrenNet/docs/deployment/bird/bird_backbone_int8.tflite

# schedule and log logic: 1,805 cases checked against Python's datetime
cc -Ifirmware/app firmware/app/schedule.c tests/schedule/test_schedule.c -lm -o /tmp/ts && python3 tests/schedule/test_schedule.py /tmp/ts

# simulator: main_app.c and the real pipeline on a software model of the board, over several simulated days
tests/sim/build_sim.sh $T $M
python3 tests/sim/make_audio.py audio.raw <audio files or folders>
python3 tests/sim/run_scenario.py tests/sim/out/bird_logger_sim audio.raw /tmp/work 2
python3 tests/sim/verify_log.py /path/to/WrenNet best_distillation_model.pt /tmp/work    # comparison with PyTorch

# file system: the real FatFS on a FAT32 image (for example: mformat -C -F -T 262144 -i card.img ::)
tests/fstest/run_fstest.sh card.img
```

The full pipeline in QEMU (PCM to logits on an emulated Cortex-M4, compared byte for byte with the computer) and the
instruction and memory profile use `firmware/bird/build_bird_qemu.sh` and `tools/memcount.c`. Procedure and results are in
[docs/VERIFICATION.md](docs/VERIFICATION.md).

## Layout

```
firmware/app      application: main_app.c (hardware), schedule.c (schedule and log), fatfs_conf/, build_app.sh
firmware/bird     front-end, TFLM backbone, GRU/attention tail, generated tables, QEMU and generator scripts
firmware/patches  TFLM patch (removes the int4 kernels, saves flash)
vendor/audiomoth  AudioMoth hardware library (emlib, FatFS, drivers, audiomoth.c) from AudioMoth-Firmware-Basic
tests/            sim (simulator), fstest (FatFS), schedule, qemu (harness)
tools/            table generator, QEMU instruction-count plugin, oscilloscope energy parser
scripts/          setup_tflm.sh, build_tflm.sh
docs/             VERIFICATION.md (all results), TFLM_BUILD.md, datasheets
benchmarks/       Raspberry Pi benchmarks (earlier work; audio-model-int8.tflite, 41 MB, is not in the repo)
```

## Known limitations

- Never run on hardware. Still to be tested: enabling the external SRAM, the SD driver (SPI), real microphone levels,
  timing (estimate: ~6–8 s of compute per window) and energy (estimate: ~0.8–1 J per window). The estimates can be off
  by about 2×.
- The model was trained on clips centred on calls; here fixed 3 s windows are used, so accuracy in the field may be lower.
  There is no false-positive mitigation yet.

## License

MIT, see [LICENSE](LICENSE). The library in `vendor/audiomoth` is by Open Acoustic Devices.
