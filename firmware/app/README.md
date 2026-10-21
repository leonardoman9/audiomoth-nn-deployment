# "Bird logger" application for AudioMoth

Every day at **07:00** (device clock), for **5 minutes**, it records 3 s windows at 32 kHz, classifies them with the bird
model and appends one line to `LOG.TXT` on the SD card:

    2026-10-21 07:00:03,Certhia brachydactyla,55,Y,...     (date time, class, probability %, Y = >= 50 %, 9 logits x100)

Outside the window the device is powered down (power-down, hourly wake-ups). The clock is set with `CONFIG.TXT` on the SD
card (one line `YYYY-MM-DD hh:mm:ss`), which is then rewritten as `APPLIED <time>`. No USB, GPS, WAV writing or
configuration app. Start time and duration: `SCHED_WINDOW_*` in `schedule.h`. The class order is the model's OUTPUT INDEX
order (the order of `allowed_bird_classes` in the training configuration), not the order of the "Class names" line in
the training log: see `docs/VERIFICATION.md` (`kClassNames` in `schedule.c`).

Files: `main_app.c` (hardware glue, follows `makeRecording` of the original firmware), `schedule.c` (pure logic, checked
on 1,805 cases against Python's `datetime`, see `tests/schedule/`), `fatfs_conf/` (trimmed FatFS), `build_app.sh`
(firmware). The host simulator is in `tests/sim/`.

## Firmware (measured, GCC 14.3.1 `-Os`)

    firmware/app/build_app.sh TFLM_ROOT bird_backbone_int8.tflite     # TFLM from scripts/build_tflm.sh

- Flash: **230,124 B** of 245,760 available (~15.6 KB margin; see `docs/VERIFICATION.md`). Model 94.7 KB, TFLM+CMSIS-NN
  ~57 KB, C tail 27.6 KB, libc/libm ~23 KB, FatFS+SD 7.4 KB, emlib 6.7 KB, audiomoth.c 5 KB.
- Trimmed FatFS (CP437, 8.3 names, **no exFAT**: use a FAT32 card): saves ~60 KB of tables.
  The complete original AudioMoth firmware (USB, GPS, filters, standard FatFS) takes ~150 KB.
- Internal RAM: .data 192 B + .bss 14,112 B (+ 3 KB heap, stack at the end). External SRAM: 160,208 B of 262,144.

## Simulator (`tests/sim/`)

Runs `main_app.c` and the real pipeline (front-end, TFLM backbone, tail, `digitalfilter.c`) on a software model of the
hardware (`hal_sim.c`). RTC and SD persist across boots (each process = one power cycle); the microphone reads an audio
file (384 kHz samples reconstructed for the real decimation filter).

    tests/sim/build_sim.sh TFLM_ROOT bird_backbone_int8.tflite
    python tests/sim/make_audio.py audio.raw <audio files or folders>
    python tests/sim/run_scenario.py tests/sim/out/bird_logger_sim audio.raw WORKDIR 2
    python tests/sim/verify_log.py WRENNET_ROOT best_distillation_model.pt WORKDIR

Result (2 simulated days, clock set to 06:58:30 by CONFIG.TXT): 28 boots, 49 log lines per day (one every 6 s, from
07:00:03 to 07:04:55), no errors; for the rest of the day only hourly wake-ups. On the same 49 windows PyTorch agrees on
the class in 48 (the flipped one has confidence 0.28); class names and percentages in the log lines match the device logits.

**What it does NOT model**: ADC/DMA timing, the microphone's analogue stage, the real SD driver, RTC drift, power
consumption, timing with external SRAM, the real power-down. The AudioMoth decimator sums 12 samples (block average),
a poorer anti-alias filter than the resampling used in training (about -4 dB at 16 kHz): a difference to keep in mind in
the field.
