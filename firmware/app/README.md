# "Bird logger" application for AudioMoth

Every day at **07:00** (device clock), for **5 minutes**, it records the whole window at 32 kHz to the SD card
(`YYYYMMDD/HHMMSS.WAV`). Then it reads the recording back in 3 s windows, classifies each with the model and appends
one line per window to `LOG.TXT`, with the time of the window in the recording:

    2026-10-21 07:00:03,Certhia brachydactyla,55,Y,...     (date time, class, probability %, Y = >= 50 %, 9 logits x100)

How the recording works:
- **Buffering:** the DMA handler fills a 1.5 s ring in the external SRAM, and the main loop writes it to the card in
  4 KB chunks, syncing every ~2.6 s.
- **SD problems:** if the card falls more than the ring behind, or a write fails (busy for more than 500 ms, card
  full), the lost stretch is written as silence and counted, so the file keeps the real timeline. A failed write
  reopens the file; the windows classified are only those really in the file.
- **Interruptions:** progress is kept in `STATE.TXT` (a 96-byte record rewritten in place, never truncated). A reset
  or power cut during the recording or the classification is resumed at the next boot, even without a clock (the
  library clears it on any reset other than a wake-up from EM4). A window that fails 3 boots in a row is skipped.
  `SESSIONS.TXT` gets one summary line per session.
- **Timing:** the board wakes 2 s before the window and the recording starts exactly on a whole second, so the file
  name and every `LOG.TXT` time are the times of the samples. The WAV header is padded to 512 bytes so the 4 KB
  writes are sector-aligned.

Outside the window and the classification the device is powered down (power-down, wake-ups every 6 h).

The clock is set with `CONFIG.TXT` on the SD card (one line `YYYY-MM-DD hh:mm:ss`), which is then rewritten as
`APPLIED <time>`. No USB, GPS or configuration app. Start time, duration, hop and threshold come from the model package
(`model_config.h`). The class order is the model's OUTPUT INDEX
order (the order of `allowed_bird_classes` in the training configuration), not the order of the "Class names" line in
the training log: see `docs/VERIFICATION.md` (`kClassNames` in `schedule.c`).

Files: `main_app.c` (hardware glue, follows `makeRecording` of the original firmware), `schedule.c` (pure logic, checked
on 3,228 cases against Python's `datetime`, see `tests/schedule/`), `fatfs_conf/` (trimmed FatFS), `build_app.sh`
(firmware). The host simulator is in `tests/sim/`.

## Firmware (measured, GCC 14.3.1 `-Os`)

    firmware/app/build_app.sh TFLM_ROOT bird_backbone_int8.tflite     # TFLM from scripts/build_tflm.sh

- Flash: **233,476 B** of 245,760 available (~12 KB margin; see `docs/VERIFICATION.md`). Model 94.7 KB, TFLM+CMSIS-NN
  ~57 KB, C tail 27.6 KB, libc/libm ~23 KB, FatFS+SD 7.4 KB, emlib 6.7 KB, audiomoth.c 5 KB.
- Trimmed FatFS (CP437, 8.3 names, **no exFAT**: use a FAT32 card): saves ~60 KB of tables.
  The complete original AudioMoth firmware (USB, GPS, filters, standard FatFS) takes ~150 KB.
- Internal RAM: .data 192 B + .bss 16,160 B (+ 3 KB heap, stack at the end). External SRAM: 258,512 B of 262,144
  (the 96 KB recording ring included).

## Simulator (`tests/sim/`)

Runs `main_app.c` and the real pipeline (front-end, TFLM backbone, tail, `digitalfilter.c`) on a software model of the
hardware (`hal_sim.c`). RTC and SD persist across boots (each process = one power cycle); the microphone reads an audio
file (384 kHz samples reconstructed for the real decimation filter).

    tests/sim/build_sim.sh TFLM_ROOT bird_backbone_int8.tflite
    python tests/sim/make_audio.py audio.raw <audio files or folders>
    python tests/sim/run_scenario.py tests/sim/out/bird_logger_sim audio.raw WORKDIR 2
    python tests/sim/verify_log.py WRENNET_ROOT best_distillation_model.pt WORKDIR

Result (2 simulated days, clock set to 06:58:30 by CONFIG.TXT): one 300 s recording per day starting exactly at
07:00:00, 100 log lines per day (07:00:00 to 07:04:57), every window equal to the matching slice of the recording,
nothing pending; for the rest of the day only wake-ups every 6 h. The fault scenarios (`run_faults.py`, 44) and the
checks are in `docs/VERIFICATION.md`, section 13.

**What it does NOT model**: ADC/DMA timing, the microphone's analogue stage, the real SD driver (its stalls and
speed are modelled only coarsely: write latency, a 500 ms timeout, sticky write errors), RTC drift, power
consumption, timing with external SRAM, the real power-down. The AudioMoth decimator sums 12 samples (block average),
a poorer anti-alias filter than the resampling used in training (about -4 dB at 16 kHz): a difference to keep in mind in
the field.
