# First test on a real AudioMoth

The firmware has only run in emulation. This checklist tests the parts emulation cannot cover, one at a time, so a
failure points to one cause. Tools are in `tools/board/`; the simulator (`tests/sim/build_sim.sh`) is needed for step 3.

## What to have
- AudioMoth **Dev** (with external SRAM and a microSD slot), batteries or a bench supply at ~4.5 V.
- microSD card formatted **FAT32** (not exFAT).
- A way to flash: AudioMoth Flash App (bootloader) or J-Link/SWD.
- For step 5: a current meter (USB power meter, or a 1 Ω shunt + oscilloscope; `tools/energy_parser.py` reads scope CSVs).
- Optional: a sound level meter or calibrator to check the level estimate in step 2.

## LED codes

| when | LED | meaning |
|---|---|---|
| at boot, then sleeps 60 s | 3 red | no clock: no SD card, or no valid `CONFIG.TXT` (also after a power loss) |
| at boot | 3 green | `REPLAY.PCM` classified and logged |
| at boot | 6 red | `REPLAY.PCM` malformed or classification failed (the file is truncated to 1 byte either way) |
| session start | 5 red | external SRAM or model initialisation failed |
| session start | 4 red | SD card does not mount |
| during the session | 1 green | confident bird (≥ 50 %, not non-bird) |
| during the session | 1 red | window lost (audio overrun or timeout) |
| during the session | 2 red | `LOG.TXT` could not be written |
| during the session | 3 red | classification failed |
| diagnostic firmware | 1 green / 2 red per file | WAV written / WAV failed or overrun |
| diagnostic firmware | 4 red | SRAM or SD failed |

## 1. Boot and SD card (diagnostic firmware)
1. `EXTRA_CF=-DBIRD_DIAG_RECORD firmware/app/build_app.sh third_party/tflite-micro firmware/bird/model/bird_backbone_int8.tflite`
2. Flash `firmware/app/out/bird_logger.bin`. In the root of the card, put `CONFIG.TXT` with the current time
   (`2026-10-21 06:58:30`; format and rules in the README, section "CONFIG.TXT").
3. Power on. Expected: 3 short green blinks over ~10 s, one per file. Three files `HHMMSS.WAV` appear on the card,
   and `CONFIG.TXT` now reads `APPLIED ...`.

This proves the boot, the clock setting, the external SRAM (the WAV buffer is there), the ADC/DMA chain and the SD card.

## 2. Microphone levels
`python3 tools/board/analyze_diag_wav.py /path/to/card/*.WAV`

Record once in a quiet room and once with speech or a phone playing bird song at ~1 m. Look for:
- no WARN lines (clipping, dropouts, DC offset, almost-zero signal, energy dominated by < 300 Hz);
- a quiet-room noise floor well below the speech level;
- an estimated level that matches the situation (speech at 1 m ≈ 60 dB SPL). The estimate uses datasheet values and can
  be off by a few dB.

Listen to the files too: clicks or gaps point to DMA problems; a hum points to the power supply.

## 3. Same result as the computer (replay)
1. Build the normal firmware (`firmware/app/build_app.sh ...`) and the simulator (`tests/sim/build_sim.sh ...`).
2. `python3 tools/board/make_replay.py tests/sim/out/bird_logger_sim /path/to/card/HHMMSS.WAV /tmp/replay`
3. Copy `/tmp/replay/REPLAY.PCM` to the card. Flash the normal firmware and power on outside 07:00–07:05.
   Expected: 3 green blinks; `LOG.TXT` gets a `REPLAY,...` line.
4. `python3 tools/board/compare_replay.py /path/to/card/LOG.TXT /tmp/replay/expected.txt`, which must print PASS.

This proves the front-end, the int8 backbone and the tail compute on the chip what they compute on the computer.

## 4. Full session
Put `CONFIG.TXT` with a time a couple of minutes before 07:00 (or change `SCHED_WINDOW_*` in `firmware/app/schedule.h`
for a test at another hour) and leave the board running past 07:05.

`python3 tools/board/log_timing.py /path/to/card/LOG.TXT` gives the real classification time per window. The QEMU
estimate is 6–8 s, i.e. lines ~9–11 s apart and ~27–31 lines per session. If it is much slower, the external SRAM
timing is the first suspect (see docs/VERIFICATION.md, section 3).

## 5. Power and energy
Measure the current during the session (expected ~21 mA while computing, estimate) and while sleeping. Energy per
window = average session current × supply voltage × line spacing. Compare with the estimate of ~0.8–1 J per window.

## Afterwards
- Leave the board running for a few days and check `LOG.TXT` with
  `python3 tests/sim/check_long_run.py <folder containing sd/> FIRST_DATE DAYS MIN MAX`, with MIN–MAX around the line
  count measured in step 4 (for example 25 35).
- Clock drift: compare the board's time with a reference after some days (datasheet: ±0.9 s/day).
- Record results in docs/VERIFICATION.md.
