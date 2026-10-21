# WrenNet on AudioMoth

Firmware for the AudioMoth Dev (EFM32WG380F256) that records from the microphone and runs the WrenNet bird
classifier (8 species + non-bird): streaming C front-end, int8 CNN backbone on TFLM + CMSIS-NN, float GRU/attention tail in C.
Once a day (07:00) it classifies about 5 minutes of audio and logs the results to the SD card.
Status: verified in emulation only (host simulator, QEMU Cortex-M4, FatFS on a FAT32 image); not yet run on a real board.

```
firmware/app      application: hardware glue (main_app.c), schedule logic, build_app.sh
firmware/bird     front-end, backbone wrapper, tail, generated tables, QEMU/profiling scripts
vendor/audiomoth  AudioMoth hardware library (emlib, FatFS, drivers, ...) from AudioMoth-Firmware-Basic
tests/            sim (host simulator), fstest (FatFS), schedule (1,805 cases), qemu (harness)
tools/            table generators, QEMU instruction-count plugin, energy parser
scripts/          setup_tflm.sh (fetch TFLM at the pinned commit + patch)
docs/             VERIFICATION.md (all results), TFLM_BUILD.md, datasheets
benchmarks/       Raspberry Pi benchmarks (earlier KWS-era work)
```

Build: `scripts/setup_tflm.sh`, build the TFLM library (docs/TFLM_BUILD.md), then
`firmware/app/build_app.sh TFLM_ROOT docs/.../bird_backbone_int8.tflite` (model and reference data are in the WrenNet repo, docs/deployment/bird).
The earlier simulated KWS deployment (never ran the trained network) is kept only in git tag `pre-cleanup`.
