# TFLM + CMSIS-NN for the AudioMoth (Cortex-M4F)

## Building TFLM

```sh
scripts/setup_tflm.sh    # tflite-micro at commit e0021ca434cd930acbab69768e88d959083a3d4b + firmware/patches/tflm-no-int4.patch
scripts/build_tflm.sh    # builds the two libraries below
```

`build_tflm.sh` runs, for `BUILD_TYPE=release` and `BUILD_TYPE=release_with_logs`:

```sh
gmake -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp \
  OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=<type> KERNEL_OPTIMIZATION_LEVEL=-Os \
  THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL=-Os OPTIMIZE_KERNELS_FOR=KERNELS_OPTIMIZED_FOR_SIZE microlite
```

- `release` (error strings stripped) is used by `firmware/app/build_app.sh`.
- `release_with_logs` is used by the QEMU tests (`firmware/bird/build_bird_qemu.sh`).
- The size flags are required: with the default `-O2` kernels the firmware grows to 244,620 B, against a limit of 245,760 B.
- The patch removes the int4 CMSIS-NN kernels, which the model does not use, to save flash.
- TFLM downloads its own toolchain (GCC 14.3.1). GNU Arm 10.3-2021.10 gives the same size (230,024 B vs 230,124 B,
  see VERIFICATION.md).

## How the port is split

TFLM has no GRU operator, so the model runs in three parts:

1. **Front-end in C** (`firmware/bird/bird_frontend.c`): streaming STFT (n_fft 1024, hop 320), 64 learned filters, dB
   with top_db 80, per-clip standardisation. Constant tables are in flash; FFT buffers are in internal RAM.
2. **int8 CNN backbone on TFLM + CMSIS-NN** (`nn_bird.cc`): input `[1,1,301,64]`, output `[1,1,301,32]`, 151 operators,
   8-operator resolver, 96 KiB arena in external SRAM.
3. **Tail in C** (`bird_tail.c`): GRU 64, projection, temporal attention, FC; int8 weights with per-row scales,
   float computation.

## Notes

- `AllocateTensors` needs more arena than the steady state: the peak is in the memory-planning phase and scales with
  the number of tensors. Rewriting the backbone as a 4-D graph without RESHAPE operators (151 operators) reduced it.
- Latency and energy have to be measured on hardware with this port; the figures so far are estimates from QEMU
  instruction and memory-access counts.
