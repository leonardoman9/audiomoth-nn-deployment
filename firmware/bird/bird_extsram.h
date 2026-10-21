// Placement of the big buffers (tensor arena, front-end tables, spectrogram) in the external SRAM.
#ifndef BIRD_EXTSRAM_H
#define BIRD_EXTSRAM_H
#if defined(__arm__)
#define BIRD_EXTSRAM __attribute__((section(".extsram"), aligned(16)))
#else
#define BIRD_EXTSRAM __attribute__((aligned(16)))  /* host builds / simulator */
#endif
#endif
