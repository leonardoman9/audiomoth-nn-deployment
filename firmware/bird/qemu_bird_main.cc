// Runs nn_bird.cc (TFLM backbone + C tail) on an emulated Cortex-M4F and checks the result.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bird_tail.h"
#include "nn_bird.h"

extern "C" {
extern const int8_t g_bird_in[], g_bird_out[];
extern const float g_bird_logits[];
}

static int Semihost(int op, const void* arg) {
  register int r0 __asm__("r0") = op;
  register const void* r1 __asm__("r1") = arg;
  __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}
extern "C" int _write(int, const char* data, int len) {
  for (int i = 0; i < len; ++i) Semihost(0x03, &data[i]);
  return len;
}
extern "C" void _init() {}
extern "C" void _fini() {}
void* __dso_handle;

// SYS_ELAPSED: with `qemu -icount shift=0` the virtual clock advances 1 ns per executed
// instruction, so tick differences are instruction counts (a lower bound for cycles).
static uint64_t Ticks() {
  uint64_t t = 0;
  Semihost(0x30, &t);
  return t;
}

int main() {
  const bool ok = NNBird_Init();
  printf("init=%d arena_used=%u\n", (int)ok, (unsigned)NNBird_ArenaUsed());
  static int8_t bb[NNB_FRAMES * 32];
  float logits[NNB_CLASSES];
  const uint64_t t0 = Ticks();
  bool run = ok && NNBird_Classify(g_bird_in, bb, logits);
  const uint64_t t1 = Ticks();
  // Tail alone, fed with the dequantised backbone output of the reference.
  BirdTail tail;
  BirdTail_Reset(&tail);
  float frame[BIRD_FEATURES];
  const uint64_t t2 = Ticks();
  for (int t = 0; t < NNB_FRAMES; ++t) {
    for (int c = 0; c < BIRD_FEATURES; ++c) frame[c] = (float)(g_bird_out[t * BIRD_FEATURES + c] + 128) * 0.1f;
    BirdTail_Step(&tail, frame);
  }
  const uint64_t t3 = Ticks();
  printf("instructions(ns at shift=0): classify=%u tail_only=%u\n", (unsigned)(t1 - t0), (unsigned)(t3 - t2));
  int bad_bytes = 0;
  float worst = 0;
  if (run) {
    for (int i = 0; i < NNB_FRAMES * 32; ++i) bad_bytes += bb[i] != g_bird_out[i];
    for (int i = 0; i < NNB_CLASSES; ++i) worst = fmaxf(worst, fabsf(logits[i] - g_bird_logits[i]));
  }
  printf("run=%d backbone_bytes_different=%d max_logit_diff_x1e6=%d\n", (int)run, bad_bytes, (int)(worst * 1e6f));
  const bool pass = run && bad_bytes == 0 && worst < 1e-3f;
  printf("%s\n", pass ? "PASS" : "FAIL");
  const uint32_t params[2] = {0x20026, pass ? 0u : 1u};
  Semihost(0x20, params);
  for (;;) {}
}
