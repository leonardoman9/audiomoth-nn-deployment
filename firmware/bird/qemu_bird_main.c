// Cortex-M4F check of bird_tail.c against the PyTorch reference, under QEMU.
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "bird_tail.h"
#include "tail_dims.h"

extern const float g_feats[], g_ref[];

static int Semihost(int op, const void* arg) {
  register int r0 __asm__("r0") = op;
  register const void* r1 __asm__("r1") = arg;
  __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}
int _write(int fd, const char* data, int len) {
  (void)fd;
  for (int i = 0; i < len; ++i) Semihost(0x03, &data[i]);
  return len;
}
void _init(void) {}
void _fini(void) {}

int main(void) {
  BirdTail t;
  float got[BIRD_CLASSES];
  BirdTail_Reset(&t);
  for (int i = 0; i < BIRD_TAIL_FRAMES; ++i) BirdTail_Step(&t, g_feats + i * BIRD_FEATURES);
  BirdTail_Finish(&t, got);
  float worst = 0;
  int a = 0, b = 0;
  for (int i = 0; i < BIRD_CLASSES; ++i) {
    const float d = fabsf(got[i] - g_ref[i]);
    if (d > worst) worst = d;
    if (got[i] > got[a]) a = i;
    if (g_ref[i] > g_ref[b]) b = i;
  }
  printf("max_diff_x1e6=%d argmax_C=%d argmax_ref=%d\n", (int)(worst * 1e6f), a, b);
  const uint32_t params[2] = {0x20026, (worst < 0.05f && a == b) ? 0u : 1u};
  Semihost(0x20, params);
  for (;;) {}
}
