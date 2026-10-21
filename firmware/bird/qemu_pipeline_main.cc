// Full pipeline on an emulated Cortex-M4F: raw 16-bit PCM in 512-sample blocks -> logits.
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "bird_marker.h"
#include "bird_pipeline.h"
#ifdef BIRD_MARKERS
#endif

extern "C" {
extern const int16_t g_pcm[];
extern const float g_host_logits[], g_torch_logits[];
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

extern "C" uint32_t _ebss, _estack;

int main() {
  const bool ok = BirdPipeline_Init();
  BIRD_MARK(0);  /* start-up + init done */
  float logits[NNB_CLASSES] = {0};
  bool run = false;
  if (ok) {
    BirdPipeline_Begin();
    for (uint32_t off = 0; off < 96000; off += 512) BirdPipeline_Push(g_pcm + off, 512);
    BIRD_MARK(1);  /* front-end frames from streamed PCM done */
    run = BirdPipeline_Ready() && BirdPipeline_Finish(logits);
  }
  /* Stack high-water mark BEFORE any printf (newlib's printf mallocs from the painted region). */
  const uint32_t* painted = &_ebss;
  while (painted < &_estack && *painted == 0xDEADBEEF) ++painted;
  const unsigned stack_used = (unsigned)((uintptr_t)&_estack - (uintptr_t)painted);
  const unsigned static_end = (unsigned)((uintptr_t)&_ebss - 0x20000000u);
  float d_host = 0, d_torch = 0;
  int a = 0, b = 0, c = 0;
  for (int i = 0; i < NNB_CLASSES; ++i) {
    d_host = fmaxf(d_host, fabsf(logits[i] - g_host_logits[i]));
    d_torch = fmaxf(d_torch, fabsf(logits[i] - g_torch_logits[i]));
    if (logits[i] > logits[a]) a = i;
    if (g_host_logits[i] > g_host_logits[b]) b = i;
    if (g_torch_logits[i] > g_torch_logits[c]) c = i;
  }
  printf("run=%d max|d| vs host pipeline=%d/1000 vs PyTorch=%d/1000 | argmax device=%d host=%d torch=%d\n", (int)run,
         (int)(d_host * 1000), (int)(d_torch * 1000), a, b, c);
  printf("init=%d internal RAM: static_end=%u stack_used_max=%u of %u\n", (int)ok, static_end, stack_used,
         (unsigned)((uintptr_t)&_estack - 0x20000000u));
  const bool pass = run && d_host < 0.05f && a == b;
  printf("%s\n", pass ? "PASS" : "FAIL");
  const uint32_t params[2] = {0x20026, pass ? 0u : 1u};
  Semihost(0x20, params);
  for (;;) {}
}
