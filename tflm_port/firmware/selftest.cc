#include "selftest.h"

#include <string.h>

#include "nn_real.h"

extern "C" {
extern const int8_t g_selftest_mel[], g_selftest_features[], g_selftest_hidden[], g_selftest_logits[];
}

static inline void CycleCounterStart(void) {
  *(volatile uint32_t*)0xE000EDFC |= (1u << 24);  // DEMCR.TRCENA
  *(volatile uint32_t*)0xE0001004 = 0;            // DWT_CYCCNT
  *(volatile uint32_t*)0xE0001000 |= 1u;          // DWT_CTRL.CYCCNTENA
}
static inline uint32_t Cycles(void) { return *(volatile uint32_t*)0xE0001004; }

void RunSelfTest(SelfTestResult* r) {
  memset(r, 0, sizeof(*r));
  CycleCounterStart();
  uint32_t t0 = Cycles();
  r->init_ok = NNReal_Init();
  r->cycles_init = Cycles() - t0;
  if (!r->init_ok) { r->mismatches = 1; return; }
  r->arena_backbone = NNReal_BackboneArenaUsed();
  r->arena_streaming = NNReal_StreamingArenaUsed();

  static int8_t features[NNR_FRAMES * NNR_FEATURES];
  t0 = Cycles();
  if (!NNReal_Backbone(g_selftest_mel, features)) r->mismatches += 1000;
  r->cycles_backbone = Cycles() - t0;
  r->mismatches += memcmp(features, g_selftest_features, sizeof(features)) != 0;

  NNReal_ResetState();
  int8_t hidden[NNR_FEATURES], logits[NNR_CLASSES];
  for (int f = 0; f < NNR_FRAMES; ++f) {
    t0 = Cycles();
    const bool step = NNReal_StreamStep(features + f * NNR_FEATURES, hidden, logits);
    r->cycles_stream_18 += Cycles() - t0;
    if (!step) { r->mismatches += 1000; break; }
    if (memcmp(hidden, g_selftest_hidden + f * NNR_FEATURES, NNR_FEATURES) != 0 ||
        memcmp(logits, g_selftest_logits + f * NNR_CLASSES, NNR_CLASSES) != 0) ++r->mismatches;
  }
}
