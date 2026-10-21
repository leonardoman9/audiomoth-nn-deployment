#ifndef SELFTEST_H
#define SELFTEST_H
#include <stdbool.h>
#include <stdint.h>
typedef struct {
  uint32_t init_ok, mismatches;
  uint32_t cycles_init, cycles_backbone, cycles_stream_18;
  uint32_t arena_backbone, arena_streaming;
} SelfTestResult;
// Backbone on a known block + 18 streaming steps, compared byte for byte with the
// TFLite reference. Cycle counts use the DWT counter (zero where unavailable).
void RunSelfTest(SelfTestResult* result);
#endif
