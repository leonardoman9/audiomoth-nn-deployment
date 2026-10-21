// Runs the production self-test (selftest.cc + nn_real.cc) on an emulated Cortex-M4.
#include <stdint.h>
#include <stdio.h>

#include "selftest.h"

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

int main() {
  SelfTestResult r;
  RunSelfTest(&r);
  printf("init_ok=%u arena_backbone=%u arena_streaming=%u\n", (unsigned)r.init_ok,
         (unsigned)r.arena_backbone, (unsigned)r.arena_streaming);
  printf("%s: %u mismatches\n", (r.init_ok && !r.mismatches) ? "PASS" : "FAIL", (unsigned)r.mismatches);
  const uint32_t params[2] = {0x20026, (r.init_ok && !r.mismatches) ? 0u : 1u};
  Semihost(0x20, params);
  for (;;) {}
}
