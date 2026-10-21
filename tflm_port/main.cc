// Bare-metal TFLM parity test for Cortex-M4 (run under QEMU with semihosting).
// Compares backbone, streaming hidden state and logits against the vectors
// produced by the TFLite reference, and reports arena and stack usage.
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "counts.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#ifndef ARENA_BYTES
#define ARENA_BYTES 28672
#endif

extern "C" {
extern uint32_t _ebss, _estack;
extern const unsigned char g_backbone_model[], g_streaming_model[];
extern const int8_t g_backbone_in[], g_backbone_out[], g_seq_feat[], g_seq_hidden[], g_seq_logits[];
}

// ---- semihosting -------------------------------------------------------
static int Semihost(int op, const void* arg) {
  register int r0 __asm__("r0") = op;
  register const void* r1 __asm__("r1") = arg;
  __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}
static void Print(const char* s) { Semihost(0x04, s); }
static void Printf(const char* fmt, ...) {
  char buf[160];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  Print(buf);
}
[[noreturn]] static void Exit(int code) {
  const uint32_t params[2] = {0x20026, static_cast<uint32_t>(code)};
  Semihost(0x20, params);
  for (;;) {}
}
// The TFLM library provides DebugLog(); route its libc output to the host.
extern "C" int _write(int, const char* data, int len) {
  for (int i = 0; i < len; ++i) Semihost(0x03, &data[i]);  // SYS_WRITEC
  return len;
}
extern "C" void _init() {}
extern "C" void _fini() {}
void* __dso_handle;

// ---- interpreters sharing one arena -------------------------------------
#ifdef ARENA_IN_EXTSRAM
__attribute__((section(".extsram")))
#endif
alignas(16) static uint8_t g_arena[ARENA_BYTES];
alignas(16) static uint8_t g_interp_storage[sizeof(tflite::MicroInterpreter)];
static tflite::MicroMutableOpResolver<12> g_resolver;
static tflite::MicroInterpreter* g_interp = nullptr;

static bool Load(const unsigned char* model_data, const char* name) {
  if (g_interp) g_interp->~MicroInterpreter();
  g_interp = nullptr;
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;
  g_interp = new (g_interp_storage) tflite::MicroInterpreter(model, g_resolver, g_arena, sizeof(g_arena));
  if (g_interp->AllocateTensors() != kTfLiteOk) {
    Printf("%s: AllocateTensors failed (arena %u bytes)\n", name, (unsigned)sizeof(g_arena));
    return false;
  }
  Printf("%s: arena_used=%u of %u\n", name, (unsigned)g_interp->arena_used_bytes(), (unsigned)sizeof(g_arena));
  return true;
}

static int8_t Requantize(int8_t v, const TfLiteTensor* from, const TfLiteTensor* to) {
  const double real = (static_cast<int>(v) - from->params.zero_point) * static_cast<double>(from->params.scale);
  long q = lrint(real / static_cast<double>(to->params.scale)) + to->params.zero_point;
  return static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
}

int main() {
  if (g_resolver.AddAdd() || g_resolver.AddConv2D() || g_resolver.AddFullyConnected() ||
      g_resolver.AddLogistic() || g_resolver.AddMean() || g_resolver.AddMul() || g_resolver.AddPad() ||
      g_resolver.AddReshape() || g_resolver.AddSoftmax() || g_resolver.AddSub() || g_resolver.AddSum() ||
      g_resolver.AddTanh()) {
    Print("resolver failed\n");
    Exit(2);
  }
  int mismatches = 0;

  if (!Load(g_backbone_model, "backbone")) Exit(3);
  for (int i = 0; i < N_BACKBONE; ++i) {
    memcpy(g_interp->input(0)->data.int8, g_backbone_in + i * 720, 720);
    if (g_interp->Invoke() != kTfLiteOk) { Print("backbone invoke failed\n"); Exit(4); }
    if (memcmp(g_interp->output(0)->data.int8, g_backbone_out + i * 576, 576) != 0) {
      Printf("backbone case %d MISMATCH\n", i);
      ++mismatches;
    }
  }

  if (!Load(g_streaming_model, "streaming")) Exit(3);
  TfLiteTensor* feature = g_interp->input(0);
  TfLiteTensor* state = g_interp->input(1);
  TfLiteTensor* hidden = g_interp->output(0);
  TfLiteTensor* logits = g_interp->output(1);
  int8_t carried[32];
  memset(carried, state->params.zero_point, sizeof(carried));  // real zero
  for (int i = 0; i < N_FRAMES; ++i) {
    memcpy(feature->data.int8, g_seq_feat + i * 32, 32);
    memcpy(state->data.int8, carried, 32);
    if (g_interp->Invoke() != kTfLiteOk) { Print("streaming invoke failed\n"); Exit(4); }
    if (memcmp(hidden->data.int8, g_seq_hidden + i * 32, 32) != 0 ||
        memcmp(logits->data.int8, g_seq_logits + i * 5, 5) != 0) {
      if (mismatches < 8) Printf("streaming frame %d MISMATCH\n", i);
      ++mismatches;
    }
    for (int j = 0; j < 32; ++j) carried[j] = Requantize(hidden->data.int8[j], hidden, state);
  }

  // Stack high-water mark: Reset_Handler painted [_ebss, sp) with 0xDEADBEEF.
  const uint32_t* p = &_ebss;
  while (p < &_estack && *p == 0xDEADBEEF) ++p;
  Printf("ram: static_end=%u stack_free_min=%u stack_used_max=%u (RAM 32768)\n",
         (unsigned)((uintptr_t)&_ebss - 0x20000000u), (unsigned)((uintptr_t)p - (uintptr_t)&_ebss),
         (unsigned)((uintptr_t)&_estack - (uintptr_t)p));
  Printf("%s: %d mismatches\n", mismatches ? "FAIL" : "PASS", mismatches);
  Exit(mismatches ? 1 : 0);
}
