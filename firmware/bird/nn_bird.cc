#include "nn_bird.h"
#include "bird_extsram.h"

#include <new>
#include <string.h>

extern "C" {
#include "audiomoth.h"
}
#include "bird_marker.h"
#include "bird_tail.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

extern "C" const unsigned char g_bird_model[];

#ifndef BIRD_ARENA_BYTES
#define BIRD_ARENA_BYTES (160 * 1024)
#endif
BIRD_EXTSRAM static uint8_t g_arena[BIRD_ARENA_BYTES];

namespace {
alignas(16) uint8_t g_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroMutableOpResolver<8> g_resolver;
tflite::MicroInterpreter* g_interp = nullptr;
}  // namespace

extern "C" {

bool NNBird_Init(void) {
  // Exactly the operators of the 4-D backbone (no Reshape/Minimum/Relu any more). Registered once: adding an
  // operator twice is an error, and Init runs more than once per boot (replay, resumed recording, session).
  static bool ops_added = false;
  if (!ops_added) {
    if (g_resolver.AddAdd() || g_resolver.AddConv2D() || g_resolver.AddDepthwiseConv2D() || g_resolver.AddHardSwish() ||
        g_resolver.AddLogistic() || g_resolver.AddMean() || g_resolver.AddMul() || g_resolver.AddPad()) return false;
    ops_added = true;
  }
  // The interpreter is rebuilt every time: its state lives in the arena, which is lost whenever the external SRAM
  // is powered off between uses. (No destructor call: it would read that state; the interpreter owns no heap memory.)
  if (!AudioMoth_enableExternalSRAM()) return false;
  const tflite::Model* model = tflite::GetModel(g_bird_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;
  g_interp = new (g_storage) tflite::MicroInterpreter(model, g_resolver, g_arena, sizeof(g_arena));
  return g_interp->AllocateTensors() == kTfLiteOk;
}

float NNBird_InputScale(void) { return g_interp->input(0)->params.scale; }
int NNBird_InputZeroPoint(void) { return g_interp->input(0)->params.zero_point; }
uint32_t NNBird_ArenaUsed(void) { return g_interp ? (uint32_t)g_interp->arena_used_bytes() : 0; }

bool NNBird_Classify(const int8_t* spec, int8_t* backbone_out, float* logits) {
  memcpy(g_interp->input(0)->data.int8, spec, NNB_FRAMES * NNB_MELS);
  if (g_interp->Invoke() != kTfLiteOk) return false;
  BIRD_MARK(3);  /* end of backbone */
  const TfLiteTensor* out = g_interp->output(0);
  if (backbone_out) memcpy(backbone_out, out->data.int8, NNB_FRAMES * BIRD_FEATURES);
  const float scale = out->params.scale;
  const int zero = out->params.zero_point;
  BirdTail tail;
  BirdTail_Reset(&tail);
  float frame[BIRD_FEATURES];
  for (int t = 0; t < NNB_FRAMES; ++t) {
    const int8_t* row = out->data.int8 + t * BIRD_FEATURES;
    for (int c = 0; c < BIRD_FEATURES; ++c) frame[c] = (float)(row[c] - zero) * scale;
    BirdTail_Step(&tail, frame);
  }
  BirdTail_Finish(&tail, logits);
  BIRD_MARK(4);  /* end of tail */
  return true;
}

}  // extern "C"
