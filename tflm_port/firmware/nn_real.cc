#include "nn_real.h"

#include <math.h>
#include <new>
#include <string.h>

extern "C" {
#include "audiomoth.h"
}
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

extern "C" {
extern const unsigned char g_backbone_model[];
extern const unsigned char g_streaming_model[];
}

// Partitions of the external SRAM, as in the thesis: 40 KiB / 20 KiB.
// The backbone needs ~29.7 KB at peak (AllocateTensors), 5.4 KB once running.
#define BACKBONE_ARENA_BYTES  (40 * 1024)
#define STREAMING_ARENA_BYTES (20 * 1024)
__attribute__((section(".extsram"), aligned(16))) static uint8_t g_backbone_arena[BACKBONE_ARENA_BYTES];
__attribute__((section(".extsram"), aligned(16))) static uint8_t g_streaming_arena[STREAMING_ARENA_BYTES];

namespace {
using Resolver = tflite::MicroMutableOpResolver<12>;
alignas(16) uint8_t g_storage_backbone[sizeof(tflite::MicroInterpreter)];
alignas(16) uint8_t g_storage_streaming[sizeof(tflite::MicroInterpreter)];
Resolver g_resolver;
tflite::MicroInterpreter* g_backbone = nullptr;
tflite::MicroInterpreter* g_streaming = nullptr;
int8_t g_state[NNR_FEATURES];

bool Create(const unsigned char* data, uint8_t* arena, size_t arena_bytes, uint8_t* storage,
            tflite::MicroInterpreter** out) {
  const tflite::Model* model = tflite::GetModel(data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;
  auto* interpreter = new (storage) tflite::MicroInterpreter(model, g_resolver, arena, arena_bytes);
  if (interpreter->AllocateTensors() != kTfLiteOk) return false;
  *out = interpreter;
  return true;
}
}  // namespace

extern "C" {

bool NNReal_Init(void) {
  if (g_resolver.AddAdd() || g_resolver.AddConv2D() || g_resolver.AddFullyConnected() ||
      g_resolver.AddLogistic() || g_resolver.AddMean() || g_resolver.AddMul() || g_resolver.AddPad() ||
      g_resolver.AddReshape() || g_resolver.AddSoftmax() || g_resolver.AddSub() || g_resolver.AddSum() ||
      g_resolver.AddTanh()) return false;
  if (!AudioMoth_enableExternalSRAM()) return false;
  if (!Create(g_backbone_model, g_backbone_arena, sizeof(g_backbone_arena), g_storage_backbone, &g_backbone))
    return false;
  if (!Create(g_streaming_model, g_streaming_arena, sizeof(g_streaming_arena), g_storage_streaming, &g_streaming))
    return false;
  NNReal_ResetState();
  return true;
}

void NNReal_ResetState(void) {
  memset(g_state, g_streaming->input(1)->params.zero_point, sizeof(g_state));
}

bool NNReal_Backbone(const int8_t* mel, int8_t* features) {
  memcpy(g_backbone->input(0)->data.int8, mel, NNR_FRAMES * NNR_MELS);
  if (g_backbone->Invoke() != kTfLiteOk) return false;
  // Backbone output and streaming feature input have different scales and
  // zero-points: convert, as the reference does between the two models.
  const TfLiteTensor* from = g_backbone->output(0);
  const TfLiteTensor* to = g_streaming->input(0);
  for (int i = 0; i < NNR_FRAMES * NNR_FEATURES; ++i) {
    const double real = (from->data.int8[i] - from->params.zero_point) * (double)from->params.scale;
    long q = lrint(real / (double)to->params.scale) + to->params.zero_point;
    features[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
  return true;
}

bool NNReal_StreamStep(const int8_t* feature, int8_t* hidden_out, int8_t* logits_out) {
  TfLiteTensor* state = g_streaming->input(1);
  TfLiteTensor* hidden = g_streaming->output(0);  // output order is fixed by the model
  TfLiteTensor* logits = g_streaming->output(1);
  memcpy(g_streaming->input(0)->data.int8, feature, NNR_FEATURES);
  memcpy(state->data.int8, g_state, NNR_FEATURES);
  if (g_streaming->Invoke() != kTfLiteOk) return false;
  if (hidden_out) memcpy(hidden_out, hidden->data.int8, NNR_FEATURES);
  memcpy(logits_out, logits->data.int8, NNR_CLASSES);
  // The state output and input use different scales: requantize, never copy.
  for (int i = 0; i < NNR_FEATURES; ++i) {
    const double real = (hidden->data.int8[i] - hidden->params.zero_point) * (double)hidden->params.scale;
    long q = lrint(real / (double)state->params.scale) + state->params.zero_point;
    g_state[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
  return true;
}

uint32_t NNReal_BackboneArenaUsed(void) { return g_backbone ? g_backbone->arena_used_bytes() : 0; }
uint32_t NNReal_StreamingArenaUsed(void) { return g_streaming ? g_streaming->arena_used_bytes() : 0; }

}  // extern "C"
