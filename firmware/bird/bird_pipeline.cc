#include "bird_pipeline.h"
#include "bird_extsram.h"
#include "bird_marker.h"

extern "C" {
#include "bird_frontend.h"
}

BIRD_EXTSRAM static BirdFrontEnd g_fe;
BIRD_EXTSRAM static int8_t g_spec[NNB_FRAMES * NNB_MELS];
#ifdef BIRD_MARKERS
extern "C" { BIRD_EXTSRAM volatile unsigned g_bird_marker; }
#endif

bool BirdPipeline_Init(void) {
  if (!NNBird_Init()) return false;  // enables the external SRAM first
  BirdFE_Init(&g_fe);  // filterbank = constant tables generated from the checkpoint's learned parameters
  return true;
}
void BirdPipeline_Begin(void) { BirdFE_Reset(&g_fe); }
void BirdPipeline_Push(const int16_t* pcm, uint32_t n) { BirdFE_Push(&g_fe, pcm, n); }
bool BirdPipeline_Ready(void) { return BirdFE_Complete(&g_fe); }

bool BirdPipeline_Finish(float* logits) {
  if (!BirdFE_Finish(&g_fe, g_spec, NNBird_InputScale(), NNBird_InputZeroPoint())) return false;
  BIRD_MARK(2);  /* end of front-end finish */
  return NNBird_Classify(g_spec, nullptr, logits);
}
