// Whole bird classifier: PCM in, 9 logits out.
//   Push 16-bit PCM @ 32 kHz as it arrives -> BirdFE (C) -> int8 backbone (TFLM) -> tail (C).
#ifndef BIRD_PIPELINE_H
#define BIRD_PIPELINE_H
#include <stdbool.h>
#include <stdint.h>
#include "nn_bird.h"
#ifdef __cplusplus
extern "C" {
#endif
bool BirdPipeline_Init(void);                       /* external SRAM, model, filterbank */
void BirdPipeline_Begin(void);                      /* start a new 3 s clip */
void BirdPipeline_Push(const int16_t* pcm, uint32_t n);
bool BirdPipeline_Ready(void);                      /* 96000 samples received */
bool BirdPipeline_Finish(float* logits);            /* NNB_CLASSES logits */
#ifdef __cplusplus
}
#endif
#endif
