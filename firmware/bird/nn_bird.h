// Bird classifier: int8 backbone on TFLM + float tail (GRU, attention, classifier) in C.
#ifndef NN_BIRD_H
#define NN_BIRD_H
#include <stdbool.h>
#include <stdint.h>

#include "model_config.h"
#ifdef __cplusplus
extern "C" {
#endif

#define NNB_FRAMES 301
#define NNB_MELS   64
#define NNB_CLASSES MODEL_CLASSES

// Loads the backbone into its arena (external SRAM on the AudioMoth).
bool NNBird_Init(void);
// Input quantization of the backbone (spectrogram -> int8).
float NNBird_InputScale(void);
int NNBird_InputZeroPoint(void);
// spec: int8 [NNB_FRAMES][NNB_MELS], standardised dB spectrogram, time-major.
// backbone_out (optional): receives the raw int8 [NNB_FRAMES][32] backbone output.
bool NNBird_Classify(const int8_t* spec, int8_t* backbone_out, float* logits);
uint32_t NNBird_ArenaUsed(void);

#ifdef __cplusplus
}
#endif
#endif
