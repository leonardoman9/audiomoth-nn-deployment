// Real TFLM inference for the two deployed INT8 models (no simulated math).
// Input contract of the backbone: int8 [18 frames][40 mel bins]. The audio ->
// spectrogram front-end is NOT implemented here: its exact definition has not
// been recovered yet (see docs/deployment/README.md in WrenNet).
#ifndef NN_REAL_H
#define NN_REAL_H
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define NNR_FRAMES   18
#define NNR_MELS     40
#define NNR_FEATURES 32
#define NNR_CLASSES  5

// Enables the external SRAM, then loads both models into their arenas.
bool NNReal_Init(void);
// Zero real state (encoded with the state input zero-point).
void NNReal_ResetState(void);
// Backbone over one [18][40] block. Writes [18][32] features already converted
// to the quantization expected by the streaming model's feature input.
bool NNReal_Backbone(const int8_t* mel, int8_t* features);
// One streaming step; hidden state is carried (requantized) between calls.
bool NNReal_StreamStep(const int8_t* feature, int8_t* hidden_out, int8_t* logits_out);
uint32_t NNReal_BackboneArenaUsed(void);
uint32_t NNReal_StreamingArenaUsed(void);

#ifdef __cplusplus
}
#endif
#endif
