// Tail of the WrenNet bird model, hand-written C: GRU(32->64), projection,
// temporal attention and classifier. The backbone stays in TFLM (int8); this part
// keeps the recurrent state in float32 so no requantization error accumulates over
// the ~300 frames of a 3 s clip. Weights are int8 with one scale per output row
// (default) or float32 (-DBIRD_TAIL_FLOAT_WEIGHTS, used as an exact reference).
#ifndef BIRD_TAIL_H
#define BIRD_TAIL_H
#include <stdint.h>

#define BIRD_FEATURES 32
#define BIRD_HIDDEN   64
#define BIRD_CLASSES  9

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float h[BIRD_HIDDEN];    // GRU state
  float ctx[BIRD_HIDDEN];  // sum_t exp(e_t - max) * p_t  (online softmax numerator)
  float max_score;
  float sum_exp;
} BirdTail;

void BirdTail_Reset(BirdTail* t);
// One backbone frame (dequantized features, length BIRD_FEATURES) in time order.
void BirdTail_Step(BirdTail* t, const float* feature);
// Softmax-weighted context -> classifier logits (length BIRD_CLASSES).
void BirdTail_Finish(const BirdTail* t, float* logits);

#ifdef __cplusplus
}
#endif
#endif
