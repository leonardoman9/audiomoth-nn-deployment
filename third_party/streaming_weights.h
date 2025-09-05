#ifndef STREAMING_WEIGHTS_H
#define STREAMING_WEIGHTS_H

#include <stdint.h>
#include <stddef.h>

// ================================================================
// STREAMING MODEL STRUCTURE (REAL FROM TFLITE)
// ================================================================

// Input 0: serving_default_feature_frame:0
#define STREAMING_INPUT_0_SIZE 32
#define STREAMING_INPUT_0_TOTAL 32

// Input 1: serving_default_hidden_state:0
#define STREAMING_INPUT_1_SIZE 32
#define STREAMING_INPUT_1_TOTAL 32

// Output 0: PartitionedCall:1
#define STREAMING_OUTPUT_0_SIZE 32

// Output 1: PartitionedCall:0
#define STREAMING_OUTPUT_1_SIZE 5

// Weight arrays (extern declarations)
extern const int8_t streaming_serving_default_feature_frame_0[32];
extern const int8_t streaming_serving_default_hidden_state_0[32];
extern const float streaming_arith_constant[3];
extern const float streaming_tfl_pseudo_qconst[5];
extern const int8_t streaming_tfl_pseudo_qconst1[160];
extern const int8_t streaming_tfl_pseudo_qconst3[32];
extern const float streaming_tfl_pseudo_qconst4[32];
extern const int8_t streaming_tfl_pseudo_qconst5[1024];
extern const float streaming_tfl_pseudo_qconst6[32];
extern const int8_t streaming_tfl_pseudo_qconst7[1024];
extern const float streaming_tfl_pseudo_qconst8[32];
extern const int8_t streaming_tfl_pseudo_qconst9[1024];
extern const float streaming_tfl_pseudo_qconst10[32];
extern const int8_t streaming_tfl_pseudo_qconst11[1024];
extern const float streaming_tfl_pseudo_qconst13[32];
extern const int8_t streaming_tfl_pseudo_qconst14[1024];
extern const float streaming_tfl_pseudo_qconst15[32];
extern const int8_t streaming_tfl_pseudo_qconst16[1024];
extern const float streaming_tfl_pseudo_qconst17[32];
extern const int8_t streaming_tfl_pseudo_qconst18[1024];
extern const int8_t streaming_PartitionedCall_1[32];
extern const int8_t streaming_model_tf_reshape_Reshape[32];
extern const int8_t streaming_PartitionedCall_0[5];

#endif // STREAMING_WEIGHTS_H
