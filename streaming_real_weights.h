/**
 * @file streaming_real_weights.h
 * @brief Real extracted weights for streaming model
 */

#ifndef STREAMING_REAL_WEIGHTS_H  
#define STREAMING_REAL_WEIGHTS_H

#include <stdint.h>

#define STREAMING_NUM_WEIGHTS 24

// serving_default_feature_frame:0
extern const int8_t streaming_serving_default_feature_frame_0[32];
extern const uint32_t streaming_serving_default_feature_frame_0_shape[2];

// serving_default_hidden_state:0
extern const int8_t streaming_serving_default_hidden_state_0[32];
extern const uint32_t streaming_serving_default_hidden_state_0_shape[2];

// tfl.pseudo_qconst
extern const int8_t streaming_tfl_pseudo_qconst[5];
extern const uint32_t streaming_tfl_pseudo_qconst_shape[1];

// tfl.pseudo_qconst1
extern const int8_t streaming_tfl_pseudo_qconst1[160];
extern const uint32_t streaming_tfl_pseudo_qconst1_shape[2];

// tfl.pseudo_qconst2
extern const int8_t streaming_tfl_pseudo_qconst2[1];
extern const uint32_t streaming_tfl_pseudo_qconst2_shape[1];

// tfl.pseudo_qconst3
extern const int8_t streaming_tfl_pseudo_qconst3[32];
extern const uint32_t streaming_tfl_pseudo_qconst3_shape[2];

// tfl.pseudo_qconst4
extern const int8_t streaming_tfl_pseudo_qconst4[32];
extern const uint32_t streaming_tfl_pseudo_qconst4_shape[1];

// tfl.pseudo_qconst5
extern const int8_t streaming_tfl_pseudo_qconst5[1024];
extern const uint32_t streaming_tfl_pseudo_qconst5_shape[2];

// tfl.pseudo_qconst6
extern const int8_t streaming_tfl_pseudo_qconst6[32];
extern const uint32_t streaming_tfl_pseudo_qconst6_shape[1];

// tfl.pseudo_qconst7
extern const int8_t streaming_tfl_pseudo_qconst7[1024];
extern const uint32_t streaming_tfl_pseudo_qconst7_shape[2];

// tfl.pseudo_qconst8
extern const int8_t streaming_tfl_pseudo_qconst8[32];
extern const uint32_t streaming_tfl_pseudo_qconst8_shape[1];

// tfl.pseudo_qconst9
extern const int8_t streaming_tfl_pseudo_qconst9[1024];
extern const uint32_t streaming_tfl_pseudo_qconst9_shape[2];

// tfl.pseudo_qconst10
extern const int8_t streaming_tfl_pseudo_qconst10[32];
extern const uint32_t streaming_tfl_pseudo_qconst10_shape[1];

// tfl.pseudo_qconst11
extern const int8_t streaming_tfl_pseudo_qconst11[1024];
extern const uint32_t streaming_tfl_pseudo_qconst11_shape[2];

// tfl.pseudo_qconst12
extern const int8_t streaming_tfl_pseudo_qconst12[1];
extern const uint32_t streaming_tfl_pseudo_qconst12_shape[1];

// tfl.pseudo_qconst13
extern const int8_t streaming_tfl_pseudo_qconst13[32];
extern const uint32_t streaming_tfl_pseudo_qconst13_shape[1];

// tfl.pseudo_qconst14
extern const int8_t streaming_tfl_pseudo_qconst14[1024];
extern const uint32_t streaming_tfl_pseudo_qconst14_shape[2];

// tfl.pseudo_qconst15
extern const int8_t streaming_tfl_pseudo_qconst15[32];
extern const uint32_t streaming_tfl_pseudo_qconst15_shape[1];

// tfl.pseudo_qconst16
extern const int8_t streaming_tfl_pseudo_qconst16[1024];
extern const uint32_t streaming_tfl_pseudo_qconst16_shape[2];

// tfl.pseudo_qconst17
extern const int8_t streaming_tfl_pseudo_qconst17[32];
extern const uint32_t streaming_tfl_pseudo_qconst17_shape[1];

// tfl.pseudo_qconst18
extern const int8_t streaming_tfl_pseudo_qconst18[1024];
extern const uint32_t streaming_tfl_pseudo_qconst18_shape[2];

// PartitionedCall:1
extern const int8_t streaming_PartitionedCall_1[32];
extern const uint32_t streaming_PartitionedCall_1_shape[2];

// model/tf.linalg.matmul_7/MatMul;model/tf.math.add_4/Add
extern const int8_t streaming_model_tf_linalg_matmul_7_MatMul_model_tf_math_add_4_Add[1];
extern const uint32_t streaming_model_tf_linalg_matmul_7_MatMul_model_tf_math_add_4_Add_shape[3];

// PartitionedCall:0
extern const int8_t streaming_PartitionedCall_0[5];
extern const uint32_t streaming_PartitionedCall_0_shape[2];

#endif // STREAMING_REAL_WEIGHTS_H
