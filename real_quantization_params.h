#ifndef REAL_QUANTIZATION_PARAMS_H
#define REAL_QUANTIZATION_PARAMS_H

#include <stdint.h>

// ================================================================
// PARAMETRI QUANTIZZAZIONE REALI ESTRATTI DAI TFLITE
// ================================================================

// BACKBONE MODEL QUANTIZATION

// STREAMING MODEL QUANTIZATION

#define STREAMING_INPUT_0_SCALE      0.01360857f
#define STREAMING_INPUT_0_ZERO_POINT 0

#define STREAMING_INPUT_1_SCALE      0.00784314f
#define STREAMING_INPUT_1_ZERO_POINT -1

#define STREAMING_OUTPUT_0_SCALE     0.00571844f
#define STREAMING_OUTPUT_0_ZERO_POINT -4

#define STREAMING_OUTPUT_1_SCALE     0.00317491f
#define STREAMING_OUTPUT_1_ZERO_POINT 18

// DIMENSIONI ESATTE DEI MODELLI
#define BACKBONE_INPUT_SHAPE_H    18
#define BACKBONE_INPUT_SHAPE_W    40
#define BACKBONE_OUTPUT_SHAPE_H   18
#define BACKBONE_OUTPUT_SHAPE_C   32

#define STREAMING_FEATURE_SIZE    32
#define STREAMING_HIDDEN_SIZE     32
#define STREAMING_OUTPUT_CLASSES  5

// CONVERSION FUNCTIONS
static inline int8_t float_to_quantized_backbone_input(float value) {
    return (int8_t)(value / BACKBONE_INPUT_SCALE + BACKBONE_INPUT_ZERO_POINT);
}

static inline float quantized_to_float_backbone_output(int8_t value) {
    return (value - BACKBONE_OUTPUT_ZERO_POINT) * BACKBONE_OUTPUT_SCALE;
}

static inline int8_t float_to_quantized_streaming_input_0(float value) {
    return (int8_t)(value / STREAMING_INPUT_0_SCALE + STREAMING_INPUT_0_ZERO_POINT);
}

static inline float quantized_to_float_streaming_output_0(int8_t value) {
    return (value - STREAMING_OUTPUT_0_ZERO_POINT) * STREAMING_OUTPUT_0_SCALE;
}

#endif // REAL_QUANTIZATION_PARAMS_H
