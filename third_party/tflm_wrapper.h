#ifndef TFLM_WRAPPER_H
#define TFLM_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles for TensorFlow Lite Micro objects
typedef void* TFLMModel;
typedef void* TFLMInterpreter;

// Status codes
typedef enum {
    TFLM_OK = 0,
    TFLM_ERROR = 1,
    TFLM_INVALID_ARGUMENT = 2,
    TFLM_OUT_OF_MEMORY = 3
} TFLMStatus;

// C wrapper functions for TensorFlow Lite Micro

/**
 * Create a model from flatbuffer data
 */
TFLMModel tflm_create_model(const unsigned char* model_data, unsigned int model_size);

/**
 * Create interpreter for a model
 */
TFLMInterpreter tflm_create_interpreter(TFLMModel model, unsigned char* arena, unsigned int arena_size);

/**
 * Allocate tensors for the interpreter
 */
TFLMStatus tflm_allocate_tensors(TFLMInterpreter interpreter);

/**
 * Get input tensor data pointer
 */
float* tflm_get_input_data(TFLMInterpreter interpreter, int input_index);

/**
 * Get output tensor data pointer  
 */
float* tflm_get_output_data(TFLMInterpreter interpreter, int output_index);

/**
 * Run inference
 */
TFLMStatus tflm_invoke(TFLMInterpreter interpreter);

/**
 * Get tensor dimensions
 */
int tflm_get_input_dims(TFLMInterpreter interpreter, int input_index, int* dims, int max_dims);
int tflm_get_output_dims(TFLMInterpreter interpreter, int output_index, int* dims, int max_dims);

/**
 * Get arena usage
 */
unsigned int tflm_get_arena_used_bytes(TFLMInterpreter interpreter);

/**
 * Cleanup functions
 */
void tflm_destroy_interpreter(TFLMInterpreter interpreter);
void tflm_destroy_model(TFLMModel model);

#ifdef __cplusplus
}
#endif

#endif // TFLM_WRAPPER_H