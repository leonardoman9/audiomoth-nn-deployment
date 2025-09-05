#ifndef CMSIS_WRAPPER_H
#define CMSIS_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================================================================
// CMSIS-NN WRAPPER FOR AUDIOMOTH
// Pure C interface - no TFLM dependency
// ================================================================

// Opaque interpreter type (compatible with existing code)
typedef struct CMSIS_Interpreter_s TFLM_Interpreter_t;

// ================================================================
// API FUNCTIONS - Compatible with nn_model.c
// ================================================================

/**
 * @brief Create interpreter from .tflite model data
 * @param model_data Pointer to .tflite model (we'll ignore this and use hardcoded models)
 * @param model_size Size of model data (ignored)
 * @param arena Memory arena for tensors
 * @param arena_size Size of arena
 * @return Interpreter handle or NULL on failure
 */
TFLM_Interpreter_t* tflm_create_interpreter(
    const unsigned char* model_data,
    size_t model_size,
    uint8_t* arena,
    size_t arena_size);

/**
 * @brief Destroy interpreter and free resources
 */
void tflm_destroy_interpreter(TFLM_Interpreter_t* interpreter);

/**
 * @brief Run inference
 * @return true on success, false on failure
 */
bool tflm_invoke(TFLM_Interpreter_t* interpreter);

/**
 * @brief Get input tensor data pointer
 * @param index Tensor index (0 for most models)
 * @return Pointer to input tensor data (int8_t* for quantized models)
 */
void* tflm_get_input_tensor_data(TFLM_Interpreter_t* interpreter, int index);

/**
 * @brief Get output tensor data pointer
 * @param index Tensor index (0 for most models)
 * @return Pointer to output tensor data (int8_t* for quantized models)
 */
void* tflm_get_output_tensor_data(TFLM_Interpreter_t* interpreter, int index);

#ifdef __cplusplus
}
#endif

#endif // CMSIS_WRAPPER_H