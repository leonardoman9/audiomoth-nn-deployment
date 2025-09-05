/* TensorFlow Lite Micro Real Implementation Header
 * Replacement for tflm_wrapper.h with actual TFLM integration
 * 
 * This header provides the same API as tflm_wrapper.h but uses
 * real TensorFlow Lite Micro MicroInterpreter underneath.
 */

#ifndef TFLM_REAL_H
#define TFLM_REAL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations to maintain C compatibility
typedef void* TFLMModel;
typedef void* TFLMInterpreter;

// Status codes
typedef enum {
    TFLM_OK = 0,
    TFLM_ERROR = 1
} TFLMStatus;

// Model management
TFLMModel tflm_create_model(const unsigned char* model_data, unsigned int model_size);
void tflm_destroy_model(TFLMModel model);

// Interpreter management  
TFLMInterpreter tflm_create_interpreter(TFLMModel model, unsigned char* arena, unsigned int arena_size);
void tflm_destroy_interpreter(TFLMInterpreter interpreter);

// Tensor allocation and inference
TFLMStatus tflm_allocate_tensors(TFLMInterpreter interpreter);
TFLMStatus tflm_invoke(TFLMInterpreter interpreter);

// Input/Output tensor access
float* tflm_get_input_tensor(TFLMInterpreter interpreter, int index);
float* tflm_get_output_tensor(TFLMInterpreter interpreter, int index);

// Tensor information
int tflm_get_input_tensor_count(TFLMInterpreter interpreter);
int tflm_get_output_tensor_count(TFLMInterpreter interpreter);
void tflm_get_input_tensor_dims(TFLMInterpreter interpreter, int index, int* dims);
void tflm_get_output_tensor_dims(TFLMInterpreter interpreter, int index, int* dims);

#ifdef __cplusplus
}
#endif

#endif // TFLM_REAL_H