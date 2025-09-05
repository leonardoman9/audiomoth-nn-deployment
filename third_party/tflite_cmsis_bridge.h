#ifndef TFLITE_CMSIS_BRIDGE_H
#define TFLITE_CMSIS_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "tflite_parser_minimal.h"

#ifdef __cplusplus
extern "C" {
#endif

// ================================================================
// TFLITE-CMSIS BRIDGE
// Combina parser TFLite + executor CMSIS-NN per AudioMoth
// ================================================================

// Bridge interpreter context
typedef struct {
    TFLiteModel* model;           // Parsed TFLite model
    uint8_t* tensor_arena;        // Memory arena for tensors
    uint32_t arena_size;          // Arena size
    bool initialized;             // Initialization state
    
    // Model info
    char model_name[64];          // "backbone" or "streaming"
    uint32_t num_inputs;          // Number of input tensors
    uint32_t num_outputs;         // Number of output tensors
    
    // Input/Output tensor info
    uint32_t input_shape[4];      // Input tensor shape [batch, height, width, channels]
    uint32_t output_shape[4];     // Output tensor shape
    float input_scale;            // Input quantization scale
    int32_t input_zero_point;     // Input quantization zero point
    float output_scale;           // Output quantization scale
    int32_t output_zero_point;    // Output quantization zero point
    
    // Tensor pointers
    int8_t* input_tensor;         // Input tensor data
    int8_t* output_tensor;        // Output tensor data
    
} TFLiteCMSISBridge;

// ================================================================
// BRIDGE API
// ================================================================

/**
 * @brief Create TFLite-CMSIS bridge from model data
 * @param model_data Raw .tflite model data
 * @param model_size Size of model data
 * @param model_name "backbone" or "streaming"
 * @param arena_ptr Memory arena for tensors
 * @param arena_size Arena size in bytes
 * @return Bridge context or NULL if error
 */
TFLiteCMSISBridge* tflite_cmsis_create(const uint8_t* model_data, 
                                       size_t model_size,
                                       const char* model_name,
                                       uint8_t* arena_ptr,
                                       uint32_t arena_size);

/**
 * @brief Initialize bridge and allocate tensors
 * @param bridge Bridge context
 * @return true if success
 */
bool tflite_cmsis_init(TFLiteCMSISBridge* bridge);

/**
 * @brief Get input tensor pointer
 * @param bridge Bridge context
 * @return Pointer to input tensor data
 */
int8_t* tflite_cmsis_get_input(TFLiteCMSISBridge* bridge);

/**
 * @brief Get output tensor pointer
 * @param bridge Bridge context
 * @return Pointer to output tensor data
 */
int8_t* tflite_cmsis_get_output(TFLiteCMSISBridge* bridge);

/**
 * @brief Run inference using CMSIS-NN
 * @param bridge Bridge context
 * @return true if success
 */
bool tflite_cmsis_invoke(TFLiteCMSISBridge* bridge);

/**
 * @brief Get input tensor info
 * @param bridge Bridge context
 * @param shape Output shape array [4]
 * @param scale Output quantization scale
 * @param zero_point Output quantization zero point
 * @return true if success
 */
bool tflite_cmsis_get_input_info(TFLiteCMSISBridge* bridge, 
                                 uint32_t* shape, 
                                 float* scale, 
                                 int32_t* zero_point);

/**
 * @brief Get output tensor info
 * @param bridge Bridge context
 * @param shape Output shape array [4]
 * @param scale Output quantization scale
 * @param zero_point Output quantization zero point
 * @return true if success
 */
bool tflite_cmsis_get_output_info(TFLiteCMSISBridge* bridge, 
                                  uint32_t* shape, 
                                  float* scale, 
                                  int32_t* zero_point);

/**
 * @brief Free bridge resources
 * @param bridge Bridge context
 */
void tflite_cmsis_free(TFLiteCMSISBridge* bridge);

/**
 * @brief Print bridge info for debugging
 * @param bridge Bridge context
 */
void tflite_cmsis_print_info(TFLiteCMSISBridge* bridge);

#ifdef __cplusplus
}
#endif

#endif // TFLITE_CMSIS_BRIDGE_H