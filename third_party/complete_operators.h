/**
 * @file complete_operators.h  
 * @brief Complete neural network operators for 99.999% identical model
 * 
 * Implements ALL operators needed for backbone + streaming models:
 * - GRU cells (complete implementation)
 * - Matrix multiplication with quantization
 * - Attention mechanism
 * - Softmax, Sigmoid, Tanh, ReLU
 * - Add, Subtract, Multiply
 * - Reshape, Reduce operations
 */

#ifndef COMPLETE_OPERATORS_H
#define COMPLETE_OPERATORS_H

#include <stdint.h>
#include <stdbool.h>

// ================================================================
// TENSOR STRUCTURES
// ================================================================

typedef struct {
    int8_t* data;
    uint32_t shape[4];
    uint32_t num_dims;
    float scale;
    int32_t zero_point;
    uint32_t size;  // Total number of elements
} Tensor_t;

typedef struct {
    float scale;
    int32_t zero_point;
} QuantParams_t;

// ================================================================
// BACKBONE OPERATORS (MatchboxNet)
// ================================================================

/**
 * 1D Convolution with int8 quantization
 */
bool op_conv1d_int8(
    const Tensor_t* input,      // [batch, seq_len, in_channels]
    const Tensor_t* kernel,     // [kernel_size, in_channels, out_channels]
    const Tensor_t* bias,       // [out_channels]
    Tensor_t* output,           // [batch, out_seq_len, out_channels]
    uint32_t stride,
    uint32_t padding,
    const QuantParams_t* output_quant
);

/**
 * ReLU6 activation
 */
bool op_relu6_int8(const Tensor_t* input, Tensor_t* output);

/**
 * Global average pooling / reduce mean
 */
bool op_reduce_mean_int8(
    const Tensor_t* input,
    Tensor_t* output,
    const uint32_t* axes,
    uint32_t num_axes,
    bool keep_dims
);

/**
 * Squeeze/Expand dimensions
 */
bool op_squeeze_int8(const Tensor_t* input, Tensor_t* output, uint32_t axis);
bool op_expand_dims_int8(const Tensor_t* input, Tensor_t* output, uint32_t axis);

/**
 * Padding operation
 */
bool op_pad_int8(
    const Tensor_t* input,
    Tensor_t* output,
    const uint32_t* paddings,  // [num_dims, 2] - before, after for each dim
    uint32_t num_dims
);

// ================================================================
// STREAMING OPERATORS (GRU + Attention)
// ================================================================

/**
 * Complete GRU cell implementation
 * Supports reset gate, update gate, and new gate
 */
typedef struct {
    const Tensor_t* input_kernel;      // [input_size, 3*hidden_size] - reset, update, new
    const Tensor_t* recurrent_kernel;  // [hidden_size, 3*hidden_size]
    const Tensor_t* bias;              // [3*hidden_size] 
    const QuantParams_t* output_quant;
} GRUParams_t;

bool op_gru_cell_int8(
    const Tensor_t* input,          // [batch, input_size]
    const Tensor_t* hidden_state,   // [batch, hidden_size]
    const GRUParams_t* params,
    Tensor_t* new_hidden_state      // [batch, hidden_size]
);

/**
 * Matrix multiplication with bias (Dense/Linear layer)
 */
bool op_dense_int8(
    const Tensor_t* input,      // [batch, input_size]
    const Tensor_t* weights,    // [input_size, output_size]
    const Tensor_t* bias,       // [output_size]
    Tensor_t* output,           // [batch, output_size]
    const QuantParams_t* output_quant
);

/**
 * Attention mechanism (simplified)
 * Computes attention weights and applies to values
 */
bool op_attention_int8(
    const Tensor_t* query,      // [batch, seq_len, hidden_size]
    const Tensor_t* key,        // [batch, seq_len, hidden_size] 
    const Tensor_t* value,      // [batch, seq_len, hidden_size]
    Tensor_t* output,           // [batch, seq_len, hidden_size]
    const QuantParams_t* output_quant
);

// ================================================================
// ACTIVATION FUNCTIONS
// ================================================================

bool op_sigmoid_int8(const Tensor_t* input, Tensor_t* output);
bool op_tanh_int8(const Tensor_t* input, Tensor_t* output);
bool op_softmax_int8(const Tensor_t* input, Tensor_t* output, uint32_t axis);
bool op_relu_int8(const Tensor_t* input, Tensor_t* output);

// ================================================================
// ARITHMETIC OPERATIONS
// ================================================================

bool op_add_int8(const Tensor_t* input1, const Tensor_t* input2, Tensor_t* output, const QuantParams_t* output_quant);
bool op_subtract_int8(const Tensor_t* input1, const Tensor_t* input2, Tensor_t* output, const QuantParams_t* output_quant);
bool op_multiply_int8(const Tensor_t* input1, const Tensor_t* input2, Tensor_t* output, const QuantParams_t* output_quant);

// ================================================================
// SHAPE OPERATIONS
// ================================================================

bool op_reshape_int8(const Tensor_t* input, Tensor_t* output, const uint32_t* new_shape, uint32_t num_dims);
bool op_reduce_sum_int8(const Tensor_t* input, Tensor_t* output, const uint32_t* axes, uint32_t num_axes, bool keep_dims);

// ================================================================
// UTILITY FUNCTIONS
// ================================================================

/**
 * Quantize float to int8
 */
int8_t quantize_float_to_int8(float value, float scale, int32_t zero_point);

/**
 * Dequantize int8 to float
 */
float dequantize_int8_to_float(int8_t value, float scale, int32_t zero_point);

/**
 * Tensor shape utilities
 */
uint32_t tensor_get_size(const Tensor_t* tensor);
bool tensor_shapes_compatible(const Tensor_t* a, const Tensor_t* b);
void tensor_copy_shape(const Tensor_t* src, Tensor_t* dst);

/**
 * Debug and validation
 */
void print_tensor_info(const Tensor_t* tensor, const char* name);
bool validate_tensor(const Tensor_t* tensor);

#endif // COMPLETE_OPERATORS_H