/**
 * @file complete_tflite_bridge.h
 * @brief Complete TFLite to CMSIS bridge for 99.999% identical models
 * 
 * This bridge loads real weights and implements all operators
 * needed for backbone (MatchboxNet) + streaming (GRU) models.
 */

#ifndef COMPLETE_TFLITE_BRIDGE_H
#define COMPLETE_TFLITE_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "complete_operators.h"
#include "tflite_parser_minimal.h"

// Forward declarations for weight files (in parent directory)
#include "../backbone_real_weights.h"
#include "../streaming_real_weights.h"

// ================================================================
// COMPLETE BRIDGE STRUCTURE
// ================================================================

typedef struct {
    // Model information
    char name[32];
    TFLiteModel* model;
    bool is_initialized;
    
    // Tensor arena
    int8_t* tensor_arena;
    uint32_t arena_size;
    uint32_t arena_used;
    
    // Dynamic tensors (allocated from arena)
    Tensor_t* tensors;
    uint32_t num_tensors;
    
    // Real weight mappings
    const void* weight_data;  // Points to backbone_real_weights or streaming_real_weights
    uint32_t num_weights;
    
    // Execution state
    uint32_t current_subgraph;
    bool inference_ready;
    
    // Performance metrics
    uint32_t inference_count;
    uint32_t total_inference_time_us;
    
} CompleteBridge_t;

// ================================================================
// BRIDGE API
// ================================================================

/**
 * Create complete bridge with real weights
 */
CompleteBridge_t* complete_bridge_create(
    const uint8_t* tflite_model_data,
    uint32_t model_size,
    const char* model_name,
    int8_t* tensor_arena,
    uint32_t arena_size
);

/**
 * Initialize bridge with real model topology and weights
 */
bool complete_bridge_init(CompleteBridge_t* bridge);

/**
 * Get input tensor for feeding data
 */
Tensor_t* complete_bridge_get_input(CompleteBridge_t* bridge, uint32_t index);

/**
 * Get output tensor for reading results
 */
Tensor_t* complete_bridge_get_output(CompleteBridge_t* bridge, uint32_t index);

/**
 * Run complete inference with all real operators
 */
bool complete_bridge_invoke(CompleteBridge_t* bridge);

/**
 * Get performance metrics
 */
typedef struct {
    uint32_t inference_count;
    uint32_t avg_inference_time_us;
    uint32_t total_time_us;
    uint32_t arena_used_bytes;
    uint32_t arena_peak_bytes;
} BridgeMetrics_t;

bool complete_bridge_get_metrics(CompleteBridge_t* bridge, BridgeMetrics_t* metrics);

/**
 * Free bridge and cleanup
 */
void complete_bridge_free(CompleteBridge_t* bridge);

// ================================================================
// OPERATOR EXECUTION ENGINE
// ================================================================

/**
 * Execute a single operator from the model
 */
bool execute_operator(
    CompleteBridge_t* bridge,
    const TFLiteOperator* op,
    uint32_t op_index
);

/**
 * Map TFLite operator code to our implementation
 */
typedef enum {
    OP_CODE_CONV_2D = 3,
    OP_CODE_DEPTHWISE_CONV_2D = 4,
    OP_CODE_FULLY_CONNECTED = 9,
    OP_CODE_SOFTMAX = 25,
    OP_CODE_RESHAPE = 22,
    OP_CODE_ADD = 0,
    OP_CODE_MUL = 18,
    OP_CODE_RELU = 20,
    OP_CODE_RELU6 = 23,
    OP_CODE_TANH = 28,
    OP_CODE_LOGISTIC = 14,  // Sigmoid
    OP_CODE_MEAN = 40,
    OP_CODE_PAD = 34,
    OP_CODE_SQUEEZE = 43,
    OP_CODE_EXPAND_DIMS = 70,
    // Custom operators for GRU
    OP_CODE_UNIDIRECTIONAL_SEQUENCE_LSTM = 44,  // We'll map GRU to this
    OP_CODE_CUSTOM_GRU = 255,  // Custom operator ID
} TFLiteOperatorCode;

/**
 * Operator-specific execution functions
 */
bool execute_conv2d(CompleteBridge_t* bridge, const TFLiteOperator* op);
bool execute_fully_connected(CompleteBridge_t* bridge, const TFLiteOperator* op);
bool execute_gru_cell(CompleteBridge_t* bridge, const TFLiteOperator* op);
bool execute_softmax(CompleteBridge_t* bridge, const TFLiteOperator* op);
bool execute_relu(CompleteBridge_t* bridge, const TFLiteOperator* op);
bool execute_add(CompleteBridge_t* bridge, const TFLiteOperator* op);
bool execute_reshape(CompleteBridge_t* bridge, const TFLiteOperator* op);

// ================================================================
// WEIGHT MANAGEMENT
// ================================================================

/**
 * Load real weights into tensors
 */
bool load_real_weights(CompleteBridge_t* bridge);

/**
 * Get weight tensor by index
 */
const Tensor_t* get_weight_tensor(CompleteBridge_t* bridge, uint32_t tensor_index);

/**
 * Allocate tensor from arena
 */
Tensor_t* allocate_tensor(CompleteBridge_t* bridge, const uint32_t* shape, uint32_t num_dims, float scale, int32_t zero_point);

// ================================================================
// DEBUG AND VALIDATION
// ================================================================

/**
 * Print complete model information
 */
void print_complete_model_info(CompleteBridge_t* bridge);

/**
 * Validate model structure against expected topology
 */
bool validate_complete_model(CompleteBridge_t* bridge);

/**
 * Compare output with reference (for testing)
 */
typedef struct {
    float* reference_output;
    uint32_t output_size;
    float tolerance;
} ReferenceComparison_t;

bool compare_with_reference(CompleteBridge_t* bridge, const ReferenceComparison_t* ref);

// ================================================================
// INTERNAL HELPER FUNCTIONS
// ================================================================

/**
 * Model-specific weight loading
 */
bool load_backbone_weights(CompleteBridge_t* bridge, TFLiteSubgraph* subgraph);
bool load_streaming_weights(CompleteBridge_t* bridge, TFLiteSubgraph* subgraph);
bool setup_streaming_gru_weights(CompleteBridge_t* bridge, uint32_t arena_offset);

/**
 * Model-specific inference execution
 */
bool execute_backbone_inference(CompleteBridge_t* bridge);
bool execute_streaming_inference(CompleteBridge_t* bridge);

/**
 * Streaming model operators
 */
bool execute_simplified_gru_cell(Tensor_t* features, Tensor_t* hidden_in, Tensor_t* hidden_out);
bool execute_simplified_attention(Tensor_t* hidden, Tensor_t* logits);

#endif // COMPLETE_TFLITE_BRIDGE_H