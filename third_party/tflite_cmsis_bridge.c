#include "tflite_cmsis_bridge.h"
#include "backbone_weights.h"
#include "streaming_weights.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Include CMSIS-NN for actual computation
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

// ================================================================
// HELPER FUNCTIONS
// ================================================================

static bool setup_backbone_model(TFLiteCMSISBridge* bridge) {
    // Backbone model info from TensorFlow analysis:
    // Input: [1, 18, 40] int8, scale=0.02352941, zero_point=-1
    // Output: [1, 18, 32] int8, scale=0.03810385, zero_point=-55
    
    bridge->num_inputs = 1;
    bridge->num_outputs = 1;
    
    // Input tensor info
    bridge->input_shape[0] = 1;   // batch
    bridge->input_shape[1] = 18;  // height/time
    bridge->input_shape[2] = 40;  // width/features
    bridge->input_shape[3] = 1;   // channels
    bridge->input_scale = 0.02352941f;
    bridge->input_zero_point = -1;
    
    // Output tensor info  
    bridge->output_shape[0] = 1;   // batch
    bridge->output_shape[1] = 18;  // height/time
    bridge->output_shape[2] = 32;  // width/features
    bridge->output_shape[3] = 1;   // channels
    bridge->output_scale = 0.03810385f;
    bridge->output_zero_point = -55;
    
    printf("TFLite-CMSIS: Backbone model configured\n");
    printf("  Input: [%d,%d,%d,%d], scale=%f, zero_point=%d\n",
           bridge->input_shape[0], bridge->input_shape[1], 
           bridge->input_shape[2], bridge->input_shape[3],
           bridge->input_scale, bridge->input_zero_point);
    printf("  Output: [%d,%d,%d,%d], scale=%f, zero_point=%d\n",
           bridge->output_shape[0], bridge->output_shape[1], 
           bridge->output_shape[2], bridge->output_shape[3],
           bridge->output_scale, bridge->output_zero_point);
    
    return true;
}

static bool setup_streaming_model(TFLiteCMSISBridge* bridge) {
    // Streaming model info from TensorFlow analysis:
    // Input 0: [1, 32] int8, scale=0.01360857, zero_point=0
    // Input 1: [1, 32] int8, scale=0.00784314, zero_point=-1 (hidden state)
    // Output 0: [1, 32] int8, scale=0.00571844, zero_point=-4 (new hidden state)
    // Output 1: [1, 5] int8, scale=0.00317491, zero_point=18 (classification)
    
    bridge->num_inputs = 2;  // feature_frame + hidden_state
    bridge->num_outputs = 2; // new_hidden_state + classification
    
    // Input tensor info (feature_frame)
    bridge->input_shape[0] = 1;   // batch
    bridge->input_shape[1] = 32;  // features
    bridge->input_shape[2] = 1;   // height
    bridge->input_shape[3] = 1;   // channels
    bridge->input_scale = 0.01360857f;
    bridge->input_zero_point = 0;
    
    // Output tensor info (classification)
    bridge->output_shape[0] = 1;   // batch
    bridge->output_shape[1] = 5;   // classes
    bridge->output_shape[2] = 1;   // height
    bridge->output_shape[3] = 1;   // channels
    bridge->output_scale = 0.00317491f;
    bridge->output_zero_point = 18;
    
    printf("TFLite-CMSIS: Streaming model configured\n");
    printf("  Input: [%d,%d,%d,%d], scale=%f, zero_point=%d\n",
           bridge->input_shape[0], bridge->input_shape[1], 
           bridge->input_shape[2], bridge->input_shape[3],
           bridge->input_scale, bridge->input_zero_point);
    printf("  Output: [%d,%d,%d,%d], scale=%f, zero_point=%d\n",
           bridge->output_shape[0], bridge->output_shape[1], 
           bridge->output_shape[2], bridge->output_shape[3],
           bridge->output_scale, bridge->output_zero_point);
    
    return true;
}

static bool run_backbone_inference(TFLiteCMSISBridge* bridge) {
    // Simplified backbone inference using CMSIS-NN
    // Input: [1, 18, 40] -> Output: [1, 18, 32]
    // 
    // This is a simplified implementation that demonstrates real CMSIS-NN usage
    // For the full 202-tensor MatchboxNet, we would need to implement all layers
    
    printf("TFLite-CMSIS: Running backbone inference...\n");
    
    // Input dimensions
    uint32_t input_size = bridge->input_shape[1] * bridge->input_shape[2]; // 18 * 40 = 720
    uint32_t output_size = bridge->output_shape[1] * bridge->output_shape[2]; // 18 * 32 = 576
    
    // Simulate a simplified backbone network:
    // 1. Fully connected layer: 720 -> 256
    // 2. ReLU activation
    // 3. Fully connected layer: 256 -> 576
    
    // Temporary buffer for intermediate results
    static int8_t temp_buffer[256];
    
    // Layer 1: FC 720 -> 256
    cmsis_nn_context fc1_ctx;
    fc1_ctx.buf = NULL; // No additional buffer needed for this size
    fc1_ctx.size = 0;
    
    cmsis_nn_fc_params fc1_params;
    fc1_params.input_offset = bridge->input_zero_point;
    fc1_params.filter_offset = 0;
    fc1_params.output_offset = 0;
    fc1_params.activation.min = -128;
    fc1_params.activation.max = 127;
    
    cmsis_nn_per_tensor_quant_params fc1_quant_params;
    fc1_quant_params.multiplier = 1073741824; // Simplified multiplier
    fc1_quant_params.shift = 0;
    
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = input_size;
    
    cmsis_nn_dims filter_dims;
    filter_dims.n = 256;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = input_size;
    
    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = 256;
    
    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = 256;
    
    // TODO: Use real extracted weights (for now, minimal placeholders)
    static const int8_t fc1_weights[8] = {1, -1, 1, -1, 1, -1, 1, -1}; // Minimal placeholder
    static const int32_t fc1_bias[8] = {0, 0, 0, 0, 0, 0, 0, 0};       // Minimal placeholder
    
    arm_cmsis_nn_status status = arm_fully_connected_s8(
        &fc1_ctx,
        &fc1_params,
        &fc1_quant_params,
        &input_dims,
        bridge->input_tensor,
        &filter_dims,
        fc1_weights,
        &bias_dims,
        fc1_bias,
        &output_dims,
        temp_buffer
    );
    
    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("TFLite-CMSIS: FC1 layer failed with status %d\n", status);
        return false;
    }
    
    // Layer 2: ReLU activation
    arm_relu_q7(temp_buffer, 256);
    
    // Layer 3: FC 256 -> 576 (output)
    cmsis_nn_context fc2_ctx;
    fc2_ctx.buf = NULL;
    fc2_ctx.size = 0;
    
    cmsis_nn_fc_params fc2_params;
    fc2_params.input_offset = 0;
    fc2_params.filter_offset = 0;
    fc2_params.output_offset = bridge->output_zero_point;
    fc2_params.activation.min = -128;
    fc2_params.activation.max = 127;
    
    cmsis_nn_per_tensor_quant_params fc2_quant_params;
    fc2_quant_params.multiplier = 1073741824;
    fc2_quant_params.shift = 0;
    
    cmsis_nn_dims fc2_input_dims;
    fc2_input_dims.n = 1;
    fc2_input_dims.h = 1;
    fc2_input_dims.w = 1;
    fc2_input_dims.c = 256;
    
    cmsis_nn_dims fc2_filter_dims;
    fc2_filter_dims.n = output_size;
    fc2_filter_dims.h = 1;
    fc2_filter_dims.w = 1;
    fc2_filter_dims.c = 256;
    
    cmsis_nn_dims fc2_bias_dims;
    fc2_bias_dims.n = 1;
    fc2_bias_dims.h = 1;
    fc2_bias_dims.w = 1;
    fc2_bias_dims.c = output_size;
    
    cmsis_nn_dims fc2_output_dims;
    fc2_output_dims.n = 1;
    fc2_output_dims.h = 1;
    fc2_output_dims.w = 1;
    fc2_output_dims.c = output_size;
    
    static const int8_t fc2_weights[8] = {1, -1, 1, -1, 1, -1, 1, -1}; // Minimal placeholder
    static const int32_t fc2_bias[8] = {0, 0, 0, 0, 0, 0, 0, 0};     // Minimal placeholder
    
    status = arm_fully_connected_s8(
        &fc2_ctx,
        &fc2_params,
        &fc2_quant_params,
        &fc2_input_dims,
        temp_buffer,
        &fc2_filter_dims,
        fc2_weights,
        &fc2_bias_dims,
        fc2_bias,
        &fc2_output_dims,
        bridge->output_tensor
    );
    
    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("TFLite-CMSIS: FC2 layer failed with status %d\n", status);
        return false;
    }
    
    printf("TFLite-CMSIS: Backbone inference completed successfully\n");
    return true;
}

static bool run_streaming_inference(TFLiteCMSISBridge* bridge) {
    printf("TFLite-CMSIS: Running streaming inference...\n");
    
    // Simplified streaming inference: [1, 32] -> [1, 5]
    // This represents the GRU + classification layers
    
    uint32_t input_size = 32;
    uint32_t output_size = 5;
    
    // Simplified GRU + classification using FC layers
    static int8_t temp_buffer[64];
    
    // Layer 1: FC 32 -> 64 (GRU simulation)
    cmsis_nn_context fc1_ctx;
    fc1_ctx.buf = NULL;
    fc1_ctx.size = 0;
    
    cmsis_nn_fc_params fc1_params;
    fc1_params.input_offset = bridge->input_zero_point;
    fc1_params.filter_offset = 0;
    fc1_params.output_offset = 0;
    fc1_params.activation.min = -128;
    fc1_params.activation.max = 127;
    
    cmsis_nn_per_tensor_quant_params fc1_quant_params;
    fc1_quant_params.multiplier = 1073741824;
    fc1_quant_params.shift = 0;
    
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = input_size;
    
    cmsis_nn_dims filter_dims;
    filter_dims.n = 64;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = input_size;
    
    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = 64;
    
    cmsis_nn_dims temp_dims;
    temp_dims.n = 1;
    temp_dims.h = 1;
    temp_dims.w = 1;
    temp_dims.c = 64;
    
    static const int8_t fc1_weights[8] = {1, -1, 1, -1, 1, -1, 1, -1}; // Minimal placeholder
    static const int32_t fc1_bias[8] = {0, 0, 0, 0, 0, 0, 0, 0};     // Minimal placeholder
    
    arm_cmsis_nn_status status = arm_fully_connected_s8(
        &fc1_ctx,
        &fc1_params,
        &fc1_quant_params,
        &input_dims,
        bridge->input_tensor,
        &filter_dims,
        fc1_weights,
        &bias_dims,
        fc1_bias,
        &temp_dims,
        temp_buffer
    );
    
    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("TFLite-CMSIS: Streaming FC1 failed with status %d\n", status);
        return false;
    }
    
    // Tanh activation (simplified)
    arm_relu_q7(temp_buffer, 64);
    
    // Layer 2: FC 64 -> 5 (classification)
    cmsis_nn_fc_params fc2_params;
    fc2_params.input_offset = 0;
    fc2_params.filter_offset = 0;
    fc2_params.output_offset = bridge->output_zero_point;
    fc2_params.activation.min = -128;
    fc2_params.activation.max = 127;
    
    cmsis_nn_dims fc2_input_dims;
    fc2_input_dims.n = 1;
    fc2_input_dims.h = 1;
    fc2_input_dims.w = 1;
    fc2_input_dims.c = 64;
    
    cmsis_nn_dims fc2_filter_dims;
    fc2_filter_dims.n = output_size;
    fc2_filter_dims.h = 1;
    fc2_filter_dims.w = 1;
    fc2_filter_dims.c = 64;
    
    cmsis_nn_dims fc2_output_dims;
    fc2_output_dims.n = 1;
    fc2_output_dims.h = 1;
    fc2_output_dims.w = 1;
    fc2_output_dims.c = output_size;
    
    static const int8_t fc2_weights[8] = {1, -1, 1, -1, 1, -1, 1, -1}; // Minimal placeholder
    static const int32_t fc2_bias[8] = {0, 0, 0, 0, 0, 0, 0, 0};     // Minimal placeholder
    
    cmsis_nn_context fc2_ctx;
    fc2_ctx.buf = NULL;
    fc2_ctx.size = 0;
    
    status = arm_fully_connected_s8(
        &fc2_ctx,
        &fc2_params,
        &fc1_quant_params, // Reuse quant params
        &fc2_input_dims,
        temp_buffer,
        &fc2_filter_dims,
        fc2_weights,
        &bias_dims,
        fc2_bias,
        &fc2_output_dims,
        bridge->output_tensor
    );
    
    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("TFLite-CMSIS: Streaming FC2 failed with status %d\n", status);
        return false;
    }
    
    printf("TFLite-CMSIS: Streaming inference completed successfully\n");
    return true;
}

// ================================================================
// PUBLIC API
// ================================================================

TFLiteCMSISBridge* tflite_cmsis_create(const uint8_t* model_data, 
                                       size_t model_size,
                                       const char* model_name,
                                       uint8_t* arena_ptr,
                                       uint32_t arena_size) {
    if (!model_data || !model_name || !arena_ptr || arena_size == 0) {
        printf("TFLite-CMSIS: Invalid parameters\n");
        return NULL;
    }
    
    printf("TFLite-CMSIS: Creating bridge for model '%s'\n", model_name);
    
    // Static allocation for embedded systems - avoid malloc
    static TFLiteCMSISBridge backbone_bridge_instance;
    static TFLiteCMSISBridge streaming_bridge_instance;
    
    // Choose instance based on model name
    TFLiteCMSISBridge* bridge = NULL;
    if (strcmp(model_name, "backbone") == 0) {
        bridge = &backbone_bridge_instance;
    } else if (strcmp(model_name, "streaming") == 0) {
        bridge = &streaming_bridge_instance;
    } else {
        printf("TFLite-CMSIS: Unknown model name '%s'\n", model_name);
        return NULL;
    }
    
    memset(bridge, 0, sizeof(TFLiteCMSISBridge));
    
    // Parse TFLite model for validation
    bridge->model = parse_tflite_model(model_data, model_size);
    if (!bridge->model) {
        printf("TFLite-CMSIS: Failed to parse TFLite model\n");
        free(bridge);
        return NULL;
    }
    
    // Setup tensor arena
    bridge->tensor_arena = arena_ptr;
    bridge->arena_size = arena_size;
    
    // Copy model name
    strncpy(bridge->model_name, model_name, sizeof(bridge->model_name) - 1);
    bridge->model_name[sizeof(bridge->model_name) - 1] = '\0';
    
    // Configure model-specific parameters
    if (strcmp(model_name, "backbone") == 0) {
        if (!setup_backbone_model(bridge)) {
            free_tflite_model(bridge->model);
            free(bridge);
            return NULL;
        }
    } else if (strcmp(model_name, "streaming") == 0) {
        if (!setup_streaming_model(bridge)) {
            free_tflite_model(bridge->model);
            free(bridge);
            return NULL;
        }
    } else {
        printf("TFLite-CMSIS: Unknown model name '%s'\n", model_name);
        free_tflite_model(bridge->model);
        free(bridge);
        return NULL;
    }
    
    printf("TFLite-CMSIS: Bridge created successfully\n");
    return bridge;
}

bool tflite_cmsis_init(TFLiteCMSISBridge* bridge) {
    if (!bridge || bridge->initialized) {
        return false;
    }
    
    printf("TFLite-CMSIS: Initializing bridge...\n");
    
    // Calculate tensor sizes
    uint32_t input_size = bridge->input_shape[0] * bridge->input_shape[1] * 
                         bridge->input_shape[2] * bridge->input_shape[3];
    uint32_t output_size = bridge->output_shape[0] * bridge->output_shape[1] * 
                          bridge->output_shape[2] * bridge->output_shape[3];
    
    // Check arena size
    uint32_t required_size = input_size + output_size + 1024; // Extra for intermediate buffers
    if (bridge->arena_size < required_size) {
        printf("TFLite-CMSIS: Arena too small. Required: %d, Available: %d\n", 
               required_size, bridge->arena_size);
        return false;
    }
    
    // Allocate input/output tensors in arena
    bridge->input_tensor = (int8_t*)bridge->tensor_arena;
    bridge->output_tensor = bridge->input_tensor + input_size;
    
    // Clear tensors
    memset(bridge->input_tensor, 0, input_size);
    memset(bridge->output_tensor, 0, output_size);
    
    bridge->initialized = true;
    
    printf("TFLite-CMSIS: Bridge initialized successfully\n");
    printf("  Input tensor: %d bytes at %p\n", input_size, (void*)bridge->input_tensor);
    printf("  Output tensor: %d bytes at %p\n", output_size, (void*)bridge->output_tensor);
    
    return true;
}

int8_t* tflite_cmsis_get_input(TFLiteCMSISBridge* bridge) {
    if (!bridge || !bridge->initialized) {
        return NULL;
    }
    return bridge->input_tensor;
}

int8_t* tflite_cmsis_get_output(TFLiteCMSISBridge* bridge) {
    if (!bridge || !bridge->initialized) {
        return NULL;
    }
    return bridge->output_tensor;
}

bool tflite_cmsis_invoke(TFLiteCMSISBridge* bridge) {
    if (!bridge || !bridge->initialized) {
        printf("TFLite-CMSIS: Bridge not initialized\n");
        return false;
    }
    
    printf("TFLite-CMSIS: Running inference for model '%s'\n", bridge->model_name);
    
    if (strcmp(bridge->model_name, "backbone") == 0) {
        return run_backbone_inference(bridge);
    } else if (strcmp(bridge->model_name, "streaming") == 0) {
        return run_streaming_inference(bridge);
    } else {
        printf("TFLite-CMSIS: Unknown model '%s'\n", bridge->model_name);
        return false;
    }
}

bool tflite_cmsis_get_input_info(TFLiteCMSISBridge* bridge, 
                                 uint32_t* shape, 
                                 float* scale, 
                                 int32_t* zero_point) {
    if (!bridge || !shape || !scale || !zero_point) {
        return false;
    }
    
    memcpy(shape, bridge->input_shape, 4 * sizeof(uint32_t));
    *scale = bridge->input_scale;
    *zero_point = bridge->input_zero_point;
    
    return true;
}

bool tflite_cmsis_get_output_info(TFLiteCMSISBridge* bridge, 
                                  uint32_t* shape, 
                                  float* scale, 
                                  int32_t* zero_point) {
    if (!bridge || !shape || !scale || !zero_point) {
        return false;
    }
    
    memcpy(shape, bridge->output_shape, 4 * sizeof(uint32_t));
    *scale = bridge->output_scale;
    *zero_point = bridge->output_zero_point;
    
    return true;
}

void tflite_cmsis_free(TFLiteCMSISBridge* bridge) {
    if (!bridge) {
        return;
    }
    
    if (bridge->model) {
        free_tflite_model(bridge->model);
    }
    
    free(bridge);
    printf("TFLite-CMSIS: Bridge freed\n");
}

void tflite_cmsis_print_info(TFLiteCMSISBridge* bridge) {
    if (!bridge) {
        printf("TFLite-CMSIS: NULL bridge\n");
        return;
    }
    
    printf("\n=== TFLITE-CMSIS BRIDGE INFO ===\n");
    printf("Model: %s\n", bridge->model_name);
    printf("Initialized: %s\n", bridge->initialized ? "Yes" : "No");
    printf("Arena: %d bytes at %p\n", bridge->arena_size, (void*)bridge->tensor_arena);
    
    if (bridge->model) {
        print_model_info(bridge->model);
    }
    
    printf("Input tensor: shape=[%d,%d,%d,%d], scale=%f, zero_point=%d\n",
           bridge->input_shape[0], bridge->input_shape[1], 
           bridge->input_shape[2], bridge->input_shape[3],
           bridge->input_scale, bridge->input_zero_point);
    
    printf("Output tensor: shape=[%d,%d,%d,%d], scale=%f, zero_point=%d\n",
           bridge->output_shape[0], bridge->output_shape[1], 
           bridge->output_shape[2], bridge->output_shape[3],
           bridge->output_scale, bridge->output_zero_point);
    
    printf("===============================\n\n");
}