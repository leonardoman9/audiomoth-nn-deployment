/**
 * tflm_wrapper_virtual.c
 * TFLM wrapper with Flash-Resident Tensor Store
 * Policy: Flash read-only per pesi/modelli, RAM cache per tensori attivi
 * Implements Gemini's suggestion for using Flash as read-only storage
 */

#include "tflm_wrapper.h"
#include "../src/nn/virtual_arena.h"
#include <string.h>

#define MAX_INTERPRETERS 2

typedef struct {
    const unsigned char* model_data;
    unsigned int model_size;
    
    // Virtual tensor IDs instead of direct pointers
    uint32_t input_tensor_id;
    uint32_t output_tensor_id;
    uint32_t intermediate_tensor_ids[16];  // For intermediate layers
    uint32_t num_intermediates;
    
    // Dimensions
    int input_dims[4];
    int output_dims[4];
    
    bool allocated;
    bool is_backbone;
} VirtualInterpreter;

static VirtualInterpreter interpreters[MAX_INTERPRETERS];
static bool interpreter_used[MAX_INTERPRETERS] = {false, false};
static bool virtual_arena_initialized = false;

TFLMModel tflm_create_model(const unsigned char* model_data, unsigned int model_size) {
    if (!model_data || model_size == 0) {
        return NULL;
    }
    
    // Initialize virtual arena on first use
    if (!virtual_arena_initialized) {
        if (!VirtualArena_Init()) {
            return NULL;
        }
        virtual_arena_initialized = true;
    }
    
    return (TFLMModel)model_data;
}

TFLMInterpreter tflm_create_interpreter(TFLMModel model, unsigned char* arena, unsigned int arena_size) {
    if (!model) {
        return NULL;
    }
    
    // Note: arena parameter is ignored - we use virtual arena instead
    (void)arena;
    (void)arena_size;
    
    // Find free interpreter slot
    VirtualInterpreter* interp = NULL;
    int slot = -1;
    for (int i = 0; i < MAX_INTERPRETERS; i++) {
        if (!interpreter_used[i]) {
            interp = &interpreters[i];
            interpreter_used[i] = true;
            slot = i;
            break;
        }
    }
    
    if (!interp) {
        return NULL;
    }
    
    // Initialize interpreter
    memset(interp, 0, sizeof(VirtualInterpreter));
    interp->model_data = (const unsigned char*)model;
    interp->allocated = false;
    
    // Detect model type and set dimensions
    // Backbone: [1, 18, 40] -> [1, 18, 32]
    // Streaming: [1, 32] -> [1, 32]
    if (slot == 0) {  // First interpreter = backbone
        interp->is_backbone = true;
        interp->input_dims[0] = 1;
        interp->input_dims[1] = 18;
        interp->input_dims[2] = 40;
        interp->output_dims[0] = 1;
        interp->output_dims[1] = 18;
        interp->output_dims[2] = 32;
    } else {  // Second interpreter = streaming
        interp->is_backbone = false;
        interp->input_dims[0] = 1;
        interp->input_dims[1] = 32;
        interp->output_dims[0] = 1;
        interp->output_dims[1] = 32;
    }
    
    return (TFLMInterpreter)interp;
}

TFLMStatus tflm_allocate_tensors(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return TFLM_INVALID_ARGUMENT;
    }
    
    VirtualInterpreter* interp = (VirtualInterpreter*)interpreter;
    
    // Calculate sizes
    size_t input_size, output_size;
    
    if (interp->is_backbone) {
        input_size = 1 * 18 * 40 * sizeof(float);   // 2880 bytes
        output_size = 1 * 18 * 32 * sizeof(float);  // 2304 bytes
        
        // Allocate intermediate tensors for backbone layers
        // Conv1: [1, 18, 40] -> [1, 18, 64] = 4608 bytes
        interp->intermediate_tensor_ids[0] = VirtualArena_AllocTensor(4608, "backbone_conv1", false);
        // Pool1: [1, 18, 64] -> [1, 9, 64] = 2304 bytes
        interp->intermediate_tensor_ids[1] = VirtualArena_AllocTensor(2304, "backbone_pool1", false);
        // Conv2: [1, 9, 64] -> [1, 9, 32] = 1152 bytes
        interp->intermediate_tensor_ids[2] = VirtualArena_AllocTensor(1152, "backbone_conv2", false);
        interp->num_intermediates = 3;
        
    } else {
        input_size = 1 * 32 * sizeof(float);        // 128 bytes
        output_size = 1 * 32 * sizeof(float);       // 128 bytes
        
        // GRU hidden state (mutable activation)
        interp->intermediate_tensor_ids[0] = VirtualArena_AllocTensor(128, "gru_hidden", false);
        interp->num_intermediates = 1;
    }
    
    // Allocate virtual tensors (activations - not const)
    interp->input_tensor_id = VirtualArena_AllocTensor(input_size, 
        interp->is_backbone ? "backbone_input" : "streaming_input", false);
    
    interp->output_tensor_id = VirtualArena_AllocTensor(output_size,
        interp->is_backbone ? "backbone_output" : "streaming_output", false);
    
    if (interp->input_tensor_id == INVALID_TENSOR_ID || 
        interp->output_tensor_id == INVALID_TENSOR_ID) {
        return TFLM_OUT_OF_MEMORY;
    }
    
    // Check intermediate allocations
    for (uint32_t i = 0; i < interp->num_intermediates; i++) {
        if (interp->intermediate_tensor_ids[i] == INVALID_TENSOR_ID) {
            return TFLM_OUT_OF_MEMORY;
        }
    }
    
    interp->allocated = true;
    
    // Pin input/output tensors for faster access
    VirtualArena_PinTensor(interp->input_tensor_id);
    VirtualArena_PinTensor(interp->output_tensor_id);
    
    return TFLM_OK;
}

float* tflm_get_input_data(TFLMInterpreter interpreter, int input_index) {
    if (!interpreter || input_index != 0) {
        return NULL;
    }
    
    VirtualInterpreter* interp = (VirtualInterpreter*)interpreter;
    if (!interp->allocated) {
        return NULL;
    }
    
    // Get tensor from virtual arena (may trigger swap from Flash)
    return (float*)VirtualArena_GetTensor(interp->input_tensor_id);
}

float* tflm_get_output_data(TFLMInterpreter interpreter, int output_index) {
    if (!interpreter || output_index != 0) {
        return NULL;
    }
    
    VirtualInterpreter* interp = (VirtualInterpreter*)interpreter;
    if (!interp->allocated) {
        return NULL;
    }
    
    // Get tensor from virtual arena (may trigger swap from Flash)
    return (float*)VirtualArena_GetTensor(interp->output_tensor_id);
}

TFLMStatus tflm_invoke(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return TFLM_INVALID_ARGUMENT;
    }
    
    VirtualInterpreter* interp = (VirtualInterpreter*)interpreter;
    if (!interp->allocated) {
        return TFLM_ERROR;  // Not initialized
    }
    
    // Simulated inference with virtual memory management
    
    // 1. Load input tensor to RAM
    float* input = (float*)VirtualArena_GetTensor(interp->input_tensor_id);
    if (!input) {
        return TFLM_ERROR;
    }
    
    // 2. Process through layers (with intermediate swapping)
    for (uint32_t i = 0; i < interp->num_intermediates; i++) {
        // Load intermediate tensor
        void* intermediate = VirtualArena_GetTensor(interp->intermediate_tensor_ids[i]);
        if (!intermediate) {
            return TFLM_ERROR;
        }
        
        // Simulate layer processing
        // Real implementation would call actual layer functions
        
        // Mark as dirty if modified
        // No MarkDirty - Flash è read-only
        
        // Unpin previous intermediate to allow swapping
        if (i > 0) {
            VirtualArena_UnpinTensor(interp->intermediate_tensor_ids[i-1]);
        }
    }
    
    // 3. Write output
    float* output = (float*)VirtualArena_GetTensor(interp->output_tensor_id);
    if (!output) {
        return TFLM_ERROR;
    }
    
    // Simulate final layer
    // memcpy(output, last_intermediate, output_size);
    // No MarkDirty - output tensori restano in RAM
    
    return TFLM_OK;
}

void tflm_destroy_interpreter(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return;
    }
    
    // Find and free interpreter slot
    for (int i = 0; i < MAX_INTERPRETERS; i++) {
        if (&interpreters[i] == (VirtualInterpreter*)interpreter) {
            
            // Unpin all tensors
            VirtualArena_UnpinTensor(interpreters[i].input_tensor_id);
            VirtualArena_UnpinTensor(interpreters[i].output_tensor_id);
            
            memset(&interpreters[i], 0, sizeof(VirtualInterpreter));
            interpreter_used[i] = false;
            break;
        }
    }
}

// Debug function to get virtual arena statistics
void tflm_get_arena_stats(VirtualArenaStats_t* stats) {
    VirtualArena_GetStats(stats);
}