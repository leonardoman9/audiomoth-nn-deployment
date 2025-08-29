#include "tflm_wrapper.h"
#include "../src/nn/tensor_arena.h"
#include "../inc/nn/nn_model.h"  // For NN_GRU_HIDDEN_DIM, NN_NUM_CLASSES
#include <string.h>

// AudioMoth-compatible TFLM stub implementation
// Optimized for our specific models: Backbone [1,18,40]->[1,18,32], Streaming [1,32]->[1,32]
// No malloc() - uses static allocation for embedded safety

#define MAX_INTERPRETERS 2

typedef struct {
    const unsigned char* data;
    unsigned int size;
    bool valid;
} ModelStub;

typedef struct {
    ModelStub model_data;  // Embedded model data, not pointer
    unsigned char* arena;
    unsigned int arena_size;
    unsigned int arena_used;
    float* input0_buffer;   // First input (features)
    float* input1_buffer;   // Second input (hidden state for streaming)
    float* output0_buffer;  // First output (backbone features or logits)
    float* output1_buffer;  // Second output (new hidden state for streaming)
    int input_dims[4];
    int output_dims[4];
    bool allocated;
    bool is_backbone;  // True for backbone, false for streaming
} InterpreterStub;

// Static allocation pool for AudioMoth
static InterpreterStub interpreter_pool[MAX_INTERPRETERS];
static bool interpreter_used[MAX_INTERPRETERS] = {false, false};

TFLMModel tflm_create_model(const unsigned char* model_data, unsigned int model_size) {
    if (!model_data || model_size == 0) {
        return NULL;
    }
    
    // Use static allocation - return pointer to model data directly
    // We'll store model info in interpreter when created
    return (TFLMModel)model_data;
}

TFLMInterpreter tflm_create_interpreter(TFLMModel model, unsigned char* arena, unsigned int arena_size) {
    if (!model || !arena || arena_size < 512) {
        return NULL;
    }
    
    // Find free interpreter slot in static pool
    InterpreterStub* interpreter = NULL;
    for (int i = 0; i < MAX_INTERPRETERS; i++) {
        if (!interpreter_used[i]) {
            interpreter = &interpreter_pool[i];
            interpreter_used[i] = true;
            break;
        }
    }
    
    if (!interpreter) {
        return NULL;  // No free slots
    }
    
    // Store model data
    interpreter->model_data.data = (const unsigned char*)model;
    interpreter->model_data.size = 0;  // We don't have size, but that's OK for stub
    interpreter->model_data.valid = true;
    
    interpreter->arena = arena;
    interpreter->arena_size = arena_size;
    interpreter->arena_used = 0;
    interpreter->input0_buffer = NULL;
    interpreter->input1_buffer = NULL;
    interpreter->output0_buffer = NULL;
    interpreter->output1_buffer = NULL;
    interpreter->allocated = false;
    
    // Detect model type by arena size to set correct dimensions
    // Backbone has larger arena, streaming has smaller
    if (arena_size >= 1024) {
        // Backbone model: [1, 18, 40] -> [1, 18, 32]
        interpreter->is_backbone = true;
        interpreter->input_dims[0] = 1;    // batch
        interpreter->input_dims[1] = 18;   // time frames
        interpreter->input_dims[2] = 40;   // features
        interpreter->input_dims[3] = 0;    // unused
        
        interpreter->output_dims[0] = 1;   // batch
        interpreter->output_dims[1] = 18;  // time frames
        interpreter->output_dims[2] = 32;  // features
        interpreter->output_dims[3] = 0;   // unused
    } else {
        // Streaming model: [1, 32] -> [1, 32]
        interpreter->is_backbone = false;
        interpreter->input_dims[0] = 1;    // batch
        interpreter->input_dims[1] = 32;   // features
        interpreter->input_dims[2] = 0;    // unused
        interpreter->input_dims[3] = 0;    // unused
        
        interpreter->output_dims[0] = 1;   // batch
        interpreter->output_dims[1] = 32;  // features (logits)
        interpreter->output_dims[2] = 0;   // unused
        interpreter->output_dims[3] = 0;   // unused
    }
    
    return (TFLMInterpreter)interpreter;
}

TFLMStatus tflm_allocate_tensors(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return TFLM_INVALID_ARGUMENT;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    
    size_t input_size, output_size;
    
    if (interp->is_backbone) {
        // Backbone: [1, 18, 40] -> [1, 18, 32]
        input_size = 1 * 18 * 40 * sizeof(float);   // 720 floats = 2880 bytes
        output_size = 1 * 18 * 32 * sizeof(float);  // 576 floats = 2304 bytes
    } else {
        // Streaming: [1, 32] -> [1, 32]
        input_size = 1 * 32 * sizeof(float);        // 32 floats = 128 bytes
        output_size = 1 * 32 * sizeof(float);       // 32 floats = 128 bytes
    }
    
    size_t total_needed = input_size + output_size + 256; // Extra for alignment
    
    if (total_needed > interp->arena_size) {
        return TFLM_OUT_OF_MEMORY;
    }
    
    // Allocate from arena with proper alignment
    size_t offset = 0;
    
    if (interp->is_backbone) {
        // Backbone: 1 input, 1 output
        interp->input0_buffer = (float*)(interp->arena + offset);
        offset += input_size;
        interp->output0_buffer = (float*)(interp->arena + offset);
        offset += output_size;
        interp->input1_buffer = NULL;   // Not used
        interp->output1_buffer = NULL;  // Not used
        
        // Initialize buffers
        memset(interp->input0_buffer, 0, input_size);
        memset(interp->output0_buffer, 0, output_size);
    } else {
        // Streaming: 2 inputs, 2 outputs
        size_t input1_size = NN_GRU_HIDDEN_DIM * sizeof(float);  // Hidden state
        size_t output1_size = NN_GRU_HIDDEN_DIM * sizeof(float); // New hidden state
        
        interp->input0_buffer = (float*)(interp->arena + offset);  // Features
        offset += input_size;
        interp->input1_buffer = (float*)(interp->arena + offset);  // Hidden in
        offset += input1_size;
        interp->output0_buffer = (float*)(interp->arena + offset); // Logits
        offset += output_size;
        interp->output1_buffer = (float*)(interp->arena + offset); // Hidden out
        offset += output1_size;
        
        // Initialize buffers
        memset(interp->input0_buffer, 0, input_size);
        memset(interp->input1_buffer, 0, input1_size);
        memset(interp->output0_buffer, 0, output_size);
        memset(interp->output1_buffer, 0, output1_size);
    }
    
    interp->arena_used = offset + 256;  // Extra padding
    interp->allocated = true;
    
    return TFLM_OK;
}

float* tflm_get_input_data(TFLMInterpreter interpreter, int input_index) {
    if (!interpreter) {
        return NULL;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    if (!interp->allocated) {
        return NULL;
    }
    
    if (input_index == 0) {
        return interp->input0_buffer;
    } else if (input_index == 1) {
        return interp->input1_buffer;  // May be NULL for backbone
    }
    
    return NULL;
}

float* tflm_get_output_data(TFLMInterpreter interpreter, int output_index) {
    if (!interpreter) {
        return NULL;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    if (!interp->allocated) {
        return NULL;
    }
    
    if (output_index == 0) {
        return interp->output0_buffer;
    } else if (output_index == 1) {
        return interp->output1_buffer;  // May be NULL for backbone
    }
    
    return NULL;
}

TFLMStatus tflm_invoke(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return TFLM_INVALID_ARGUMENT;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    if (!interp->allocated) {
        return TFLM_ERROR;
    }
    
    if (interp->is_backbone) {
        // Backbone inference: [1, 18, 40] -> [1, 18, 32]
        // Simple feature extraction simulation
        for (int t = 0; t < 18; t++) {  // For each time frame
            for (int f = 0; f < 32; f++) {  // For each output feature
                float sum = 0.0f;
                // Sample input features for this output
                for (int i = 0; i < 40; i++) {
                    int input_idx = t * 40 + i;
                    sum += interp->input0_buffer[input_idx] * 0.1f;  // Simple weighting
                }
                int output_idx = t * 32 + f;
                interp->output0_buffer[output_idx] = sum / 40.0f + (float)f * 0.01f;  // Add feature bias
            }
        }
    } else {
        // Streaming inference: 2 inputs [features + hidden] -> 2 outputs [logits + new_hidden]
        // Simulate GRU processing with hidden state
        
        // Process features to logits
        for (int i = 0; i < NN_NUM_CLASSES; i++) {
            float feat_val = (i < 32) ? interp->input0_buffer[i] : 0.0f;  // Features input
            interp->output0_buffer[i] = feat_val * 0.8f + 0.1f * (float)i / NN_NUM_CLASSES;  // Logits output
        }
        
        // Update hidden state
        for (int h = 0; h < NN_GRU_HIDDEN_DIM; h++) {
            float hidden_val = interp->input1_buffer ? interp->input1_buffer[h] : 0.0f;  // Hidden input
            interp->output1_buffer[h] = 0.9f * hidden_val + 0.01f;  // New hidden output (dummy update)
        }
    }
    
    return TFLM_OK;
}

int tflm_get_input_dims(TFLMInterpreter interpreter, int input_index, int* dims, int max_dims) {
    if (!interpreter || input_index != 0 || !dims) {
        return -1;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    
    if (interp->is_backbone) {
        // Backbone: [1, 18, 40]
        if (max_dims >= 3) {
            dims[0] = 1;   // batch
            dims[1] = 18;  // time
            dims[2] = 40;  // features
            return 3;
        }
    } else {
        // Streaming: [1, 32]
        if (max_dims >= 2) {
            dims[0] = 1;   // batch
            dims[1] = 32;  // features
            return 2;
        }
    }
    return -1;
}

int tflm_get_output_dims(TFLMInterpreter interpreter, int output_index, int* dims, int max_dims) {
    if (!interpreter || output_index != 0 || !dims) {
        return -1;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    
    if (interp->is_backbone) {
        // Backbone: [1, 18, 32]
        if (max_dims >= 3) {
            dims[0] = 1;   // batch
            dims[1] = 18;  // time
            dims[2] = 32;  // features
            return 3;
        }
    } else {
        // Streaming: [1, 32]
        if (max_dims >= 2) {
            dims[0] = 1;   // batch
            dims[1] = 32;  // features (logits)
            return 2;
        }
    }
    return -1;
}

size_t tflm_get_arena_used_bytes(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return 0;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    return interp->allocated ? interp->arena_used : 0;
}

void tflm_destroy_interpreter(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return;
    }
    
    // Find and free interpreter slot in static pool
    for (int i = 0; i < MAX_INTERPRETERS; i++) {
        if (&interpreter_pool[i] == (InterpreterStub*)interpreter) {
            interpreter_used[i] = false;
            // Clear the interpreter data
            memset(&interpreter_pool[i], 0, sizeof(InterpreterStub));
            break;
        }
    }
}

void tflm_destroy_model(TFLMModel model) {
    // Model is just a pointer to data, no cleanup needed for static allocation
    (void)model;  // Suppress unused parameter warning
}