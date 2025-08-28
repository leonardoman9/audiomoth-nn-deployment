#include "tflm_wrapper.h"
#include <string.h>
#include "../inc/audiomoth.h"  // For LED debug functions

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
    float* input_buffer;
    float* output_buffer;
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
    interpreter->input_buffer = NULL;
    interpreter->output_buffer = NULL;
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
    interp->input_buffer = (float*)interp->arena;
    interp->output_buffer = (float*)(interp->arena + input_size);
    interp->arena_used = total_needed;
    interp->allocated = true;
    
    // Initialize buffers to zero
    memset(interp->input_buffer, 0, input_size);
    memset(interp->output_buffer, 0, output_size);
    
    return TFLM_OK;
}

float* tflm_get_input_data(TFLMInterpreter interpreter, int input_index) {
    if (!interpreter || input_index != 0) {
        return NULL;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    return interp->allocated ? interp->input_buffer : NULL;
}

float* tflm_get_output_data(TFLMInterpreter interpreter, int output_index) {
    if (!interpreter || output_index != 0) {
        return NULL;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    return interp->allocated ? interp->output_buffer : NULL;
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
                    sum += interp->input_buffer[input_idx] * 0.1f;  // Simple weighting
                }
                int output_idx = t * 32 + f;
                interp->output_buffer[output_idx] = sum / 40.0f + (float)f * 0.01f;  // Add feature bias
            }
        }
    } else {
        // Streaming inference: [1, 32] -> [1, 32]
        // Simulate RNN/GRU processing
        for (int i = 0; i < 32; i++) {
            // Simple transformation with some non-linearity simulation
            float val = interp->input_buffer[i];
            interp->output_buffer[i] = val * 0.8f + 0.1f * (float)i / 32.0f;  // Feature-dependent processing
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

unsigned int tflm_get_arena_used_bytes(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return 0;
    }
    
    InterpreterStub* interp = (InterpreterStub*)interpreter;
    return interp->arena_used;
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