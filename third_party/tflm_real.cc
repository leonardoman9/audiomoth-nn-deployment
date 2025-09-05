/* TensorFlow Lite Micro Real Implementation
 * 
 * This replaces tflm_wrapper.c with actual TFLM functionality
 * while maintaining the same C API for compatibility with existing code.
 */

#include "tflm_real.h"
#include "audiomoth_op_resolver.h"

// TFLM includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

// For error reporting  
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include <cstring>

// Internal structures (hidden from C interface)
struct TFLMModelImpl {
    const tflite::Model* model;
    bool valid;
};

struct TFLMInterpreterImpl {
    tflite::MicroInterpreter* interpreter;
    tflite::AudioMothOpResolver* resolver;
    tflite::MicroErrorReporter* error_reporter;
    uint8_t* arena;
    size_t arena_size;
    bool allocated;
    bool valid;
};

// Maximum number of interpreters (backbone + streaming)
#define MAX_INTERPRETERS 2
static TFLMInterpreterImpl g_interpreters[MAX_INTERPRETERS];
static bool g_interpreters_used[MAX_INTERPRETERS] = {false, false};

// Helper function to find free interpreter slot
static TFLMInterpreterImpl* find_free_interpreter() {
    for (int i = 0; i < MAX_INTERPRETERS; i++) {
        if (!g_interpreters_used[i]) {
            g_interpreters_used[i] = true;
            memset(&g_interpreters[i], 0, sizeof(TFLMInterpreterImpl));
            return &g_interpreters[i];
        }
    }
    return nullptr;
}

// Helper function to free interpreter slot
static void free_interpreter_slot(TFLMInterpreterImpl* impl) {
    for (int i = 0; i < MAX_INTERPRETERS; i++) {
        if (&g_interpreters[i] == impl) {
            g_interpreters_used[i] = false;
            break;
        }
    }
}

extern "C" {

TFLMModel tflm_create_model(const unsigned char* model_data, unsigned int model_size) {
    if (!model_data || model_size == 0) {
        return nullptr;
    }
    
    // Verify model integrity
    if (!tflite::GetModel(model_data)) {
        MicroPrintf("Failed to parse model");
        return nullptr;
    }
    
    TFLMModelImpl* impl = new TFLMModelImpl();
    impl->model = tflite::GetModel(model_data);
    impl->valid = true;
    
    // Verify schema version
    if (impl->model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported version %d",
                    impl->model->version(), TFLITE_SCHEMA_VERSION);
        delete impl;
        return nullptr;
    }
    
    return static_cast<TFLMModel>(impl);
}

void tflm_destroy_model(TFLMModel model) {
    if (model) {
        TFLMModelImpl* impl = static_cast<TFLMModelImpl*>(model);
        delete impl;
    }
}

TFLMInterpreter tflm_create_interpreter(TFLMModel model, unsigned char* arena, unsigned int arena_size) {
    if (!model || !arena || arena_size < 1024) {
        return nullptr;
    }
    
    TFLMModelImpl* model_impl = static_cast<TFLMModelImpl*>(model);
    if (!model_impl->valid) {
        return nullptr;
    }
    
    TFLMInterpreterImpl* impl = find_free_interpreter();
    if (!impl) {
        MicroPrintf("No free interpreter slots available");
        return nullptr;
    }
    
    // Create error reporter
    impl->error_reporter = new tflite::MicroErrorReporter();
    
    // Create op resolver
    impl->resolver = new tflite::AudioMothOpResolver();
    
    // Store arena info
    impl->arena = arena;
    impl->arena_size = arena_size;
    
    // Create interpreter
    impl->interpreter = new tflite::MicroInterpreter(
        model_impl->model, 
        *impl->resolver, 
        arena, 
        arena_size, 
        impl->error_reporter
    );
    
    impl->allocated = false;
    impl->valid = true;
    
    return static_cast<TFLMInterpreter>(impl);
}

void tflm_destroy_interpreter(TFLMInterpreter interpreter) {
    if (interpreter) {
        TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
        
        delete impl->interpreter;
        delete impl->resolver;
        delete impl->error_reporter;
        
        free_interpreter_slot(impl);
    }
}

TFLMStatus tflm_allocate_tensors(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return TFLM_ERROR;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid) {
        return TFLM_ERROR;
    }
    
    TfLiteStatus status = impl->interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return TFLM_ERROR;
    }
    
    impl->allocated = true;
    return TFLM_OK;
}

TFLMStatus tflm_invoke(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return TFLM_ERROR;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid || !impl->allocated) {
        MicroPrintf("Interpreter not ready for inference");
        return TFLM_ERROR;
    }
    
    TfLiteStatus status = impl->interpreter->Invoke();
    if (status != kTfLiteOk) {
        MicroPrintf("Invoke() failed");
        return TFLM_ERROR;
    }
    
    return TFLM_OK;
}

float* tflm_get_input_tensor(TFLMInterpreter interpreter, int index) {
    if (!interpreter) {
        return nullptr;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid || !impl->allocated) {
        return nullptr;
    }
    
    TfLiteTensor* tensor = impl->interpreter->input(index);
    if (!tensor) {
        return nullptr;
    }
    
    // Support both float32 and int8 tensors
    if (tensor->type == kTfLiteFloat32) {
        return tensor->data.f;
    } else if (tensor->type == kTfLiteInt8) {
        // For int8, we'll need quantization/dequantization
        // For now, return the raw pointer (caller must handle conversion)
        return reinterpret_cast<float*>(tensor->data.int8);
    }
    
    return nullptr;
}

float* tflm_get_output_tensor(TFLMInterpreter interpreter, int index) {
    if (!interpreter) {
        return nullptr;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid || !impl->allocated) {
        return nullptr;
    }
    
    TfLiteTensor* tensor = impl->interpreter->output(index);
    if (!tensor) {
        return nullptr;
    }
    
    // Support both float32 and int8 tensors
    if (tensor->type == kTfLiteFloat32) {
        return tensor->data.f;
    } else if (tensor->type == kTfLiteInt8) {
        // For int8, we'll need quantization/dequantization
        return reinterpret_cast<float*>(tensor->data.int8);
    }
    
    return nullptr;
}

int tflm_get_input_tensor_count(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return 0;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid) {
        return 0;
    }
    
    return impl->interpreter->inputs_size();
}

int tflm_get_output_tensor_count(TFLMInterpreter interpreter) {
    if (!interpreter) {
        return 0;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid) {
        return 0;
    }
    
    return impl->interpreter->outputs_size();
}

void tflm_get_input_tensor_dims(TFLMInterpreter interpreter, int index, int* dims) {
    if (!interpreter || !dims) {
        return;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid || !impl->allocated) {
        return;
    }
    
    TfLiteTensor* tensor = impl->interpreter->input(index);
    if (!tensor || !tensor->dims) {
        return;
    }
    
    // Copy dimensions (assume dims array is large enough)
    for (int i = 0; i < tensor->dims->size && i < 4; i++) {
        dims[i] = tensor->dims->data[i];
    }
    
    // Fill remaining with 0
    for (int i = tensor->dims->size; i < 4; i++) {
        dims[i] = 0;
    }
}

void tflm_get_output_tensor_dims(TFLMInterpreter interpreter, int index, int* dims) {
    if (!interpreter || !dims) {
        return;
    }
    
    TFLMInterpreterImpl* impl = static_cast<TFLMInterpreterImpl*>(interpreter);
    if (!impl->valid || !impl->allocated) {
        return;
    }
    
    TfLiteTensor* tensor = impl->interpreter->output(index);
    if (!tensor || !tensor->dims) {
        return;
    }
    
    // Copy dimensions (assume dims array is large enough)
    for (int i = 0; i < tensor->dims->size && i < 4; i++) {
        dims[i] = tensor->dims->data[i];
    }
    
    // Fill remaining with 0
    for (int i = tensor->dims->size; i < 4; i++) {
        dims[i] = 0;
    }
}

} // extern "C"