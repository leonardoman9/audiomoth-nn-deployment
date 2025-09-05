#include "tflm_wrapper.h"
#include <cstdint>
#include <cstddef>
#include <new>

// Include i veri header di TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <cstdio>

// Include aggiuntivi per i kernel che useremo
#include "tensorflow/lite/micro/kernels/micro_ops.h"

// Uso del namespace tflite per semplificare il codice
using namespace tflite;

// Questa è la nostra implementazione nascosta. Il codice C vede solo una dichiarazione incompleta.
struct TFLM_Interpreter_s {
    const Model* model = nullptr;
    MicroInterpreter* interpreter = nullptr;
    MicroMutableOpResolver<10>* resolver = nullptr; // Resolver per max 10 operatori
    bool initialized = false;
};

// --- Implementazione delle funzioni del wrapper ---

TFLM_Interpreter_t* tflm_create_interpreter(
    const unsigned char* model_data,
    size_t model_size,
    uint8_t* arena,
    size_t arena_size) 
{
    if (!model_data || !arena || model_size == 0 || arena_size == 0) {
        printf("TFLM: Argomenti non validi.\n");
        return nullptr;
    }

    // 1. Carica il modello TFLite
    const Model* model = GetModel(model_data);
    if (!model) {
        printf("TFLM: Fallimento nel caricare il modello.\n");
        return nullptr;
    }
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("TFLM: Versione schema modello %d non supportata. Richiesta %d.\n",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return nullptr;
    }

    // 2. Alloca la nostra struttura wrapper
    TFLM_Interpreter_t* interp_wrapper = new (std::nothrow) TFLM_Interpreter_t();
    if (!interp_wrapper) {
        printf("TFLM: Fallimento nell'allocare il wrapper dell'interprete.\n");
        return nullptr;
    }
    interp_wrapper->model = model;

    // 3. Crea il resolver (usiamo MicroMutableOpResolver per produzione)
    interp_wrapper->resolver = new (std::nothrow) MicroMutableOpResolver<10>();
     if (!interp_wrapper->resolver) {
        printf("TFLM: Fallimento nell'allocare MicroMutableOpResolver.\n");
        delete interp_wrapper;
        return nullptr;
    }
    
    // Aggiungi operatori comuni per modelli CNN/RNN
    interp_wrapper->resolver->AddConv2D();
    interp_wrapper->resolver->AddFullyConnected();
    interp_wrapper->resolver->AddSoftmax();
    interp_wrapper->resolver->AddReshape();
    interp_wrapper->resolver->AddAdd();
    interp_wrapper->resolver->AddMul();
    interp_wrapper->resolver->AddRelu();

    // 4. Crea il vero interprete TFLM
    interp_wrapper->interpreter = new (std::nothrow) MicroInterpreter(
        interp_wrapper->model, 
        *(interp_wrapper->resolver), 
        arena, 
        arena_size);

    if (!interp_wrapper->interpreter) {
        printf("TFLM: Fallimento nell'allocare MicroInterpreter.\n");
        delete interp_wrapper->resolver;
        delete interp_wrapper;
        return nullptr;
    }

    // 5. Alloca i tensori
    if (interp_wrapper->interpreter->AllocateTensors() != kTfLiteOk) {
        printf("TFLM: Fallimento nell'allocare i tensori.\n");
        // Facciamo pulizia completa in caso di fallimento
        delete interp_wrapper->interpreter;
        delete interp_wrapper->resolver;
        delete interp_wrapper;
        return nullptr;
    }

    interp_wrapper->initialized = true;
    printf("TFLM: Interprete creato con successo.\n");
    return interp_wrapper;
}

void tflm_destroy_interpreter(TFLM_Interpreter_t* interpreter) {
    if (!interpreter) return;
    
    // Dealloca tutte le risorse C++ nell'ordine inverso di creazione
    delete interpreter->interpreter;
    delete interpreter->resolver;
    delete interpreter;
}

bool tflm_invoke(TFLM_Interpreter_t* interpreter) {
    if (!interpreter || !interpreter->initialized) return false;

    return (interpreter->interpreter->Invoke() == kTfLiteOk);
}

TfLiteTensor* tflm_get_input_tensor(TFLM_Interpreter_t* interpreter, int index) {
    if (!interpreter || !interpreter->initialized) return nullptr;
    return interpreter->interpreter->input(index);
}

TfLiteTensor* tflm_get_output_tensor(TFLM_Interpreter_t* interpreter, int index) {
    if (!interpreter || !interpreter->initialized) return nullptr;
    return interpreter->interpreter->output(index);
}

uint32_t tflm_get_arena_used_bytes(TFLM_Interpreter_t* interpreter) {
    if (!interpreter || !interpreter->initialized) return 0;
    return interpreter->interpreter->arena_used_bytes();
}