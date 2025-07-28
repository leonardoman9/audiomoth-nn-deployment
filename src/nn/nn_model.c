/*
 * nn_model.c
 *
 *  Created on: 25 May 2024
 *      Author: leonardomannini
 *
 *  Stub implementation of the Neural Network inference API.
 *  This file provides the basic structure and allows the project to compile
 *  before the full TFLite Micro logic is integrated.
 */

#include "nn_model.h"

// TODO: Integrate TFLite Micro headers
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// ...

/* --- Private Variables --- */

// A placeholder for the last decision
static NN_Decision_t last_decision;
static bool new_decision_available = false;

// Names of the classes the model is trained on
static const char* const CLASS_NAMES[NN_NUM_CLASSES] = {
    "Ardea_cinerea",
    "Corvus_corax",
    "Falco_tinnunculus",
    "Hirundo_rustica",
    "Otus_scops",
    "Phylloscopus_collybita",
    "Picus_canus",
    "Strix_aluco",
    "no_bird"
};


/* --- Public API Implementation --- */

NN_Result_t NN_Init(void) {
    // TODO: Initialize TFLM interpreters, models, and arenas
    return NN_SUCCESS;
}

void NN_Deinit(void) {
    // TODO: Free any allocated resources
}

NN_Result_t NN_ResetStreamState(void) {
    // TODO: Reset LSTM state tensors (h, c)
    new_decision_available = false;
    return NN_SUCCESS;
}

NN_Result_t NN_ProcessAudio(const int16_t* audio_samples, uint32_t num_samples) {
    // TODO:
    // 1. Accumulate audio in a ring buffer.
    // 2. When enough samples are available, run feature extraction.
    // 3. Run CNN inference.
    // 4. Run LSTM inference with state management.
    // 5. Accumulate logits.
    // 6. If decision interval is reached, call FinalizeDecision().
    return NN_SUCCESS;
}

bool NN_HasNewDecision(void) {
    return new_decision_available;
}

NN_Result_t NN_GetLastDecision(NN_Decision_t* decision) {
    if (!new_decision_available) {
        return NN_ERROR_INFERENCE_FAILED;
    }
    *decision = last_decision;
    new_decision_available = false; // Consume the decision
    return NN_SUCCESS;
}

const char* const* NN_GetClassNames(void) {
    return CLASS_NAMES;
}

/* --- Debug and Monitoring Functions --- */

uint32_t NN_GetCNNArenaUsedBytes(void) {
    // TODO: Return interpreter->arena_used_bytes() for CNN
    return 0;
}

uint32_t NN_GetLSTMArenaUsedBytes(void) {
    // TODO: Return interpreter->arena_used_bytes() for LSTM
    return 0;
}

uint32_t NN_GetFreeRAM(void) {
    // A rough, platform-specific estimation.
    // For EFM32, we could check the stack pointer against the end of RAM.
    volatile uint8_t stack_dummy;
    extern uint8_t_estack; // From linker script
    return &stack_dummy - &_estack;
} 