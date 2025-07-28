/*
 * nn_model.h
 *
 *  Created on: 25 May 2024
 *      Author: leonardomannini
 *
 *  Public API for the Neural Network inference engine.
 */

#ifndef INC_NN_NN_MODEL_H_
#define INC_NN_NN_MODEL_H_

#include <stdint.h>
#include <stdbool.h>
#include "nn_config.h"

/* Result enumeration for API functions */
typedef enum {
    NN_SUCCESS = 0,
    NN_ERROR_INIT_FAILED,
    NN_ERROR_MEMORY_INSUFFICIENT,
    NN_ERROR_MODEL_INVALID,
    NN_ERROR_INFERENCE_FAILED,
    NN_ERROR_FEATURE_EXTRACTION_FAILED
} NN_Result_t;

/* Structure to hold a classification decision */
typedef struct {
    uint8_t predicted_class;
    float confidence;
    float logits[NN_NUM_CLASSES];
    uint32_t processing_time_ms;
    uint32_t arena_used_bytes_cnn;
    uint32_t arena_used_bytes_lstm;
} NN_Decision_t;

/**
 * @brief Initializes the entire NN pipeline.
 *
 * This function must be called once at startup. It sets up TFLite Micro,
 * allocates memory for arenas, loads the models, and prepares the
 * spectral processing module.
 *
 * @return NN_SUCCESS on success, or an error code otherwise.
 */
NN_Result_t NN_Init(void);

/**
 * @brief De-initializes the NN pipeline and frees resources.
 */
void NN_Deinit(void);

/**
 * @brief Resets the state of the streaming inference.
 *
 * This should be called before starting a new, independent audio clip analysis.
 * It clears the LSTM hidden state (h,c) and any accumulated features or logits.
 *
 * @return NN_SUCCESS on success.
 */
NN_Result_t NN_ResetStreamState(void);

/**
 * @brief Processes a window of audio samples.
 *
 * This is the main workhorse function. It takes a chunk of audio, performs
 * STFT and feature extraction, runs the CNN and LSTM models, and accumulates
 * results.
 *
 * @param audio_samples Pointer to an array of 16-bit audio samples.
 * @param num_samples The number of samples in the array.
 * @return NN_SUCCESS on success, or an error code otherwise.
 */
NN_Result_t NN_ProcessAudio(const int16_t* audio_samples, uint32_t num_samples);

/**
 * @brief Checks if a new classification decision is ready.
 *
 * A decision is typically ready after processing a full clip duration
 * (e.g., 3 seconds).
 *
 * @return true if a new decision is available, false otherwise.
 */
bool NN_HasNewDecision(void);

/**
 * @brief Retrieves the last computed classification decision.
 *
 * This function should be called after NN_HasNewDecision() returns true.
 *
 * @param decision Pointer to an NN_Decision_t struct to be filled.
 * @return NN_SUCCESS if a decision was retrieved, NN_ERROR_INFERENCE_FAILED otherwise.
 */
NN_Result_t NN_GetLastDecision(NN_Decision_t* decision);

/**
 * @brief Retrieves the names of the classes.
 *
 * @return A constant array of constant strings representing the class names.
 */
const char* const* NN_GetClassNames(void);

/* --- Debug and Monitoring Functions --- */

/**
 * @brief Gets the peak memory usage of the CNN tensor arena.
 * @return Peak bytes used in the CNN arena.
 */
uint32_t NN_GetCNNArenaUsedBytes(void);

/**
 * @brief Gets the peak memory usage of the LSTM tensor arena.
 * @return Peak bytes used in the LSTM arena.
 */
uint32_t NN_GetLSTMArenaUsedBytes(void);

/**
 * @brief Estimates the amount of free RAM.
 * This is a rough estimate based on the current stack pointer.
 * @return Estimated free bytes.
 */
uint32_t NN_GetFreeRAM(void);

#endif /* INC_NN_NN_MODEL_H_ */
