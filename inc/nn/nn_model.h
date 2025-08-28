#ifndef NN_MODEL_H
#define NN_MODEL_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "nn_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for TensorFlow Lite Micro types
typedef struct TfLiteTensor TfLiteTensor;
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreter TfLiteInterpreter;

// Neural Network State
typedef enum {
    NN_STATE_UNINITIALIZED = 0,
    NN_STATE_INITIALIZED,
    NN_STATE_READY,
    NN_STATE_ERROR
} NN_State_t;

// Detection Result
typedef struct {
    uint8_t class_id;
    float confidence;
    uint32_t timestamp_ms;
    bool valid;
} NN_Detection_t;

// Decision Result (can contain multiple detections)
typedef struct {
    NN_Detection_t detections[NN_MAX_DETECTIONS_PER_SEC];
    uint8_t num_detections;
    uint32_t frame_id;
} NN_Decision_t;

// External model data (defined in model_data.c files)
extern const unsigned char backbone_model_data[];
extern const unsigned int backbone_model_data_len;
extern const unsigned char streaming_model_data[];
extern const unsigned int streaming_model_data_len;

// Public API Functions

/**
 * Initialize the neural network system
 * @return true if successful, false otherwise
 */
bool NN_Init(void);

/**
 * Deinitialize the neural network system
 */
void NN_Deinit(void);

/**
 * Reset the streaming state (call at start of new recording)
 */
void NN_ResetStreamState(void);

/**
 * Process audio data through the neural network
 * @param audio_data Pointer to audio samples (16-bit PCM)
 * @param num_samples Number of samples in the buffer
 * @param decision Output decision structure
 * @return true if processing successful, false otherwise
 */
bool NN_ProcessAudio(const int16_t* audio_data, uint32_t num_samples, NN_Decision_t* decision);

/**
 * Get current neural network state
 * @return Current state
 */
NN_State_t NN_GetState(void);

/**
 * Get memory usage statistics
 */
uint32_t NN_GetBackboneArenaUsedBytes(void);
uint32_t NN_GetStreamingArenaUsedBytes(void);

/**
 * Get inference timing statistics (in microseconds)
 */
uint32_t NN_GetLastInferenceTime(void);

/**
 * Run performance test sequence: LED → 10 inferences → pause → 100 → pause → 1000 → LED
 */
void NN_runPerformanceTestSequence(const int16_t* audio_data, uint32_t num_samples);

/**
 * Set seed for dummy data generation (for reproducible testing)
 */
void NN_setDummySeed(uint32_t seed);

/**
 * Flash-based memory swap functions for larger models
 */
bool NN_FlashSwap_Init(void);
int NN_FlashSwap_AllocateSlot(uint32_t size);
bool NN_FlashSwap_StoreTensor(int slot_id, const void* data, uint32_t size);
bool NN_FlashSwap_LoadTensor(int slot_id, void* data, uint32_t size);
void NN_FlashSwap_FreeSlot(int slot_id);

/**
 * Swap arena management for large model support
 */
bool NN_SwapArena_Init(void);
uint8_t* NN_SwapArena_GetBuffer(int tensor_id, uint32_t size);
void NN_SwapArena_MarkDirty(void);

#ifdef __cplusplus
}
#endif

#endif // NN_MODEL_H