#include "../../inc/nn/nn_model.h"
#include "../../third_party/tflm_wrapper.h"
#include "../../inc/audiomoth.h"  // For LED functions
#include "tensor_arena.h"  // External SRAM tensor arena
#include <string.h>
#include <math.h>
#include <stdlib.h>  // For malloc/free

// External SRAM tensor arena - replaces old Flash virtual arena

// Include external model data arrays
extern const unsigned char backbone_model_data[];
extern const unsigned int backbone_model_data_len;
extern const unsigned char streaming_model_data[];
extern const unsigned int streaming_model_data_len;

// Global state variables
static NN_State_t g_nn_state = NN_STATE_UNINITIALIZED;

// TensorFlow Lite Micro components
static TFLMModel g_backbone_model = NULL;
static TFLMModel g_streaming_model = NULL;
static TFLMInterpreter g_backbone_interpreter = NULL;
static TFLMInterpreter g_streaming_interpreter = NULL;

// Memory arenas - use external SRAM via tensor arena
static uint8_t* g_backbone_arena = NULL;   // Will point to external SRAM
static uint8_t* g_streaming_arena = NULL;  // Will point to external SRAM

// Streaming state - dynamically allocated for GRU-64
static float* g_gru_hidden_state = NULL;  // Will malloc for GRU-64
static uint32_t g_frame_counter = 0;
static uint32_t g_last_inference_time_us = 0;

// Dynamic buffers allocated from external SRAM to avoid BSS bloat
static float* g_spectrogram_buffer = NULL;
static float* g_backbone_features_buffer = NULL;
static float* g_accumulated_logits_buffer = NULL;
static float* g_step_logits_buffer = NULL;
static float* g_new_hidden_state_buffer = NULL;
static float* g_att_scores_buffer = NULL;

// Simple PRNG for dummy data generation
static uint32_t g_dummy_seed = 12345;

// Class names (placeholder - update with real bird species)
const char* NN_CLASS_NAMES[NN_NUM_CLASSES] = {
    "Background",
    "Bird_Species_1",
    "Bird_Species_2",
    "Bird_Species_3",
    "Bird_Species_4",
    "Bird_Species_5",
    "Bird_Species_6",
    "Bird_Species_7",
    "Bird_Species_8",
    "Bird_Species_9"
};

// Private function declarations
static bool initialize_models(void);
static void reset_gru_state(void);
static bool preprocess_audio_to_spectrogram(const int16_t* audio_data, uint32_t num_samples, float* spectrogram);
static bool run_backbone_inference(const float* spectrogram, float* features);
static bool run_streaming_inference(const float* features, float* logits, float* new_hidden_state);
static void apply_softmax(float* logits, int num_classes);
static bool finalize_decision(const float* logits, NN_Decision_t* decision);
static uint32_t get_timestamp_ms(void);

// Public API Implementation

bool NN_Init(void) {
    if (g_nn_state != NN_STATE_UNINITIALIZED) {
        return false; // Already initialized
    }
    
    // DEBUG: Signal entering NN_Init (BREAKPOINT: Line 66)
    AudioMoth_setRedLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(50);
    AudioMoth_setRedLED(false);
    
    // FLASH SWAP APPROACH: Initialize Flash-based virtual memory system
    AudioMoth_setGreenLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(25);
    AudioMoth_setGreenLED(false);
    
    // DEBUG: Signal External SRAM initialization
    AudioMoth_setGreenLED(true);
    AudioMoth_setRedLED(true);  // Both LEDs = External SRAM mode
    // AudioMoth_delay - REMOVED to avoid timer hang(100);
    AudioMoth_setGreenLED(false);
    AudioMoth_setRedLED(false);
    
    // Initialize external SRAM tensor arena
    if (!TensorArena_Init()) {
        // DEBUG: External SRAM init failed - 10 red blinks
        for (int i = 0; i < 10; i++) {
            AudioMoth_setRedLED(true);
            // AudioMoth_delay - REMOVED to avoid timer hang(150);
            AudioMoth_setRedLED(false);
            // AudioMoth_delay - REMOVED to avoid timer hang(150);
        }
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
    
    // Get external SRAM buffer and partition it
    uint8_t* arena_buffer = TensorArena_GetBuffer();
    uint32_t arena_size = TensorArena_GetSize();
    
    if (!arena_buffer || arena_size < (NN_BACKBONE_ARENA_SIZE + NN_STREAMING_ARENA_SIZE)) {
        // Not enough external SRAM - signal error
        for (int i = 0; i < 5; i++) {
            AudioMoth_setRedLED(true);
            // AudioMoth_delay - REMOVED to avoid timer hang(50);
            AudioMoth_setRedLED(false);
            // AudioMoth_delay - REMOVED to avoid timer hang(50);
        }
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
    
    // Partition external SRAM: first part for backbone, second for streaming
    g_backbone_arena = arena_buffer;
    g_streaming_arena = arena_buffer + NN_BACKBONE_ARENA_SIZE;
    
    // Allocate working buffers from internal RAM (malloc) to avoid BSS bloat
    g_gru_hidden_state = (float*)malloc(NN_GRU_HIDDEN_DIM * sizeof(float));
    g_spectrogram_buffer = (float*)malloc(NN_INPUT_HEIGHT * NN_INPUT_WIDTH * sizeof(float));
    g_backbone_features_buffer = (float*)malloc(NN_BACKBONE_FEATURES * NN_TIME_FRAMES * sizeof(float));
    g_accumulated_logits_buffer = (float*)malloc(NN_NUM_CLASSES * sizeof(float));
    g_step_logits_buffer = (float*)malloc(NN_NUM_CLASSES * sizeof(float));
    g_new_hidden_state_buffer = (float*)malloc(NN_GRU_HIDDEN_DIM * sizeof(float));
    g_att_scores_buffer = (float*)malloc(NN_TIME_FRAMES * sizeof(float));
    
    if (!g_gru_hidden_state || !g_spectrogram_buffer || !g_backbone_features_buffer ||
        !g_accumulated_logits_buffer || !g_step_logits_buffer || !g_new_hidden_state_buffer || !g_att_scores_buffer) {
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
    
    // Clear allocated memory
    memset(g_backbone_arena, 0, NN_BACKBONE_ARENA_SIZE);
    memset(g_streaming_arena, 0, NN_STREAMING_ARENA_SIZE);
    memset(g_gru_hidden_state, 0, NN_GRU_HIDDEN_DIM * sizeof(float));
    memset(g_spectrogram_buffer, 0, NN_INPUT_HEIGHT * NN_INPUT_WIDTH * sizeof(float));
    memset(g_backbone_features_buffer, 0, NN_BACKBONE_FEATURES * NN_TIME_FRAMES * sizeof(float));
    memset(g_accumulated_logits_buffer, 0, NN_NUM_CLASSES * sizeof(float));
    memset(g_step_logits_buffer, 0, NN_NUM_CLASSES * sizeof(float));
    memset(g_new_hidden_state_buffer, 0, NN_GRU_HIDDEN_DIM * sizeof(float));
    memset(g_att_scores_buffer, 0, NN_TIME_FRAMES * sizeof(float));
    
    // DEBUG: External SRAM initialized - alternating green/red
    for (int i = 0; i < 3; i++) {
        AudioMoth_setGreenLED(true);
        for (volatile int j = 0; j < 200000; j++);  // Busy wait
        AudioMoth_setGreenLED(false);
        AudioMoth_setRedLED(true);
        for (volatile int j = 0; j < 200000; j++);  // Busy wait
        AudioMoth_setRedLED(false);
    }
    
    // DEBUG: Arena diagnostics - print arena addresses for verification
    TensorArenaStats_t arena_stats;
    TensorArena_GetStats(&arena_stats);
    
    // Signal arena type: Green = External SRAM, Red = Error
    if (arena_stats.is_external_sram) {
        // Success: 3 quick green flashes
        for (int i = 0; i < 3; i++) {
            AudioMoth_setGreenLED(true);
            for (volatile int j = 0; j < 100000; j++);  // Busy wait
            AudioMoth_setGreenLED(false);
            for (volatile int j = 0; j < 100000; j++);  // Busy wait
        }
    } else {
        // Warning: Arena not in external SRAM - 3 red flashes
        for (int i = 0; i < 3; i++) {
            AudioMoth_setRedLED(true);
            for (volatile int j = 0; j < 100000; j++);  // Busy wait
            AudioMoth_setRedLED(false);
            for (volatile int j = 0; j < 100000; j++);  // Busy wait
        }
    }
    
    // Initialize models
    if (!initialize_models()) {
        // DEBUG: Signal model initialization failed
        for (int i = 0; i < 5; i++) {
            AudioMoth_setRedLED(true);
            // AudioMoth_delay - REMOVED to avoid timer hang(100);
            AudioMoth_setRedLED(false);
            // AudioMoth_delay - REMOVED to avoid timer hang(100);
        }
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
    
    // DEBUG: Models initialized successfully - 1 long green blink
    AudioMoth_setGreenLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(200);
    AudioMoth_setGreenLED(false);
    
    // Reset streaming state
    reset_gru_state();
    g_frame_counter = 0;
    
    g_nn_state = NN_STATE_READY;
    return true;
}

void NN_Deinit(void) {
    // Cleanup TensorFlow Lite Micro resources
    if (g_backbone_interpreter) {
        tflm_destroy_interpreter(g_backbone_interpreter);
        g_backbone_interpreter = NULL;
    }
    
    if (g_streaming_interpreter) {
        tflm_destroy_interpreter(g_streaming_interpreter);
        g_streaming_interpreter = NULL;
    }
    
    if (g_backbone_model) {
        tflm_destroy_model(g_backbone_model);
        g_backbone_model = NULL;
    }
    
    if (g_streaming_model) {
        tflm_destroy_model(g_streaming_model);
        g_streaming_model = NULL;
    }
    
    g_nn_state = NN_STATE_UNINITIALIZED;
}

void NN_ResetStreamState(void) {
    reset_gru_state();
    g_frame_counter = 0;
}

bool NN_ProcessAudio(const int16_t* audio_data, uint32_t num_samples, NN_Decision_t* decision) {
    if (g_nn_state != NN_STATE_READY || !audio_data || !decision) {
        return false;
    }
    
    // Initialize decision structure
    memset(decision, 0, sizeof(NN_Decision_t));
    decision->frame_id = g_frame_counter++;
    
    uint32_t start_time = get_timestamp_ms();
    
    // Step 1: Convert audio to spectrogram (using static buffer)
    if (!preprocess_audio_to_spectrogram(audio_data, num_samples, g_spectrogram_buffer)) {
        return false;
    }
    
    // Step 2: Run backbone CNN inference (using static buffer)
    if (!run_backbone_inference(g_spectrogram_buffer, g_backbone_features_buffer)) {
        return false;
    }
    
    // Step 3: Process full temporal sequence with streaming model
    // Backbone output shape: [time=NN_TIME_FRAMES, feat=NN_BACKBONE_FEATURES]
    // Run the streaming model for each timestep sequentially, carrying hidden state
    // Attention-like aggregation over timesteps using hidden-state energies (using static buffers)
    
    // Allocate history of per-timestep logits on heap to avoid stack bloat
    const int logits_hist_count = NN_TIME_FRAMES * NN_NUM_CLASSES;
    float* logits_history = (float*)malloc(sizeof(float) * logits_hist_count);
    if (logits_history == NULL) {
        return false;
    }
    
    for (int t = 0; t < NN_TIME_FRAMES; t++) {
        const float* timestep_features = &g_backbone_features_buffer[t * NN_BACKBONE_FEATURES];
        
        if (!run_streaming_inference(timestep_features, g_step_logits_buffer, g_new_hidden_state_buffer)) {
            free(logits_history);
            return false;
        }
        
        // Save logits_t
        memcpy(&logits_history[t * NN_NUM_CLASSES], g_step_logits_buffer, sizeof(float) * NN_NUM_CLASSES);
        
        // Compute a simple attention score from hidden state energy
        float sum_abs = 0.0f;
        for (int h = 0; h < NN_GRU_HIDDEN_DIM; h++) {
            float v = g_new_hidden_state_buffer[h];
            sum_abs += (v >= 0.0f ? v : -v);
        }
        g_att_scores_buffer[t] = sum_abs / (float)NN_GRU_HIDDEN_DIM;
        
        // Carry hidden state forward to next timestep
        memcpy(g_gru_hidden_state, g_new_hidden_state_buffer, NN_GRU_HIDDEN_DIM * sizeof(float));
    }
    
    // Softmax over attention scores to obtain temporal weights
    // Find max for numerical stability
    float max_score = g_att_scores_buffer[0];
    for (int t = 1; t < NN_TIME_FRAMES; t++) {
        if (g_att_scores_buffer[t] > max_score) max_score = g_att_scores_buffer[t];
    }
    float sum_exp = 0.0f;
    for (int t = 0; t < NN_TIME_FRAMES; t++) {
        g_att_scores_buffer[t] = expf(g_att_scores_buffer[t] - max_score);
        sum_exp += g_att_scores_buffer[t];
    }
    const float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < NN_TIME_FRAMES; t++) {
        g_att_scores_buffer[t] *= inv_sum; // now att_scores are alphas
    }
    
    // Weighted sum of logits over time using attention weights
    for (int c = 0; c < NN_NUM_CLASSES; c++) {
        g_accumulated_logits_buffer[c] = 0.0f;
    }
    for (int t = 0; t < NN_TIME_FRAMES; t++) {
        const float alpha = g_att_scores_buffer[t];
        const float* logits_t = &logits_history[t * NN_NUM_CLASSES];
        for (int c = 0; c < NN_NUM_CLASSES; c++) {
            g_accumulated_logits_buffer[c] += alpha * logits_t[c];
        }
    }
    
    free(logits_history);
    
    // Step 5: Apply softmax and generate decision
    apply_softmax(g_accumulated_logits_buffer, NN_NUM_CLASSES);
    if (!finalize_decision(g_accumulated_logits_buffer, decision)) {
        return false;
    }
    
    g_last_inference_time_us = (get_timestamp_ms() - start_time) * 1000;
    return true;
}

NN_State_t NN_GetState(void) {
    return g_nn_state;
}

uint32_t NN_GetBackboneArenaUsedBytes(void) {
    if (g_backbone_interpreter) {
        return tflm_get_arena_used_bytes(g_backbone_interpreter);
    }
    return 0;
}

uint32_t NN_GetStreamingArenaUsedBytes(void) {
    if (g_streaming_interpreter) {
        return tflm_get_arena_used_bytes(g_streaming_interpreter);
    }
    return 0;
}

uint32_t NN_GetLastInferenceTime(void) {
    return g_last_inference_time_us;
}

void NN_setDummySeed(uint32_t seed) {
    g_dummy_seed = seed;
}

void NN_runPerformanceTestSequence(const int16_t* audio_data, uint32_t num_samples) {
    if (g_nn_state != NN_STATE_READY || !audio_data) {
        return;
    }
    
    // Initialize random seed based on current time for varied dummy data
    g_dummy_seed = get_timestamp_ms() ^ 0xDEADBEEF;
    
    NN_Decision_t decision;
    
    // === 1 BLINK ROSSO ACCESO/SPENTO ===
    AudioMoth_setRedLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(500);
    AudioMoth_setRedLED(false);
    
    // === PAUSA 3 SECONDI ===
    // AudioMoth_delay - REMOVED to avoid timer hang(3000);
    
    // === TEST 10 INFERENZE ===
    for (int i = 0; i < 10; i++) {
        NN_ProcessAudio(audio_data, num_samples, &decision);
    }
    
    // === 1 BLINK VERDE ACCESO/SPENTO ===
    AudioMoth_setGreenLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(500);
    AudioMoth_setGreenLED(false);
    
    // === PAUSA 3 SECONDI ===
    // AudioMoth_delay - REMOVED to avoid timer hang(3000);
    
    // === TEST 100 INFERENZE ===
    for (int i = 0; i < 100; i++) {
        NN_ProcessAudio(audio_data, num_samples, &decision);
    }
    
    // === 1 BLINK VERDE ACCESO/SPENTO ===
    AudioMoth_setGreenLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(500);
    AudioMoth_setGreenLED(false);
    
    // === PAUSA 3 SECONDI ===
    // AudioMoth_delay - REMOVED to avoid timer hang(3000);
    
    // === TEST 1000 INFERENZE ===
    for (int i = 0; i < 1000; i++) {
        NN_ProcessAudio(audio_data, num_samples, &decision);
    }
    
    // === 1 BLINK ROSSO ACCESO/SPENTO ===
    AudioMoth_setRedLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(500);
    AudioMoth_setRedLED(false);
    
    // === BASTA! STOP TUTTO ===
    // Imposta flag per fermare il loop audio
    g_nn_state = NN_STATE_ERROR;
}

// Private function implementations

static bool initialize_models(void) {
    // DEBUG: Entering initialize_models
    AudioMoth_setGreenLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(100);
    AudioMoth_setGreenLED(false);
    // AudioMoth_delay - REMOVED to avoid timer hang(100);
    
    // Create backbone model
    g_backbone_model = tflm_create_model(backbone_model_data, backbone_model_data_len);
    if (!g_backbone_model) {
        // DEBUG: Backbone model creation failed - 1 red blink
        AudioMoth_setRedLED(true);
        // AudioMoth_delay - REMOVED to avoid timer hang(300);
        AudioMoth_setRedLED(false);
        return false;
    }
    
    // DEBUG: Backbone model created - 1 green blink
    AudioMoth_setGreenLED(true);
    // AudioMoth_delay - REMOVED to avoid timer hang(100);
    AudioMoth_setGreenLED(false);
    
    // Create backbone interpreter
    g_backbone_interpreter = tflm_create_interpreter(g_backbone_model, g_backbone_arena, NN_BACKBONE_ARENA_SIZE);
    if (!g_backbone_interpreter) {
        // DEBUG: Backbone interpreter creation failed - 2 red blinks
        for (int i = 0; i < 2; i++) {
            AudioMoth_setRedLED(true);
            // AudioMoth_delay - REMOVED to avoid timer hang(200);
            AudioMoth_setRedLED(false);
            // AudioMoth_delay - REMOVED to avoid timer hang(200);
        }
        return false;
    }
    
    // DEBUG: Backbone interpreter created - 2 green blinks
    for (int i = 0; i < 2; i++) {
        AudioMoth_setGreenLED(true);
        // AudioMoth_delay - REMOVED to avoid timer hang(100);
        AudioMoth_setGreenLED(false);
        // AudioMoth_delay - REMOVED to avoid timer hang(100);
    }
    
    // Allocate backbone tensors
    if (tflm_allocate_tensors(g_backbone_interpreter) != TFLM_OK) {
        // DEBUG: Backbone tensor allocation failed - 3 red blinks
        for (int i = 0; i < 3; i++) {
            AudioMoth_setRedLED(true);
            // AudioMoth_delay - REMOVED to avoid timer hang(200);
            AudioMoth_setRedLED(false);
            // AudioMoth_delay - REMOVED to avoid timer hang(200);
        }
        return false;
    }
    
    // DEBUG: Backbone tensors allocated - show arena usage via LED pattern
    size_t backbone_used = tflm_get_arena_used_bytes(g_backbone_interpreter);
    // Convert KB to LED blinks: 1KB = 1 blink, max 5 blinks
    int kb_used = (int)(backbone_used / 1024);
    int blinks = (kb_used > 5) ? 5 : kb_used;
    if (blinks == 0) blinks = 1;  // At least 1 blink for success
    
    for (int i = 0; i < blinks; i++) {
        AudioMoth_setGreenLED(true);
        // AudioMoth_delay - REMOVED to avoid timer hang(100);
        AudioMoth_setGreenLED(false);
        // AudioMoth_delay - REMOVED to avoid timer hang(100);
    }
    
    // Create streaming model
    g_streaming_model = tflm_create_model(streaming_model_data, streaming_model_data_len);
    if (!g_streaming_model) {
        return false;
    }
    
    // Create streaming interpreter
    g_streaming_interpreter = tflm_create_interpreter(g_streaming_model, g_streaming_arena, NN_STREAMING_ARENA_SIZE);
    if (!g_streaming_interpreter) {
        return false;
    }
    
    // Allocate streaming tensors
    if (tflm_allocate_tensors(g_streaming_interpreter) != TFLM_OK) {
        return false;
    }
    
    return true;
}

static void reset_gru_state(void) {
    memset(g_gru_hidden_state, 0, NN_GRU_HIDDEN_DIM * sizeof(float));
}

// Simple PRNG for dummy data generation (declaration moved to top)

static uint32_t dummy_rand(void) {
    // Simple Linear Congruential Generator
    g_dummy_seed = (g_dummy_seed * 1103515245 + 12345) & 0x7fffffff;
    return g_dummy_seed;
}

static bool preprocess_audio_to_spectrogram(const int16_t* audio_data, uint32_t num_samples, float* spectrogram) {
    // DUMMY DATA GENERATION FOR PERFORMANCE TESTING
    // Generate random spectrogram data instead of processing real audio
    
    (void)audio_data;   // Suppress unused parameter warning
    (void)num_samples;  // Suppress unused parameter warning
    
    // Generate random float values between 0.0 and 1.0 (typical spectrogram range)
    for (int i = 0; i < NN_INPUT_HEIGHT * NN_INPUT_WIDTH; i++) {
        uint32_t rand_val = dummy_rand();
        spectrogram[i] = (float)(rand_val % 1000) / 1000.0f;  // 0.0 - 1.0 range
    }
    
    return true;
}

/* 
// REAL AUDIO PROCESSING VERSION (for future deployment)
static bool preprocess_audio_to_spectrogram_REAL(const int16_t* audio_data, uint32_t num_samples, float* spectrogram) {
    // Real implementation would perform STFT and mel-scale conversion
    
    if (num_samples < NN_FRAME_SIZE) {
        return false;
    }
    
    // Create energy-based spectrogram from real audio
    const int samples_per_bin = num_samples / (NN_INPUT_HEIGHT * NN_INPUT_WIDTH);
    
    for (int freq = 0; freq < NN_INPUT_HEIGHT; freq++) {
        for (int time = 0; time < NN_INPUT_WIDTH; time++) {
            float energy = 0.0f;
            int start_idx = (freq * NN_INPUT_WIDTH + time) * samples_per_bin;
            
            // Compute energy in this time-frequency bin
            for (int i = 0; i < samples_per_bin && (start_idx + i) < num_samples; i++) {
                float sample = (float)audio_data[start_idx + i] / 32768.0f; // Normalize
                energy += sample * sample;
            }
            
            energy = sqrtf(energy / samples_per_bin); // RMS energy
            spectrogram[freq * NN_INPUT_WIDTH + time] = energy;
        }
    }
    
    return true;
}
*/

static bool run_backbone_inference(const float* spectrogram, float* features) {
    // Get input tensor
    float* input_data = tflm_get_input_data(g_backbone_interpreter, 0);
    if (!input_data) {
        return false;
    }
    
    // Copy spectrogram to input tensor
    memcpy(input_data, spectrogram, NN_INPUT_HEIGHT * NN_INPUT_WIDTH * sizeof(float));
    
    // Run inference
    if (tflm_invoke(g_backbone_interpreter) != TFLM_OK) {
        return false;
    }
    
    // Get output tensor
    float* output_data = tflm_get_output_data(g_backbone_interpreter, 0);
    if (!output_data) {
        return false;
    }
    
    // Copy features from output tensor
    memcpy(features, output_data, NN_BACKBONE_FEATURES * NN_TIME_FRAMES * sizeof(float));
    
    return true;
}

static bool run_streaming_inference(const float* features, float* logits, float* new_hidden_state) {
    // Get input tensors (features + hidden state)
    float* input_features = tflm_get_input_data(g_streaming_interpreter, 0);
    float* input_hidden = tflm_get_input_data(g_streaming_interpreter, 1);
    
    if (!input_features || !input_hidden) {
        return false;
    }
    
    // Copy inputs
    memcpy(input_features, features, NN_BACKBONE_FEATURES * sizeof(float));
    memcpy(input_hidden, g_gru_hidden_state, NN_GRU_HIDDEN_DIM * sizeof(float));
    
    // Run inference
    if (tflm_invoke(g_streaming_interpreter) != TFLM_OK) {
        return false;
    }
    
    // Get output tensors (logits + new hidden state)
    float* output_logits = tflm_get_output_data(g_streaming_interpreter, 0);
    float* output_hidden = tflm_get_output_data(g_streaming_interpreter, 1);
    
    if (!output_logits || !output_hidden) {
        return false;
    }
    
    // Copy outputs
    memcpy(logits, output_logits, NN_NUM_CLASSES * sizeof(float));
    memcpy(new_hidden_state, output_hidden, NN_GRU_HIDDEN_DIM * sizeof(float));
    
    return true;
}

static void apply_softmax(float* logits, int num_classes) {
    // Find max for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Compute exp(logits - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    
    // Normalize to get probabilities
    for (int i = 0; i < num_classes; i++) {
        logits[i] /= sum;
    }
}

static bool finalize_decision(const float* probabilities, NN_Decision_t* decision) {
    decision->num_detections = 0;
    
    // Find top predictions above threshold
    for (int i = 0; i < NN_NUM_CLASSES && decision->num_detections < NN_MAX_DETECTIONS_PER_SEC; i++) {
        if (probabilities[i] >= NN_CONFIDENCE_THRESHOLD) {
            NN_Detection_t* detection = &decision->detections[decision->num_detections];
            detection->class_id = i;
            detection->confidence = probabilities[i];
            detection->timestamp_ms = get_timestamp_ms();
            detection->valid = true;
            decision->num_detections++;
        }
    }
    
    return true;
}

static uint32_t get_timestamp_ms(void) {
    // Placeholder - in real implementation would use AudioMoth timer
    // For now, return frame counter as approximate timestamp
    return g_frame_counter * (NN_FRAME_SIZE * 1000 / NN_SAMPLE_RATE);
} 