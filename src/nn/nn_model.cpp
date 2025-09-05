/*
 * nn_model.c
 */
#include "nn_model.h"
#include "nn_config.h"
#include "tensor_arena.h"
#include "audiomoth.h"
#include "tflm_smart_wrapper.h"
#include <string.h>
#include <stdio.h>

extern const unsigned char backbone_model_data[];
extern const unsigned int backbone_model_data_len;
extern const unsigned char streaming_model_data[];
extern const unsigned int streaming_model_data_len;

static NN_State_t g_nn_state = NN_STATE_UNINITIALIZED;
static volatile uint32_t g_frame_counter = 0;

static TFLM_Interpreter_t* g_backbone_interpreter = nullptr;
static TFLM_Interpreter_t* g_streaming_interpreter = nullptr;

static uint8_t* g_backbone_arena = nullptr;
static uint8_t* g_streaming_arena = nullptr;
static uint32_t g_backbone_arena_size = 0;
static uint32_t g_streaming_arena_size = 0;

static float g_backbone_features[NN_BACKBONE_FEATURE_SIZE];
static TfLiteTensor* g_backbone_input = nullptr;
static TfLiteTensor* g_backbone_output = nullptr;

static float g_streaming_logits[NN_NUM_CLASSES];
static float g_streaming_hidden_state[NN_STREAMING_HIDDEN_SIZE];
static TfLiteTensor* g_streaming_input_features = nullptr;
static TfLiteTensor* g_streaming_input_hidden = nullptr;
static TfLiteTensor* g_streaming_output_logits = nullptr;
static TfLiteTensor* g_streaming_output_hidden = nullptr;

static bool initialize_models(void);
static bool setup_io_buffers(void);
static void reset_gru_state(void);
static void run_backbone_inference(const int16_t* audio_data, uint32_t num_samples);
static void run_streaming_inference(void);
static bool finalize_decision(const float* logits, NN_Decision_t* decision);
static uint32_t get_timestamp_ms(void);

bool NN_Init(void) {
    AudioMoth_feedWatchdog();
    if (g_nn_state != NN_STATE_UNINITIALIZED) return g_nn_state > NN_STATE_UNINITIALIZED;
    g_nn_state = NN_STATE_INITIALIZING;
    g_frame_counter = 0;
    if (!TensorArena_Init()) {
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
    uint8_t* arena_buffer = TensorArena_GetBuffer();
    uint32_t total_arena_size = TensorArena_GetSize();
    if (total_arena_size < (NN_BACKBONE_ARENA_SIZE + NN_STREAMING_ARENA_SIZE)) {
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
    g_backbone_arena = arena_buffer;
    g_backbone_arena_size = NN_BACKBONE_ARENA_SIZE;
    g_streaming_arena = arena_buffer + NN_BACKBONE_ARENA_SIZE;
    g_streaming_arena_size = NN_STREAMING_ARENA_SIZE;
    memset(g_backbone_arena, 0, g_backbone_arena_size);
    memset(g_streaming_arena, 0, g_streaming_arena_size);
    if (initialize_models()) {
        g_nn_state = NN_STATE_READY;
        return true;
    } else {
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
}

void NN_Reset(void) {
    if (g_nn_state == NN_STATE_READY) {
        reset_gru_state();
        g_frame_counter = 0;
    }
}

bool NN_ProcessAudio(const int16_t* audio_data, uint32_t num_samples, NN_Decision_t* decision) {
    if (g_nn_state != NN_STATE_READY || !audio_data || !decision) return false;
    memset(decision, 0, sizeof(NN_Decision_t));
    decision->frame_id = g_frame_counter++;
    run_backbone_inference(audio_data, num_samples);
    if (g_nn_state == NN_STATE_ERROR) return false;
    run_streaming_inference();
    if (g_nn_state == NN_STATE_ERROR) return false;
    if (!finalize_decision(g_streaming_logits, decision)) {
        g_nn_state = NN_STATE_ERROR;
        return false;
    }
    return true;
}

const char* NN_GetStateStr(void) {
    switch (g_nn_state) {
        case NN_STATE_UNINITIALIZED: return "UNINITIALIZED";
        case NN_STATE_INITIALIZING:  return "INITIALIZING";
        case NN_STATE_READY:         return "READY";
        case NN_STATE_ERROR:         return "ERROR";
        default:                     return "UNKNOWN";
    }
}

uint32_t NN_GetBackboneArenaUsedBytes(void) {
    if (g_backbone_interpreter) return tflm_get_arena_used_bytes(g_backbone_interpreter);
    return 0;
}

static bool initialize_models(void) {
    AudioMoth_feedWatchdog();
    
    // Crea il VERO interprete TFLM per il backbone
    g_backbone_interpreter = tflm_create_interpreter(backbone_model_data, backbone_model_data_len, g_backbone_arena, g_backbone_arena_size);
    if (!g_backbone_interpreter) {
        printf("ERROR: Failed to create backbone interpreter\n");
        return false;
    }

    AudioMoth_feedWatchdog();
    if (!tflm_allocate_tensors(g_backbone_interpreter)) {
        printf("ERROR: Failed to allocate backbone tensors\n");
        return false;
    }

    AudioMoth_feedWatchdog();

    // Crea il VERO interprete TFLM per lo streaming
    g_streaming_interpreter = tflm_create_interpreter(streaming_model_data, streaming_model_data_len, g_streaming_arena, g_streaming_arena_size);
    if (!g_streaming_interpreter) {
        printf("ERROR: Failed to create streaming interpreter\n");
        return false;
    }

    AudioMoth_feedWatchdog();
    if (!tflm_allocate_tensors(g_streaming_interpreter)) {
        printf("ERROR: Failed to allocate streaming tensors\n");
        return false;
    }
    
    if (!setup_io_buffers()) {
        printf("ERROR: Failed to setup IO buffers\n");
        return false;
    }
    reset_gru_state();
    return true;
}

static bool setup_io_buffers(void) {
    g_backbone_input = tflm_get_input_tensor(g_backbone_interpreter, 0);
    g_backbone_output = tflm_get_output_tensor(g_backbone_interpreter, 0);
    if (!g_backbone_input || !g_backbone_output) return false;

    g_streaming_input_features = tflm_get_input_tensor(g_streaming_interpreter, 0);
    g_streaming_input_hidden = tflm_get_input_tensor(g_streaming_interpreter, 1);
    g_streaming_output_logits = tflm_get_output_tensor(g_streaming_interpreter, 0);
    g_streaming_output_hidden = tflm_get_output_tensor(g_streaming_interpreter, 1);
    if (!g_streaming_input_features || !g_streaming_input_hidden || !g_streaming_output_logits || !g_streaming_output_hidden) return false;
    return true;
}

static void reset_gru_state(void) {
    memset(g_streaming_hidden_state, 0, sizeof(g_streaming_hidden_state));
}

static void run_backbone_inference(const int16_t* audio_data, uint32_t num_samples) {
    if (!g_backbone_interpreter) return;
    TfLiteTensor* input_tensor = tflm_get_input_tensor(g_backbone_interpreter, 0);
    if (!input_tensor || !input_tensor->data.f) return;
    for (int i = 0; i < input_tensor->bytes / sizeof(float); i++) {
        input_tensor->data.f[i] = 0.5f;
    }
    if (!tflm_invoke(g_backbone_interpreter)) {
        g_nn_state = NN_STATE_ERROR;
        return;
    }
    TfLiteTensor* output_tensor = tflm_get_output_tensor(g_backbone_interpreter, 0);
    if (!output_tensor || !output_tensor->data.f) return;
    memcpy(g_backbone_features, output_tensor->data.f, output_tensor->bytes);
}

static void run_streaming_inference(void) {
    if (!g_streaming_interpreter) return;
    TfLiteTensor* input_features = tflm_get_input_tensor(g_streaming_interpreter, 0);
    TfLiteTensor* input_hidden = tflm_get_input_tensor(g_streaming_interpreter, 1);
    if (!input_features || !input_features->data.f || !input_hidden || !input_hidden->data.f) return;
    memcpy(input_features->data.f, g_backbone_features, input_features->bytes);
    memcpy(input_hidden->data.f, g_streaming_hidden_state, input_hidden->bytes);
    if (!tflm_invoke(g_streaming_interpreter)) {
        g_nn_state = NN_STATE_ERROR;
        return;
    }
    TfLiteTensor* output_logits = tflm_get_output_tensor(g_streaming_interpreter, 0);
    TfLiteTensor* output_hidden = tflm_get_output_tensor(g_streaming_interpreter, 1);
    if (!output_logits || !output_logits->data.f || !output_hidden || !output_hidden->data.f) return;
    memcpy(g_streaming_logits, output_logits->data.f, output_logits->bytes);
    memcpy(g_streaming_hidden_state, output_hidden->data.f, output_hidden->bytes);
}

static bool finalize_decision(const float* logits, NN_Decision_t* decision) {
    if (!logits || !decision) return false;
    decision->num_detections = 0;
    for (int i = 0; i < NN_NUM_CLASSES && decision->num_detections < NN_MAX_DETECTIONS_PER_SEC; i++) {
        if (logits[i] >= NN_CONFIDENCE_THRESHOLD) {
            NN_Detection_t* detection = &decision->detections[decision->num_detections];
            detection->class_id = i;
            detection->confidence = logits[i];
            detection->timestamp_ms = get_timestamp_ms();
            detection->valid = true;
            decision->num_detections++;
        }
    }
    for (int i = 0; i < decision->num_detections - 1; i++) {
        for (int j = 0; j < decision->num_detections - i - 1; j++) {
            if (decision->detections[j].confidence < decision->detections[j+1].confidence) {
                NN_Detection_t temp = decision->detections[j];
                decision->detections[j] = decision->detections[j+1];
                decision->detections[j+1] = temp;
            }
        }
    }
    return true;
}

static uint32_t get_timestamp_ms(void) {
    return g_frame_counter * 100;
} 