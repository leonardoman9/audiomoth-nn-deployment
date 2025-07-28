/*
 * nn_config.h
 *
 *  Created on: 25 May 2024
 *      Author: leonardomannini
 *
 *  Configuration constants for the Neural Network model and audio processing pipeline.
 *  These values are derived from the model training configuration.
 */

#ifndef INC_NN_NN_CONFIG_H_
#define INC_NN_NN_CONFIG_H_

/* Compilation flags */
#ifndef ENABLE_NN
#define ENABLE_NN 0
#endif

/* Audio processing parameters (from training config) */
#define NN_SAMPLE_RATE 32000
#define NN_CLIP_DURATION_S 3.0f
#define NN_DECISION_INTERVAL_MS 3000

/*
 * The NN pipeline processes audio in small windows.
 * These are NOT the STFT windows, but larger chunks for processing efficiency.
 */
#define NN_PROCESS_WINDOW_MS 200
#define NN_PROCESS_WINDOW_SAMPLES (NN_SAMPLE_RATE * NN_PROCESS_WINDOW_MS / 1000) // 6400 samples

/* Spectral processing (matches training) */
#define NN_FFT_SIZE 512
#define NN_HOP_LENGTH 320
#define NN_N_MEL_BINS 64
#define NN_N_LINEAR_FILTERS 64
#define NN_SPECTRAL_FEATURES (NN_N_MEL_BINS + NN_N_LINEAR_FILTERS) // 128
#define NN_FREQ_RANGE_LOW 150.0f
#define NN_FREQ_RANGE_HIGH 16000.0f

/* Model architecture */
#define NN_NUM_CLASSES 71            // 70 bird species + no_bird
#define NN_LSTM_HIDDEN_DIM 64
#define NN_CNN_BASE_FILTERS 32
#define NN_CNN_LAYERS 3

/*
 * Memory allocation estimates.
 * WARNING: These are tight and will need careful profiling.
 * The EFM32GG has 32KB of RAM.
 */
#define NN_CNN_ARENA_SIZE           (15 * 1024)    // 15KB initial estimate
#define NN_LSTM_ARENA_SIZE          (12 * 1024)    // 12KB initial estimate
#define NN_SPECTRAL_BUFFER_SIZE     (NN_SPECTRAL_FEATURES * sizeof(int8_t))
#define NN_STFT_WORKSPACE_SIZE      (NN_FFT_SIZE * 2 * sizeof(float)) // For complex float FFT

#endif /* INC_NN_NN_CONFIG_H_ */ 