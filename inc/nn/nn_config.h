#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

// Neural Network Configuration - FULL GRU-64 MODEL WITH DYNAMIC ALLOCATION!
#define NN_NUM_CLASSES              35      // FULL: 35 classes (complete dataset)

// Model Architecture Configuration - MAXIMUM SPECS WITH GRU-64!
#define NN_BACKBONE_FEATURES        32      // Base filters: 32
#define NN_TIME_FRAMES              18      // Time frames: 18
#define NN_GRU_HIDDEN_DIM           64      // GRU-64: Maximum hidden dimensions!

// Input Configuration - ORIGINAL: BACKBONE [1, 18, 40] 
#define NN_INPUT_HEIGHT             40      // ORIGINAL: 40 mel features (was 12)
#define NN_INPUT_WIDTH              18      // ORIGINAL: 18 time frames (was 12)  
#define NN_INPUT_CHANNELS           1       // Single channel spectrogram

// TensorFlow Lite Micro Configuration - VIRTUAL ARENA FOR GRU-64!  
#define NN_BACKBONE_ARENA_SIZE      (40 * 1024)    // 40KB virtual arena for GRU-64
#define NN_STREAMING_ARENA_SIZE     (20 * 1024)    // 20KB virtual arena for streaming

// Flash-based memory extension for larger models
#define NN_USE_FLASH_SWAP           1              // ENABLED: Flash memory swapping for original model!
#define NN_FLASH_SWAP_SIZE          (64 * 1024)    // 64KB Flash area for tensor swap

// Inference Configuration
#define NN_CONFIDENCE_THRESHOLD     0.7f    // Minimum confidence for detection
#define NN_MAX_DETECTIONS_PER_SEC   2       // REDUCED: From 5 to 2 detections

// Audio Processing Configuration
#define NN_SAMPLE_RATE              48000   // AudioMoth sample rate
#define NN_FRAME_SIZE               1024    // Samples per frame
#define NN_HOP_LENGTH               512     // Hop length for STFT

// Class Labels (placeholder - to be updated with real bird classes)
extern const char* NN_CLASS_NAMES[NN_NUM_CLASSES];

#ifdef __cplusplus
}
#endif

#endif // NN_CONFIG_H