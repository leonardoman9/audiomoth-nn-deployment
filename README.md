# AudioMoth Neural Network Deployment

🎵🤖 **Real-time audio classification on AudioMoth devices using TensorFlow Lite Micro**

This repository integrates neural network inference capabilities into the [AudioMoth-Firmware-Basic](https://github.com/OpenAcousticDevices/AudioMoth-Firmware-Basic), enabling on-device audio classification with minimal power consumption.

## 🎯 Project Overview

**AudioMoth** is a low-cost, full-spectrum acoustic logger designed for wildlife monitoring. This project extends its capabilities by adding real-time neural network inference for automatic audio event detection and classification.

### Key Features
- 🧠 **Dual-model pipeline**: CNN feature extraction + LSTM temporal modeling
- ⚡ **Real-time processing**: 200ms windows, 3-second decisions
- 💾 **Memory-optimized**: Fits in ~32KB RAM, ~256KB flash (EFM32)
- 🔋 **Low power**: Maintains AudioMoth's battery efficiency
- 📊 **Streaming LSTM**: Persistent state for temporal context
- 📁 **Full compatibility**: Works with existing AudioMoth ecosystem

## 🏗️ Architecture

```
Audio Input (32kHz) → STFT (512pt) → Mel+Linear → CNN → LSTM → 9 Classes
                   ↓               ↓         ↓      ↓       ↓
                 3s clips      64 mel +    Features H=64   Species
                 200ms win     64 linear   (INT8)   State   + no_bird
```

### Models
- **CNN**: MatchboxNet backbone (3 layers, 32 base filters, 3x3 kernels) for spectral feature extraction
- **LSTM**: Temporal sequence modeling with 64 hidden dimensions + attention/FC layers
- **Knowledge Distillation**: Student models trained with soft labels (α=0.4, T=3.0)
- **Quantization**: INT8 models optimized for TensorFlow Lite Micro

## 🛠️ Quick Start

### Prerequisites
- AudioMoth hardware or compatible EFM32 development board
- [AudioMoth-Project](https://github.com/OpenAcousticDevices/AudioMoth-Project) build environment
- TensorFlow Lite models (CNN + LSTM) in INT8 format

### Build
```bash
# Clone this repository
git clone https://github.com/leonardoman9/audiomoth-nn-deployment.git
cd audiomoth-nn-deployment

# Enable neural network features
make ENABLE_NN=1 -j4

# Check memory usage
make size

# Flash to device
make flash
```

### Model Deployment
```bash
# Convert TFLite models to C arrays
xxd -i cnn_model.tflite > src/nn/cnn_model_data.cc
xxd -i lstm_model.tflite > src/nn/lstm_model_data.cc
```

## 📊 Memory Budget

| Component | RAM Usage | Flash Usage | Notes |
|-----------|-----------|-------------|-------|
| CNN Arena | ~15KB | - | MatchboxNet-32 (3 layers) |
| LSTM Arena | ~12KB | - | Hidden dim 64 |
| Audio Buffers | ~6KB | - | Ring buffer + window |
| Spectral Features | ~1KB | - | 128 features (mel+linear) |
| STFT Workspace | ~4KB | - | FFT-512 + magnitude |
| **Total NN** | **~38KB** | **~45KB** |
| **Available** | 32KB | 256KB |

*⚠️ Current estimates exceed 32KB - optimization critical*

## 🔧 Configuration

Neural network inference can be enabled/disabled at compile time:

```c
// Enable NN processing
#define ENABLE_NN 1

// Audio processing parameters (from training config)
#define NN_SAMPLE_RATE 32000
#define NN_WINDOW_SAMPLES 6400      // 200ms @ 32kHz
#define NN_DECISION_INTERVAL_MS 3000 // Decision every 3 seconds
#define NN_CLIP_DURATION_S 3        // Full clip duration

// Spectral processing (matches training)
#define NN_FFT_SIZE 512
#define NN_HOP_LENGTH 320
#define NN_N_MEL_BINS 64
#define NN_N_LINEAR_FILTERS 64
#define NN_FREQ_RANGE_LOW 150       // Hz
#define NN_FREQ_RANGE_HIGH 16000    // Hz

// Model architecture
#define NN_NUM_CLASSES 9            // 8 bird species + no_bird
#define NN_LSTM_HIDDEN_DIM 64
#define NN_CNN_BASE_FILTERS 32
#define NN_CNN_LAYERS 3
```

## 📈 Performance Targets

- **Latency**: < 200ms per window processing
- **Memory**: ⚠️ **Critical**: Must optimize to fit 32KB RAM constraint  
- **Power**: < 10% increase vs. standard AudioMoth battery life
- **Accuracy**: > 85% on 9-class bird species classification
- **False Positives**: < 5% for no-bird class (crucial for wildlife monitoring)
- **Inference Rate**: Real-time processing of 32kHz audio streams

### Memory Optimization Strategy
Current estimates exceed hardware limits. Optimization priorities:
1. **Shared tensor arenas** between CNN and LSTM
2. **Streaming LSTM** with minimal state persistence  
3. **In-place spectral processing** to reduce buffer copies
4. **Model quantization** review (potentially INT4 for weights)

## 🤝 Contributing

This project extends the open-source AudioMoth ecosystem. Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-optimization`)
3. Commit your changes (`git commit -m 'Add amazing optimization'`)
4. Push to the branch (`git push origin feature/amazing-optimization`)
5. Open a Pull Request

## 📚 Related Projects

- [AudioMoth-Project](https://github.com/OpenAcousticDevices/AudioMoth-Project) - Main build framework
- [AudioMoth-Firmware-Basic](https://github.com/OpenAcousticDevices/AudioMoth-Firmware-Basic) - Base firmware
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers) - ML framework

## 📝 License

MIT License - Same as the original AudioMoth project

## 🙏 Acknowledgments

- [Open Acoustic Devices](https://www.openacousticdevices.info/) for the AudioMoth platform
- TensorFlow team for TensorFlow Lite Micro
- AudioMoth community for the open-source ecosystem

---
