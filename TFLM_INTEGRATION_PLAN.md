# TensorFlow Lite Micro Integration Plan

## Overview
Replace the stub `tflm_wrapper.c` with real TensorFlow Lite Micro implementation for AudioMoth neural network inference.

## Current State Analysis
- ✅ External SRAM tensor arena (256KB) working
- ✅ Dual model architecture (backbone + streaming) 
- ✅ Pipeline flow: Audio → Spectrogram → Backbone → Streaming → Decision
- ✅ TFLM source code available in `third_party/tflite-micro/`
- ✅ Stub wrapper provides API compatibility
- ⚠️ GRU operator not natively supported (solution: use LSTM)

## Implementation Steps

### 1. Build System Integration
- [ ] Configure Makefile to include TFLM source files
- [ ] Set up minimal OpResolver for required operators
- [ ] Enable CMSIS-NN optimizations for Cortex-M4
- [ ] Configure external SRAM memory mapping

### 2. Operator Requirements Analysis
Based on typical CNN backbone + RNN streaming models:

**Backbone Model Operators:**
- Conv2D (for spectral processing)
- DepthwiseConv2D (efficient convolutions)
- FullyConnected/Dense
- ReLU activation
- MaxPool2D/AvgPool2D
- Reshape
- Softmax (for output)

**Streaming Model Operators:**
- FullyConnected/Dense (for feature transformation)
- UnidirectionalSequenceLSTM (replacement for GRU)
- ReLU/Tanh activations
- Softmax (for final classification)

### 3. Minimal OpResolver Implementation
Create `AudioMothOpResolver` with only required operators to minimize ROM/RAM usage.

### 4. Memory Arena Integration
- [ ] Map TFLM tensor arena to external SRAM
- [ ] Calculate arena sizes for backbone (~150KB) and streaming (~50KB)
- [ ] Implement proper memory alignment (16-byte for Cortex-M4)

### 5. Dual Model Management
- [ ] Create separate MicroInterpreter instances
- [ ] Implement hidden state persistence between streaming inferences
- [ ] Handle model switching and state management

### 6. Input/Output Tensor Handling
- [ ] Float32 vs Int8 quantization support
- [ ] Tensor shape validation and error handling
- [ ] Input preprocessing and output postprocessing

### 7. Testing and Validation
- [ ] Compare outputs with stub implementation
- [ ] Performance benchmarking
- [ ] Memory usage verification
- [ ] Stability testing under load

## GRU Solution Decision
**Chosen Approach:** Replace GRU with UnidirectionalSequenceLSTM
- Pros: Native TFLM support, similar performance, proven stability
- Cons: Slightly higher memory usage (~10-15%)
- Alternative: Would require custom operator implementation (months of work)

## Quantization Strategy
**Target:** Int8 quantization for both models
- Reduces memory footprint by 4x vs Float32
- Enables CMSIS-NN optimizations
- Maintains acceptable accuracy for bird species classification

## Expected Performance Improvements
- **Real inference**: Actual neural network computation
- **CMSIS-NN optimizations**: 2-5x speedup on Cortex-M4
- **Int8 quantization**: 4x memory reduction
- **Dedicated operators**: Proper mathematical operations vs dummy calculations

## Risk Mitigation
1. **Memory overflow**: Careful arena size calculation and monitoring
2. **Performance degradation**: Benchmark against stub timings
3. **Accuracy loss**: Validate against ground truth data
4. **Stability issues**: Extensive testing and error handling

## Success Criteria
- [ ] Real model inference producing correct classifications
- [ ] Memory usage within 256KB SRAM limit
- [ ] Inference time < 50ms per frame
- [ ] No crashes during extended operation
- [ ] Accuracy comparable to original Python models