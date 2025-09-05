/* AudioMoth Minimal OpResolver Implementation
 * Registers only the operators we actually need
 */

#include "audiomoth_op_resolver.h"

namespace tflite {

AudioMothOpResolver::AudioMothOpResolver() {
  // Core operators for CNN backbone
  AddConv2D();
  AddDepthwiseConv2D();
  AddFullyConnected();
  AddRelu();
  AddMaxPool2D();
  AddAveragePool2D();
  AddReshape();
  AddSoftmax();
  
  // Additional operators that might be needed
  AddAdd();
  AddMul();
  AddQuantize();
  AddDequantize();
  
  // RNN operators for streaming model
  AddUnidirectionalSequenceLSTM();  // Replacement for GRU
  AddTanh();
  
  // Add Logistic (Sigmoid) if needed
  // AddLogistic();
}

}  // namespace tflite

// C interface implementation
extern "C" {

void* audiomoth_create_op_resolver(void) {
  return new tflite::AudioMothOpResolver();
}

void audiomoth_destroy_op_resolver(void* resolver) {
  delete static_cast<tflite::AudioMothOpResolver*>(resolver);
}

}