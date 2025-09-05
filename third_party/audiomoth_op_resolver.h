/* AudioMoth Minimal OpResolver
 * Contains only the operators needed for our specific models:
 * - Backbone: Conv2D, DepthwiseConv2D, FullyConnected, ReLU, MaxPool, Reshape, Softmax
 * - Streaming: FullyConnected, UnidirectionalSequenceLSTM, ReLU, Tanh, Softmax
 */

#ifndef AUDIOMOTH_OP_RESOLVER_H
#define AUDIOMOTH_OP_RESOLVER_H

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#ifdef __cplusplus

namespace tflite {

// AudioMoth-specific operator resolver with minimal memory footprint
// Supports maximum 15 operators (can be adjusted based on actual model requirements)
class AudioMothOpResolver : public MicroMutableOpResolver<15> {
 public:
  AudioMothOpResolver();
  
 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

// C interface for creating the resolver
extern "C" {
void* audiomoth_create_op_resolver(void);
void audiomoth_destroy_op_resolver(void* resolver);
}

#else
// Pure C interface
void* audiomoth_create_op_resolver(void);
void audiomoth_destroy_op_resolver(void* resolver);
#endif

#endif  // AUDIOMOTH_OP_RESOLVER_H