# TensorFlow Lite Micro source files for AudioMoth
# This makefile fragment defines the minimal set of TFLM files needed

# Base TFLM directory
TFLM_ROOT = ../third_party/tflite-micro

# Core TFLM source files (minimal set)
TFLM_CORE_SRCS = \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_interpreter.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_allocator.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_error_reporter.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_profiler.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_time.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_utils.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_log.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_allocation_info.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/memory_helpers.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/flatbuffer_utils.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/micro_op_resolver.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/system_setup.cc

# Schema and flatbuffers
TFLM_SCHEMA_SRCS = \
	$(TFLM_ROOT)/tensorflow/lite/schema/schema_utils.cc \
	$(TFLM_ROOT)/tensorflow/lite/core/api/error_reporter.cc \
	$(TFLM_ROOT)/tensorflow/lite/core/api/flatbuffer_conversions.cc \
	$(TFLM_ROOT)/tensorflow/lite/core/api/tensor_utils.cc \
	$(TFLM_ROOT)/tensorflow/lite/core/api/op_resolver.cc

# Essential kernels for AudioMoth models
TFLM_KERNEL_SRCS = \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/conv.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/conv_common.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/depthwise_conv.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/fully_connected.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/fully_connected_common.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/activations.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/activations_common.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/pooling.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/reshape.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/softmax.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/add.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/add_common.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/mul.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/quantize.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/dequantize.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cc

# CMSIS-NN optimized kernels (if available)
ifeq ($(USE_CMSIS_NN),1)
TFLM_CMSIS_SRCS = \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/cmsis_nn/conv.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/cmsis_nn/pooling.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/cmsis_nn/softmax.cc \
	$(TFLM_ROOT)/tensorflow/lite/micro/kernels/cmsis_nn/add.cc
else
TFLM_CMSIS_SRCS =
endif

# Reference implementations for math operations
TFLM_REFERENCE_SRCS = \
	$(TFLM_ROOT)/tensorflow/lite/kernels/internal/common.cc \
	$(TFLM_ROOT)/tensorflow/lite/kernels/internal/quantization_util.cc \
	$(TFLM_ROOT)/tensorflow/lite/kernels/internal/reference/comparisons.cc \
	$(TFLM_ROOT)/tensorflow/lite/kernels/internal/reference/conv.cc

# AudioMoth specific files
AUDIOMOTH_TFLM_SRCS = \
	../third_party/audiomoth_op_resolver.cc \
	../third_party/tflm_real.cc

# Combine all sources
TFLM_ALL_SRCS = $(TFLM_CORE_SRCS) $(TFLM_SCHEMA_SRCS) $(TFLM_KERNEL_SRCS) $(TFLM_CMSIS_SRCS) $(TFLM_REFERENCE_SRCS) $(AUDIOMOTH_TFLM_SRCS)

# Include paths for TFLM
TFLM_INCLUDES = \
	-I$(TFLM_ROOT) \
	-I$(TFLM_ROOT)/tensorflow/lite \
	-I$(TFLM_ROOT)/tensorflow/lite/micro \
	-I$(TFLM_ROOT)/tensorflow/lite/c \
	-I$(TFLM_ROOT)/tensorflow/lite/core \
	-I$(TFLM_ROOT)/tensorflow/lite/kernels \
	-I$(TFLM_ROOT)/tensorflow/lite/kernels/internal \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/kernels \
	-I$(TFLM_ROOT)/third_party/flatbuffers/include \
	-I$(TFLM_ROOT)/third_party/gemmlowp \
	-I$(TFLM_ROOT)/third_party/ruy

# Compiler flags for TFLM
TFLM_CXXFLAGS = \
	-DTF_LITE_STATIC_MEMORY \
	-DTF_LITE_MCU_DEBUG_LOG \
	-DTFLITE_EMULATE_FLOAT \
	-DARM_MATH_CM4 \
	-std=c++11 \
	-fno-rtti \
	-fno-exceptions \
	-fno-threadsafe-statics

# Optional CMSIS-NN optimization
ifeq ($(USE_CMSIS_NN),1)
TFLM_CXXFLAGS += -DCMSIS_NN
endif