/**
 * @file streaming_topology.h
 * @brief Exact topology mapping for streaming model
 * 
 * Contains precise tensor indices, shapes, and connections
 * for 99.999% identical implementation.
 */

#ifndef STREAMING_TOPOLOGY_H
#define STREAMING_TOPOLOGY_H

#include <stdint.h>

// Model dimensions
#define STREAMING_NUM_TENSORS 47
#define STREAMING_NUM_INPUTS 2
#define STREAMING_NUM_OUTPUTS 2

// Input tensor indices
#define STREAMING_INPUT_0_INDEX 0
#define STREAMING_INPUT_0_SHAPE { 1, 32 }
#define STREAMING_INPUT_1_INDEX 1
#define STREAMING_INPUT_1_SHAPE { 1, 32 }

// Output tensor indices
#define STREAMING_OUTPUT_0_INDEX 39
#define STREAMING_OUTPUT_0_SHAPE { 1, 32 }
#define STREAMING_OUTPUT_1_INDEX 46
#define STREAMING_OUTPUT_1_SHAPE { 1, 5 }

// All tensor information
typedef struct {
    uint32_t index;
    uint32_t shape[4];
    uint32_t num_dims;
    float scale;
    int32_t zero_point;
    const char* name;
} streaming_tensor_info_t;

extern const streaming_tensor_info_t streaming_tensor_info[STREAMING_NUM_TENSORS];

#endif // STREAMING_TOPOLOGY_H
