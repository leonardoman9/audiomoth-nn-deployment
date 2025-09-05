/**
 * @file backbone_topology.h
 * @brief Exact topology mapping for backbone model
 * 
 * Contains precise tensor indices, shapes, and connections
 * for 99.999% identical implementation.
 */

#ifndef BACKBONE_TOPOLOGY_H
#define BACKBONE_TOPOLOGY_H

#include <stdint.h>

// Model dimensions
#define BACKBONE_NUM_TENSORS 202
#define BACKBONE_NUM_INPUTS 1
#define BACKBONE_NUM_OUTPUTS 1

// Input tensor indices
#define BACKBONE_INPUT_0_INDEX 0
#define BACKBONE_INPUT_0_SHAPE { 1, 18, 40 }

// Output tensor indices
#define BACKBONE_OUTPUT_0_INDEX 201
#define BACKBONE_OUTPUT_0_SHAPE { 1, 18, 32 }

// All tensor information
typedef struct {
    uint32_t index;
    uint32_t shape[4];
    uint32_t num_dims;
    float scale;
    int32_t zero_point;
    const char* name;
} backbone_tensor_info_t;

extern const backbone_tensor_info_t backbone_tensor_info[BACKBONE_NUM_TENSORS];

#endif // BACKBONE_TOPOLOGY_H
