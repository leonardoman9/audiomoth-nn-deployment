#include <stdio.h>
#include <stdint.h>
#include "inc/nn/nn_model.h"

static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint16_t read_uint16_be(const uint8_t* data) {
    return ((uint16_t)data[0] << 8) | (uint16_t)data[1];
}

static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static uint32_t read_uint32_be(const uint8_t* data) {
    return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) | 
           ((uint32_t)data[2] << 8) | (uint32_t)data[3];
}

int main() {
    printf("🔍 VTABLE ENDIANNESS INVESTIGATION\n");
    
    // Get to the first subgraph
    uint32_t root_offset = read_uint32_le(streaming_model_data);
    uint32_t subgraphs_offset = read_uint32_le(streaming_model_data + 8 + root_offset + 8);
    uint32_t first_subgraph_rel = read_uint32_le(streaming_model_data + subgraphs_offset + 4);
    uint32_t first_subgraph_abs = subgraphs_offset + 4 + first_subgraph_rel;
    
    printf("First subgraph at: %u\n", first_subgraph_abs);
    
    // VTable offset (we know this is big-endian)
    uint32_t vtable_offset_be = read_uint32_be(streaming_model_data + first_subgraph_abs);
    uint32_t vtable_abs = first_subgraph_abs - vtable_offset_be;
    
    printf("VTable offset (BE): %u\n", vtable_offset_be);
    printf("VTable absolute: %u\n", vtable_abs);
    
    printf("\n🔍 VTABLE CONTENT ANALYSIS (both endianness)\n");
    
    // Test VTable size and object size in both endianness
    uint16_t vtable_size_le = read_uint16_le(streaming_model_data + vtable_abs);
    uint16_t vtable_size_be = read_uint16_be(streaming_model_data + vtable_abs);
    
    uint16_t object_size_le = read_uint16_le(streaming_model_data + vtable_abs + 2);
    uint16_t object_size_be = read_uint16_be(streaming_model_data + vtable_abs + 2);
    
    printf("VTable size - LE: %u, BE: %u\n", vtable_size_le, vtable_size_be);
    printf("Object size - LE: %u, BE: %u\n", object_size_le, object_size_be);
    
    // Reasonable values for VTable should be small
    printf("\nReasonableness check:\n");
    printf("VTable size LE reasonable: %s\n", (vtable_size_le > 0 && vtable_size_le < 100) ? "YES" : "NO");
    printf("VTable size BE reasonable: %s\n", (vtable_size_be > 0 && vtable_size_be < 100) ? "YES" : "NO");
    printf("Object size LE reasonable: %s\n", (object_size_le > 0 && object_size_le < 200) ? "YES" : "NO");
    printf("Object size BE reasonable: %s\n", (object_size_be > 0 && object_size_be < 200) ? "YES" : "NO");
    
    // Test field offsets
    printf("\n🔍 FIELD OFFSETS (first few)\n");
    for (int i = 4; i < 16; i += 2) {
        uint16_t field_le = read_uint16_le(streaming_model_data + vtable_abs + i);
        uint16_t field_be = read_uint16_be(streaming_model_data + vtable_abs + i);
        printf("Field[%d] - LE: %u, BE: %u\n", (i-4)/2, field_le, field_be);
    }
    
    printf("\n🎯 CONCLUSION: Check which endianness produces reasonable values\n");
    return 0;
}