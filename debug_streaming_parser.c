#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "inc/nn/nn_model.h" // For streaming_model_data

static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static void hexdump(const uint8_t* data, uint32_t offset, uint32_t len) {
    printf("Hexdump at offset %u:\n", offset);
    for (uint32_t i = 0; i < len; i++) {
        if (i % 16 == 0) printf("%04X: ", offset + i);
        printf("%02X ", data[offset + i]);
        if (i % 16 == 15) printf("\n");
    }
    if (len % 16 != 0) printf("\n");
}

int main() {
    printf("🔍 STREAMING MODEL PARSER DEBUG\n");
    printf("Model size: %zu bytes\n", streaming_model_data_len);
    
    // 1. Root table offset
    uint32_t root_offset = read_uint32_le(streaming_model_data);
    printf("Root offset: %u\n", root_offset);
    
    // 2. Subgraphs offset (field 2 della root table)
    uint32_t subgraphs_offset = read_uint32_le(streaming_model_data + 8 + root_offset + 8);
    printf("Subgraphs offset: %u\n", subgraphs_offset);
    
    // 3. Numero subgraphs
    uint32_t num_subgraphs = read_uint32_le(streaming_model_data + subgraphs_offset);
    printf("Number of subgraphs: %u\n", num_subgraphs);
    
    // 4. First subgraph offset
    uint32_t first_subgraph_offset_rel = read_uint32_le(streaming_model_data + subgraphs_offset + 4);
    uint32_t first_subgraph_offset = subgraphs_offset + 4 + first_subgraph_offset_rel;
    printf("First subgraph relative offset: %u\n", first_subgraph_offset_rel);
    printf("First subgraph absolute offset: %u\n", first_subgraph_offset);
    
    // 5. CRITICAL: Analizza VTable del primo subgraph
    printf("\n=== VTABLE ANALYSIS ===\n");
    
    // Read VTable offset (primo uint32 del subgraph object)
    uint32_t vtable_offset_raw = read_uint32_le(streaming_model_data + first_subgraph_offset);
    printf("VTable offset raw (from subgraph): %u (0x%08X)\n", vtable_offset_raw, vtable_offset_raw);
    
    // Calcolo VTable address CORRETTO secondo FlatBuffers spec
    // VTable si trova PRIMA del table object, non dopo
    if (vtable_offset_raw > first_subgraph_offset) {
        printf("❌ ERROR: VTable offset %u > subgraph offset %u!\n", 
               vtable_offset_raw, first_subgraph_offset);
        printf("This indicates parser bug or corrupted data\n");
        
        // Hexdump around the problematic area
        hexdump(streaming_model_data, first_subgraph_offset - 16, 32);
        return 1;
    }
    
    uint32_t vtable_abs = first_subgraph_offset - vtable_offset_raw;
    printf("VTable absolute offset: %u (subgraph %u - vtable_offset %u)\n", 
           vtable_abs, first_subgraph_offset, vtable_offset_raw);
    
    // 6. Leggi VTable header
    if (vtable_abs + 8 > streaming_model_data_len) {
        printf("❌ ERROR: VTable %u + 8 > model size %zu\n", vtable_abs, streaming_model_data_len);
        return 1;
    }
    
    uint16_t vtable_size = read_uint16_le(streaming_model_data + vtable_abs);
    uint16_t object_size = read_uint16_le(streaming_model_data + vtable_abs + 2);
    
    printf("VTable size: %u bytes\n", vtable_size);
    printf("Object size: %u bytes\n", object_size);
    
    // 7. Hexdump VTable
    printf("\n=== VTABLE HEXDUMP ===\n");
    hexdump(streaming_model_data, vtable_abs, vtable_size < 64 ? vtable_size : 64);
    
    printf("\n✅ STREAMING PARSER DEBUG COMPLETED\n");
    return 0;
}