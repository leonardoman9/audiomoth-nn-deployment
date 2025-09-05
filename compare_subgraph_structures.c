#include <stdio.h>
#include <stdint.h>
#include "inc/nn/nn_model.h"

static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static void analyze_subgraph_structure(const char* name, const uint8_t* data, uint32_t data_len) {
    printf("\n🔍 ===== %s SUBGRAPH STRUCTURE =====\n", name);
    
    // Navigate to first subgraph
    uint32_t root_offset = read_uint32_le(data);
    uint32_t subgraphs_offset = read_uint32_le(data + 8 + root_offset + 8);
    uint32_t first_subgraph_rel = read_uint32_le(data + subgraphs_offset + 4);
    uint32_t first_subgraph_abs = subgraphs_offset + 4 + first_subgraph_rel;
    
    printf("First subgraph at: %u\n", first_subgraph_abs);
    
    // Dump first 64 bytes of subgraph
    printf("Raw bytes (first 64):\n");
    for (int i = 0; i < 64; i += 16) {
        printf("%04X: ", first_subgraph_abs + i);
        for (int j = 0; j < 16 && i + j < 64; j++) {
            if (first_subgraph_abs + i + j < data_len) {
                printf("%02X ", data[first_subgraph_abs + i + j]);
            } else {
                printf("-- ");
            }
        }
        printf("\n");
    }
    
    // Try to parse as standard FlatBuffer table
    uint32_t vtable_offset = read_uint32_le(data + first_subgraph_abs);
    printf("\nVTable offset (LE): %u (0x%X)\n", vtable_offset, vtable_offset);
    
    if (vtable_offset > 0 && vtable_offset <= first_subgraph_abs) {
        uint32_t vtable_abs = first_subgraph_abs - vtable_offset;
        printf("VTable absolute: %u\n", vtable_abs);
        
        if (vtable_abs < data_len - 8) {
            uint16_t vtable_size = read_uint16_le(data + vtable_abs);
            uint16_t object_size = read_uint16_le(data + vtable_abs + 2);
            printf("VTable size: %u, Object size: %u\n", vtable_size, object_size);
            
            if (vtable_size > 0 && vtable_size < 100) {
                printf("Field offsets: ");
                for (int i = 4; i < vtable_size && i < 20; i += 2) {
                    uint16_t field = read_uint16_le(data + vtable_abs + i);
                    printf("%u ", field);
                }
                printf("\n");
                printf("✅ Standard FlatBuffer structure detected\n");
            } else {
                printf("❌ Invalid VTable size\n");
            }
        }
    } else {
        printf("❌ Invalid VTable offset\n");
    }
}

int main() {
    printf("🔬 COMPARATIVE SUBGRAPH STRUCTURE ANALYSIS\n");
    
    analyze_subgraph_structure("BACKBONE", backbone_model_data, backbone_model_data_len);
    analyze_subgraph_structure("STREAMING", streaming_model_data, streaming_model_data_len);
    
    printf("\n🎯 COMPARISON COMPLETE\n");
    printf("Look for structural differences between backbone and streaming\n");
    
    return 0;
}