#include <stdio.h>
#include <stdint.h>
#include "inc/nn/nn_model.h"

static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

int main() {
    printf("🔍 SEARCHING FOR VALID VTABLE PATTERNS\n");
    
    uint32_t first_subgraph_abs = 9369;  // From previous analysis
    
    printf("Searching around subgraph offset %u...\n", first_subgraph_abs);
    
    // Search in a wider range around the subgraph
    for (int search_offset = -200; search_offset <= 200; search_offset += 4) {
        uint32_t test_pos = first_subgraph_abs + search_offset;
        
        if (test_pos < 4 || test_pos >= streaming_model_data_len - 8) continue;
        
        uint16_t vtable_size = read_uint16_le(streaming_model_data + test_pos);
        uint16_t object_size = read_uint16_le(streaming_model_data + test_pos + 2);
        
        // Look for reasonable VTable patterns
        if (vtable_size >= 8 && vtable_size <= 50 && 
            object_size >= 8 && object_size <= 100 &&
            vtable_size % 2 == 0 && object_size % 4 == 0) {
            
            printf("🎯 POTENTIAL VTABLE at offset %d (abs %u):\n", search_offset, test_pos);
            printf("  VTable size: %u, Object size: %u\n", vtable_size, object_size);
            
            // Check field offsets
            printf("  Field offsets: ");
            bool valid_fields = true;
            for (int i = 4; i < vtable_size && i < 20; i += 2) {
                uint16_t field_offset = read_uint16_le(streaming_model_data + test_pos + i);
                printf("%u ", field_offset);
                if (field_offset > object_size && field_offset != 0) {
                    valid_fields = false;
                }
            }
            printf("\n");
            
            if (valid_fields) {
                printf("  ✅ VALID VTable candidate!\n");
                
                // Calculate what the VTable offset would be from subgraph
                uint32_t vtable_offset_needed = first_subgraph_abs - test_pos;
                printf("  Required VTable offset from subgraph: %u (0x%X)\n", 
                       vtable_offset_needed, vtable_offset_needed);
                
                // Check if this matches any value in the subgraph
                printf("  Values in subgraph object:\n");
                for (int j = 0; j < 32; j += 4) {
                    uint32_t val = read_uint32_le(streaming_model_data + first_subgraph_abs + j);
                    if (val == vtable_offset_needed) {
                        printf("    [%d] 0x%08X *** MATCH! ***\n", j, val);
                    } else {
                        printf("    [%d] 0x%08X\n", j, val);
                    }
                }
            }
            printf("\n");
        }
    }
    
    printf("🔍 Search completed.\n");
    return 0;
}