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

int main() {
    printf("🔧 TESTING OFFSET ADJUSTMENT FIX\n");
    
    uint32_t first_subgraph_abs = 9369;  // From previous analysis
    printf("Streaming subgraph at: %u\n", first_subgraph_abs);
    
    // Test VTable offset at different positions
    for (int offset_test = 0; offset_test <= 12; offset_test += 4) {
        printf("\n--- Testing VTable offset at position +%d ---\n", offset_test);
        
        uint32_t vtable_offset = read_uint32_le(streaming_model_data + first_subgraph_abs + offset_test);
        printf("VTable offset: %u (0x%X)\n", vtable_offset, vtable_offset);
        
        if (vtable_offset > 0 && vtable_offset <= first_subgraph_abs) {
            uint32_t vtable_abs = first_subgraph_abs + offset_test - vtable_offset;
            printf("VTable absolute: %u\n", vtable_abs);
            
            if (vtable_abs < streaming_model_data_len - 8) {
                uint16_t vtable_size = read_uint16_le(streaming_model_data + vtable_abs);
                uint16_t object_size = read_uint16_le(streaming_model_data + vtable_abs + 2);
                printf("VTable size: %u, Object size: %u\n", vtable_size, object_size);
                
                if (vtable_size >= 8 && vtable_size <= 50 && object_size >= 8 && object_size <= 100) {
                    printf("✅ REASONABLE VALUES! This might be correct\n");
                    
                    printf("Field offsets: ");
                    for (int i = 4; i < vtable_size && i < 20; i += 2) {
                        uint16_t field = read_uint16_le(streaming_model_data + vtable_abs + i);
                        printf("%u ", field);
                    }
                    printf("\n");
                    printf("*** POTENTIAL FIX: Use offset +%d for streaming model ***\n", offset_test);
                } else {
                    printf("❌ Unreasonable values\n");
                }
            } else {
                printf("❌ VTable position out of bounds\n");
            }
        } else {
            printf("❌ Invalid VTable offset\n");
        }
    }
    
    return 0;
}