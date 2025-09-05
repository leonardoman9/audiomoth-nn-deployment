#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "inc/nn/nn_model.h" // For both model data

static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static void analyze_flatbuffer_structure(const char* name, const uint8_t* data, uint32_t data_len) {
    printf("\n🔍 ===== ANALYZING %s =====\n", name);
    printf("Size: %u bytes\n", data_len);
    
    // 1. Root table offset
    uint32_t root_offset = read_uint32_le(data);
    printf("Root offset: %u (0x%X)\n", root_offset, root_offset);
    
    // 2. Magic number
    uint32_t magic = read_uint32_le(data + 4);
    printf("Magic: 0x%08X\n", magic);
    
    // 3. Root table analysis
    printf("\n--- ROOT TABLE ---\n");
    uint32_t root_start = 8 + root_offset;
    printf("Root table starts at: %u\n", root_start);
    
    // Root table VTable offset
    uint32_t root_vtable_offset = read_uint32_le(data + root_start);
    uint32_t root_vtable_abs = root_start - root_vtable_offset;
    printf("Root VTable offset: %u, absolute: %u\n", root_vtable_offset, root_vtable_abs);
    
    // Root VTable info
    uint16_t root_vtable_size = read_uint16_le(data + root_vtable_abs);
    uint16_t root_object_size = read_uint16_le(data + root_vtable_abs + 2);
    printf("Root VTable size: %u, object size: %u\n", root_vtable_size, root_object_size);
    
    // 4. Subgraphs offset
    uint32_t subgraphs_offset = read_uint32_le(data + root_start + 8);
    printf("\n--- SUBGRAPHS ---\n");
    printf("Subgraphs offset: %u\n", subgraphs_offset);
    
    uint32_t num_subgraphs = read_uint32_le(data + subgraphs_offset);
    printf("Number of subgraphs: %u\n", num_subgraphs);
    
    // 5. First subgraph detailed analysis
    if (num_subgraphs > 0) {
        uint32_t first_subgraph_rel = read_uint32_le(data + subgraphs_offset + 4);
        uint32_t first_subgraph_abs = subgraphs_offset + 4 + first_subgraph_rel;
        printf("First subgraph rel: %u, abs: %u\n", first_subgraph_rel, first_subgraph_abs);
        
        printf("\n--- FIRST SUBGRAPH HEXDUMP ---\n");
        printf("Offset %u:\n", first_subgraph_abs);
        for (int i = 0; i < 32; i += 4) {
            if (first_subgraph_abs + i + 3 < data_len) {
                uint32_t val = read_uint32_le(data + first_subgraph_abs + i);
                printf("  [%2d] 0x%08X (%d)\n", i, val, val);
            }
        }
        
        // CRITICAL: Try different interpretations
        printf("\n--- INTERPRETATION ATTEMPTS ---\n");
        
        // Attempt 1: Standard FlatBuffer table
        uint32_t attempt1_vtable_offset = read_uint32_le(data + first_subgraph_abs);
        printf("Attempt 1 - VTable offset: 0x%08X\n", attempt1_vtable_offset);
        
        // Attempt 2: What if it's not little-endian?
        uint32_t attempt2_be = (data[first_subgraph_abs] << 24) | 
                               (data[first_subgraph_abs + 1] << 16) |
                               (data[first_subgraph_abs + 2] << 8) |
                               data[first_subgraph_abs + 3];
        printf("Attempt 2 - Big-endian: 0x%08X\n", attempt2_be);
        
        // Attempt 3: What if first 4 bytes are not VTable offset?
        printf("Attempt 3 - Skip first 4 bytes:\n");
        for (int skip = 4; skip <= 12; skip += 4) {
            if (first_subgraph_abs + skip + 3 < data_len) {
                uint32_t val = read_uint32_le(data + first_subgraph_abs + skip);
                if (val > 0 && val < first_subgraph_abs) {
                    printf("  Skip %d: 0x%08X (valid VTable candidate!)\n", skip, val);
                }
            }
        }
        
        // Attempt 4: Look for valid VTable patterns nearby
        printf("\n--- VTABLE PATTERN SEARCH ---\n");
        for (int offset = -20; offset <= 20; offset += 4) {
            uint32_t test_pos = first_subgraph_abs + offset;
            if (test_pos >= 4 && test_pos + 3 < data_len) {
                uint32_t val = read_uint32_le(data + test_pos);
                // Valid VTable offset should be small and positive
                if (val > 0 && val < 1000 && val < test_pos) {
                    uint32_t vtable_candidate = test_pos - val;
                    if (vtable_candidate < data_len - 4) {
                        uint16_t vt_size = read_uint16_le(data + vtable_candidate);
                        uint16_t obj_size = read_uint16_le(data + vtable_candidate + 2);
                        if (vt_size > 0 && vt_size < 200 && obj_size > 0 && obj_size < 1000) {
                            printf("  FOUND at offset %+d: VTable offset=0x%X, size=%u, obj_size=%u\n", 
                                   offset, val, vt_size, obj_size);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    printf("🔬 DEEP FLATBUFFER STRUCTURE ANALYSIS\n");
    
    // Compare backbone (working) vs streaming (broken)
    analyze_flatbuffer_structure("BACKBONE", backbone_model_data, backbone_model_data_len);
    analyze_flatbuffer_structure("STREAMING", streaming_model_data, streaming_model_data_len);
    
    printf("\n✅ Analysis complete. Look for pattern differences!\n");
    return 0;
}