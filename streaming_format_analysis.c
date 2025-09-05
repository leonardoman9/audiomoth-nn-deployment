#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "inc/nn/nn_model.h"

static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static void search_for_patterns(const uint8_t* data, uint32_t len) {
    printf("🔍 SEARCHING FOR RECOGNIZABLE PATTERNS\n");
    
    // 1. Look for tensor count patterns (streaming model should have ~47 tensors)
    printf("\n--- SEARCHING FOR TENSOR COUNT (47) ---\n");
    for (uint32_t i = 0; i < len - 4; i += 4) {
        uint32_t val = read_uint32_le(data + i);
        if (val == 47) {
            printf("Found 47 at offset %u (0x%X)\n", i, i);
            
            // Check surrounding context
            printf("  Context: ");
            for (int j = -8; j <= 8; j += 4) {
                if (i + j >= 0 && i + j < len - 4) {
                    uint32_t ctx = read_uint32_le(data + i + j);
                    if (j == 0) {
                        printf("[%u] ", ctx);
                    } else {
                        printf("%u ", ctx);
                    }
                }
            }
            printf("\n");
        }
    }
    
    // 2. Look for operator count patterns (streaming should have ~10-15 operators)
    printf("\n--- SEARCHING FOR OPERATOR COUNT (10-15) ---\n");
    for (uint32_t i = 0; i < len - 4; i += 4) {
        uint32_t val = read_uint32_le(data + i);
        if (val >= 10 && val <= 15) {
            printf("Found %u at offset %u (0x%X)\n", val, i, i);
        }
    }
    
    // 3. Look for string patterns that indicate structure
    printf("\n--- SEARCHING FOR STRUCTURE INDICATORS ---\n");
    const char* patterns[] = {"MLIR", "Conv", "Dense", "GRU", "Softmax", "input", "output"};
    for (int p = 0; p < 7; p++) {
        printf("Searching for '%s':\n", patterns[p]);
        for (uint32_t i = 0; i < len - strlen(patterns[p]); i++) {
            if (memcmp(data + i, patterns[p], strlen(patterns[p])) == 0) {
                printf("  Found at offset %u (0x%X)\n", i, i);
                
                // Print some context around the string
                uint32_t start = (i >= 16) ? i - 16 : 0;
                uint32_t end = (i + 32 < len) ? i + 32 : len;
                printf("  Context: ");
                for (uint32_t j = start; j < end; j++) {
                    if (data[j] >= 32 && data[j] < 127) {
                        printf("%c", data[j]);
                    } else {
                        printf(".");
                    }
                }
                printf("\n");
            }
        }
    }
    
    // 4. Look for reasonable small integers that could be dimensions
    printf("\n--- POTENTIAL TENSOR SHAPES (64, 32, 5) ---\n");
    uint32_t targets[] = {64, 32, 5, 1};
    for (int t = 0; t < 4; t++) {
        printf("Searching for %u:\n", targets[t]);
        int count = 0;
        for (uint32_t i = 0; i < len - 4 && count < 10; i += 4) {
            uint32_t val = read_uint32_le(data + i);
            if (val == targets[t]) {
                printf("  Offset %u (0x%X)\n", i, i);
                count++;
            }
        }
    }
}

static void analyze_subgraph_region() {
    printf("\n🎯 DETAILED SUBGRAPH REGION ANALYSIS\n");
    
    // We know the problematic subgraph is around offset 9369
    uint32_t region_start = 9300;
    uint32_t region_end = 9500;
    
    printf("Analyzing region %u to %u:\n", region_start, region_end);
    
    // Print as both hex dump and uint32 values
    for (uint32_t i = region_start; i < region_end; i += 16) {
        printf("%04X: ", i);
        
        // Hex bytes
        for (int j = 0; j < 16; j++) {
            if (i + j < streaming_model_data_len) {
                printf("%02X ", streaming_model_data[i + j]);
            } else {
                printf("-- ");
            }
        }
        
        printf(" | ");
        
        // As uint32 values
        for (int j = 0; j < 16; j += 4) {
            if (i + j + 3 < streaming_model_data_len) {
                uint32_t val = read_uint32_le(streaming_model_data + i + j);
                printf("%10u ", val);
            }
        }
        
        printf("\n");
    }
}

int main() {
    printf("🧬 STREAMING MODEL FORMAT REVERSE ENGINEERING\n");
    printf("Model size: %u bytes\n", streaming_model_data_len);
    
    search_for_patterns(streaming_model_data, streaming_model_data_len);
    analyze_subgraph_region();
    
    printf("\n✅ Analysis complete. Look for patterns that indicate data structure.\n");
    return 0;
}