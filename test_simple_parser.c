#include <stdio.h>
#include <stdint.h>
#include "inc/nn/nn_model.h"

// Helper per leggere int32 little-endian
static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | 
           ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | 
           ((uint32_t)data[3] << 24);
}

// Helper per leggere int16 little-endian  
static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

void hexdump(const uint8_t* data, int start, int len) {
    printf("Hexdump at offset %d:\n", start);
    for (int i = 0; i < len; i += 16) {
        printf("%04X: ", start + i);
        for (int j = 0; j < 16 && i + j < len; j++) {
            printf("%02X ", data[start + i + j]);
        }
        printf("\n");
    }
}

int main() {
    printf("🔍 ANALISI STRUTTURA TFLITE\n");
    printf("============================\n");
    
    printf("Model size: %d bytes\n", backbone_model_data_len);
    
    // Analizza i primi bytes
    hexdump(backbone_model_data, 0, 64);
    
    // Leggi struttura base
    uint32_t root_offset = read_uint32_le(backbone_model_data);
    uint32_t magic = read_uint32_le(backbone_model_data + 4);
    
    printf("\nRoot offset: %d (0x%08X)\n", root_offset, root_offset);
    printf("Magic: 0x%08X\n", magic);
    
    // Analizza la root table
    printf("\nRoot table at offset %d:\n", 8 + root_offset);
    hexdump(backbone_model_data, 8 + root_offset, 64);
    
    // Analizza cosa c'è agli offset che leggiamo
    uint32_t field1 = read_uint32_le(backbone_model_data + 8 + root_offset + 0);
    uint32_t field2 = read_uint32_le(backbone_model_data + 8 + root_offset + 4);
    uint32_t field3 = read_uint32_le(backbone_model_data + 8 + root_offset + 8);
    uint32_t field4 = read_uint32_le(backbone_model_data + 8 + root_offset + 12);
    
    printf("\nRoot table fields:\n");
    printf("  Field 0: %d (0x%08X) - probably version\n", field1, field1);
    printf("  Field 1: %d (0x%08X) - probably operator_codes\n", field2, field2);
    printf("  Field 2: %d (0x%08X) - probably subgraphs\n", field3, field3);
    printf("  Field 3: %d (0x%08X) - probably buffers\n", field4, field4);
    
    // Verifica subgraphs (field2 = 10200)
    printf("\nAnalyzing subgraphs at absolute offset %d:\n", field3);
    if (field3 < backbone_model_data_len - 8) {
        hexdump(backbone_model_data, field3, 64);
        uint32_t num_subgraphs = read_uint32_le(backbone_model_data + field3);
        printf("Number of subgraphs: %d\n", num_subgraphs);
        
        // Analizza il primo subgraph
        if (num_subgraphs > 0) {
            uint32_t first_sg_offset_rel = read_uint32_le(backbone_model_data + field3 + 4);
            uint32_t first_sg_offset_abs = field3 + 4 + first_sg_offset_rel;
            printf("First subgraph relative offset: %d\n", first_sg_offset_rel);
            printf("First subgraph absolute offset: %d\n", first_sg_offset_abs);
            
            printf("\nFirst subgraph data:\n");
            hexdump(backbone_model_data, first_sg_offset_abs, 64);
            
            // Analizza la vtable del subgraph
            printf("\nAnalyzing subgraph vtable:\n");
            uint32_t vtable_offset = read_uint32_le(backbone_model_data + first_sg_offset_abs);
            printf("VTable offset: %d\n", vtable_offset);
            if (vtable_offset > 0 && vtable_offset < 1000) {
                uint32_t vtable_abs = first_sg_offset_abs - vtable_offset;
                printf("VTable absolute: %d\n", vtable_abs);
                hexdump(backbone_model_data, vtable_abs, 32);
                
                // Parse VTable details
                uint16_t vtable_size = read_uint16_le(backbone_model_data + vtable_abs);
                uint16_t object_size = read_uint16_le(backbone_model_data + vtable_abs + 2);
                printf("VTable size: %d, object size: %d\n", vtable_size, object_size);
                
                // DEBUG: mostra i byte raw
                printf("Raw VTable bytes: ");
                for (int j = 0; j < 8; j++) {
                    printf("%02X ", backbone_model_data[vtable_abs + j]);
                }
                printf("\n");
                
                // List field offsets
                for (int i = 4; i < vtable_size && i < 20; i += 2) {
                    uint16_t field_offset = read_uint16_le(backbone_model_data + vtable_abs + i);
                    printf("Field %d: offset %d\n", (i-4)/2, field_offset);
                }
            }
            
            // Analizza anche i dati raw del subgraph
            printf("\nSubgraph object fields:\n");
            for (int i = 4; i < 20 && first_sg_offset_abs + i < backbone_model_data_len; i += 4) {
                uint32_t field_value = read_uint32_le(backbone_model_data + first_sg_offset_abs + i);
                printf("Offset %d: 0x%08X (%d)\n", i, field_value, field_value);
            }
        }
    }
    
    // Verifica buffers (field3 = 10216)
    printf("\nAnalyzing buffers at absolute offset %d:\n", field4);
    if (field4 < backbone_model_data_len - 8) {
        hexdump(backbone_model_data, field4, 32);
        uint32_t num_buffers = read_uint32_le(backbone_model_data + field4);
        printf("Number of buffers: %d\n", num_buffers);
    }
    
    return 0;
}