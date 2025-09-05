#include <stdio.h>
#include <stdint.h>
#include "inc/nn/nn_model.h"

static uint16_t read_uint16_le(const uint8_t* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t read_uint32_le(const uint8_t* data) {
    return (uint32_t)data[0] | 
           ((uint32_t)data[1] << 8) | 
           ((uint32_t)data[2] << 16) | 
           ((uint32_t)data[3] << 24);
}

int main() {
    printf("🔍 VTABLE DEBUG\n");
    
    // VTable è a offset 10184 (0x27C8)
    int vtable_offset = 10184;
    
    printf("VTable at offset %d (0x%04X):\n", vtable_offset, vtable_offset);
    
    // Stampa byte singoli
    for (int i = 0; i < 16; i++) {
        printf("Offset %d: 0x%02X\n", i, backbone_model_data[vtable_offset + i]);
    }
    
    // Test lettura uint16
    printf("\nUint16 reads:\n");
    printf("Offset 0: %d\n", read_uint16_le(backbone_model_data + vtable_offset + 0));
    printf("Offset 2: %d\n", read_uint16_le(backbone_model_data + vtable_offset + 2)); 
    printf("Offset 4: %d\n", read_uint16_le(backbone_model_data + vtable_offset + 4));
    printf("Offset 6: %d\n", read_uint16_le(backbone_model_data + vtable_offset + 6));
    
    // Test lettura uint32
    printf("\nUint32 reads:\n");
    printf("Offset 0: %d\n", read_uint32_le(backbone_model_data + vtable_offset + 0));
    printf("Offset 4: %d\n", read_uint32_le(backbone_model_data + vtable_offset + 4));
    
    return 0;
}