#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "third_party/tflite_cmsis_bridge.h"
#include "inc/nn/nn_model.h"
#include "inc/nn/tensor_arena.h"

int main() {
    printf("🌉 TEST TFLITE-CMSIS BRIDGE\n");
    printf("============================\n");
    
    // Test tensor arena (use external SRAM)
    static uint8_t test_arena[TENSOR_ARENA_SIZE];
    printf("Test arena: %d bytes\n", TENSOR_ARENA_SIZE);
    
    // Test backbone model
    printf("\n📊 TESTING BACKBONE MODEL:\n");
    TFLiteCMSISBridge* backbone_bridge = tflite_cmsis_create(
        backbone_model_data, 
        backbone_model_data_len,
        "backbone",
        test_arena,
        TENSOR_ARENA_SIZE
    );
    
    if (backbone_bridge) {
        tflite_cmsis_print_info(backbone_bridge);
        
        // Initialize bridge
        if (tflite_cmsis_init(backbone_bridge)) {
            printf("✅ Backbone bridge initialized\n");
            
            // Get tensor info
            uint32_t input_shape[4];
            float input_scale;
            int32_t input_zero_point;
            
            if (tflite_cmsis_get_input_info(backbone_bridge, input_shape, &input_scale, &input_zero_point)) {
                printf("📥 Input info: [%d,%d,%d,%d], scale=%f, zero_point=%d\n",
                       input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                       input_scale, input_zero_point);
            }
            
            // Get input/output tensors
            int8_t* input_tensor = tflite_cmsis_get_input(backbone_bridge);
            int8_t* output_tensor = tflite_cmsis_get_output(backbone_bridge);
            
            if (input_tensor && output_tensor) {
                printf("📍 Tensors allocated: input=%p, output=%p\n", 
                       (void*)input_tensor, (void*)output_tensor);
                
                // Fill input with test data
                for (int i = 0; i < 18 * 40; i++) {
                    input_tensor[i] = (int8_t)(i % 256 - 128);
                }
                
                // Run inference
                printf("\n🚀 Running backbone inference...\n");
                if (tflite_cmsis_invoke(backbone_bridge)) {
                    printf("✅ Backbone inference successful!\n");
                    
                    // Print first few output values
                    printf("📤 Output samples: ");
                    for (int i = 0; i < 10 && i < 18 * 32; i++) {
                        printf("%d ", output_tensor[i]);
                    }
                    printf("...\n");
                } else {
                    printf("❌ Backbone inference failed\n");
                }
            } else {
                printf("❌ Failed to get tensor pointers\n");
            }
        } else {
            printf("❌ Failed to initialize backbone bridge\n");
        }
        
        tflite_cmsis_free(backbone_bridge);
    } else {
        printf("❌ Failed to create backbone bridge\n");
    }
    
    // Test streaming model
    printf("\n📊 TESTING STREAMING MODEL:\n");
    TFLiteCMSISBridge* streaming_bridge = tflite_cmsis_create(
        streaming_model_data, 
        streaming_model_data_len,
        "streaming",
        test_arena,
        TENSOR_ARENA_SIZE
    );
    
    if (streaming_bridge) {
        tflite_cmsis_print_info(streaming_bridge);
        
        if (tflite_cmsis_init(streaming_bridge)) {
            printf("✅ Streaming bridge initialized\n");
            
            // Get input/output tensors
            int8_t* input_tensor = tflite_cmsis_get_input(streaming_bridge);
            int8_t* output_tensor = tflite_cmsis_get_output(streaming_bridge);
            
            if (input_tensor && output_tensor) {
                // Fill input with test data
                for (int i = 0; i < 32; i++) {
                    input_tensor[i] = (int8_t)(i % 64 - 32);
                }
                
                // Run inference
                printf("\n🚀 Running streaming inference...\n");
                if (tflite_cmsis_invoke(streaming_bridge)) {
                    printf("✅ Streaming inference successful!\n");
                    
                    // Print output (classification scores)
                    printf("📤 Classification scores: ");
                    for (int i = 0; i < 5; i++) {
                        printf("%d ", output_tensor[i]);
                    }
                    printf("\n");
                } else {
                    printf("❌ Streaming inference failed\n");
                }
            }
        } else {
            printf("❌ Failed to initialize streaming bridge\n");
        }
        
        tflite_cmsis_free(streaming_bridge);
    } else {
        printf("❌ Failed to create streaming bridge\n");
    }
    
    printf("\n🎯 BRIDGE TEST COMPLETED!\n");
    return 0;
}