#include <stdio.h>
#include "third_party/tflite_parser_minimal.h"
#include "inc/nn/nn_model.h"

// Test parser con i nostri modelli embedded
int main() {
    printf("🧪 TEST TFLITE PARSER MINIMALE\n");
    printf("===============================\n");
    
    // Test backbone model
    printf("\n📊 TESTING BACKBONE MODEL:\n");
    TFLiteModel* backbone = parse_tflite_model(backbone_model_data, backbone_model_data_len);
    if (backbone) {
        print_model_info(backbone);
        
        // Test accesso ai tensori
        uint32_t shape[4];
        uint32_t dims = get_tensor_shape(backbone, 0, 0, shape, 4);
        printf("Input tensor shape: ");
        for (uint32_t i = 0; i < dims; i++) {
            printf("%d ", shape[i]);
        }
        printf("\n");
        
        TFLiteQuantization quant;
        if (get_tensor_quantization(backbone, 0, 0, &quant)) {
            printf("Input quantization: scale=%f, zero_point=%d\n", quant.scale, quant.zero_point);
        }
        
        free_tflite_model(backbone);
        printf("✅ Backbone model parsed successfully!\n");
    } else {
        printf("❌ Failed to parse backbone model\n");
        return 1;
    }
    
    // Test streaming model
    printf("\n📊 TESTING STREAMING MODEL:\n");
    TFLiteModel* streaming = parse_tflite_model(streaming_model_data, streaming_model_data_len);
    if (streaming) {
        print_model_info(streaming);
        free_tflite_model(streaming);
        printf("✅ Streaming model parsed successfully!\n");
    } else {
        printf("❌ Failed to parse streaming model\n");
        return 1;
    }
    
    printf("\n🎯 ALL TESTS PASSED!\n");
    return 0;
}