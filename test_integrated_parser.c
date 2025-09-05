#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "third_party/tflite_parser_minimal.h"
#include "inc/nn/nn_model.h"

int main() {
    printf("🧪 TESTING INTEGRATED PARSER WITH CUSTOM STREAMING\n");
    
    printf("\n=== TESTING BACKBONE MODEL (should use standard parser) ===\n");
    TFLiteModel* backbone_model = parse_tflite_model(backbone_model_data, backbone_model_data_len);
    if (backbone_model) {
        printf("✅ Backbone model parsed successfully\n");
        print_model_info(backbone_model);
        free_tflite_model(backbone_model);
    } else {
        printf("❌ Backbone model parsing failed\n");
    }
    
    printf("\n=== TESTING STREAMING MODEL (should use custom parser) ===\n");
    TFLiteModel* streaming_model = parse_tflite_model(streaming_model_data, streaming_model_data_len);
    if (streaming_model) {
        printf("✅ Streaming model parsed successfully\n");
        print_model_info(streaming_model);
        free_tflite_model(streaming_model);
    } else {
        printf("❌ Streaming model parsing failed\n");
    }
    
    printf("\n🎯 Integration test completed!\n");
    return 0;
}