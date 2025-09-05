#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "third_party/tflite_parser_minimal.h"
#include "inc/nn/nn_model.h"

int main() {
    printf("🔧 TESTING STREAMING MODEL PARSER FIX\n");
    
    printf("Parsing streaming model with endianness auto-detection...\n");
    TFLiteModel* model = parse_tflite_model(streaming_model_data, streaming_model_data_len);
    
    if (model) {
        printf("✅ SUCCESS! Streaming model parsed successfully\n");
        print_model_info(model);
        
        // Test access to first subgraph
        if (model->num_subgraphs > 0) {
            printf("\nFirst subgraph details:\n");
            printf("  Tensors: %u\n", model->subgraphs[0].num_tensors);
            printf("  Operators: %u\n", model->subgraphs[0].num_operators);
            printf("  Inputs: %u\n", model->subgraphs[0].num_inputs);
            printf("  Outputs: %u\n", model->subgraphs[0].num_outputs);
        }
        
        free_tflite_model(model);
    } else {
        printf("❌ FAILED! Streaming model parsing still broken\n");
        return 1;
    }
    
    printf("\n🎯 Test completed successfully!\n");
    return 0;
}