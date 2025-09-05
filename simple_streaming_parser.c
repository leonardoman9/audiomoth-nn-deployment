#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Simplified structures for testing
typedef struct {
    uint32_t shape[4];
    uint32_t type;
    uint32_t buffer_id;
    float scale;
    int32_t zero_point;
} SimpleTensor;

typedef struct {
    uint32_t num_tensors;
    uint32_t num_operators;
    uint32_t num_inputs;
    uint32_t num_outputs;
    SimpleTensor* tensors;
} SimpleSubgraph;

typedef struct {
    uint32_t num_subgraphs;
    SimpleSubgraph* subgraphs;
} SimpleModel;

bool create_streaming_model(SimpleModel* model) {
    printf("🔧 CREATING SIMPLIFIED STREAMING MODEL\n");
    
    // Initialize
    model->num_subgraphs = 1;
    model->subgraphs = malloc(sizeof(SimpleSubgraph));
    if (!model->subgraphs) {
        printf("ERROR: Failed to allocate subgraph\n");
        return false;
    }
    
    SimpleSubgraph* sg = &model->subgraphs[0];
    sg->num_tensors = 3;  // Start small for testing
    sg->num_operators = 2;
    sg->num_inputs = 1;
    sg->num_outputs = 1;
    
    // Allocate tensors
    sg->tensors = malloc(sg->num_tensors * sizeof(SimpleTensor));
    if (!sg->tensors) {
        printf("ERROR: Failed to allocate tensors\n");
        free(model->subgraphs);
        return false;
    }
    
    // Input tensor: [1, 64] - features(32) + hidden_state(32)
    sg->tensors[0].shape[0] = 1;
    sg->tensors[0].shape[1] = 64;
    sg->tensors[0].shape[2] = 1;
    sg->tensors[0].shape[3] = 1;
    sg->tensors[0].type = 9; // int8
    sg->tensors[0].buffer_id = 0;
    sg->tensors[0].scale = 0.0078125f;
    sg->tensors[0].zero_point = 0;
    
    // Intermediate tensor: [1, 32]
    sg->tensors[1].shape[0] = 1;
    sg->tensors[1].shape[1] = 32;
    sg->tensors[1].shape[2] = 1;
    sg->tensors[1].shape[3] = 1;
    sg->tensors[1].type = 9; // int8
    sg->tensors[1].buffer_id = 1;
    sg->tensors[1].scale = 0.0078125f;
    sg->tensors[1].zero_point = 0;
    
    // Output tensor: [1, 37] - new_hidden(32) + logits(5)
    sg->tensors[2].shape[0] = 1;
    sg->tensors[2].shape[1] = 37;
    sg->tensors[2].shape[2] = 1;
    sg->tensors[2].shape[3] = 1;
    sg->tensors[2].type = 9; // int8
    sg->tensors[2].buffer_id = 2;
    sg->tensors[2].scale = 0.0078125f;
    sg->tensors[2].zero_point = 0;
    
    printf("✅ Simplified streaming model created successfully!\n");
    return true;
}

void print_model_info(SimpleModel* model) {
    printf("\n📊 MODEL INFO:\n");
    printf("Subgraphs: %u\n", model->num_subgraphs);
    
    if (model->num_subgraphs > 0) {
        SimpleSubgraph* sg = &model->subgraphs[0];
        printf("  Tensors: %u\n", sg->num_tensors);
        printf("  Operators: %u\n", sg->num_operators);
        printf("  Inputs: %u\n", sg->num_inputs);
        printf("  Outputs: %u\n", sg->num_outputs);
        
        for (uint32_t i = 0; i < sg->num_tensors; i++) {
            printf("  Tensor[%u]: shape=[%u,%u,%u,%u], type=%u, scale=%f\n",
                   i, sg->tensors[i].shape[0], sg->tensors[i].shape[1],
                   sg->tensors[i].shape[2], sg->tensors[i].shape[3],
                   sg->tensors[i].type, sg->tensors[i].scale);
        }
    }
}

void cleanup_model(SimpleModel* model) {
    if (model->subgraphs) {
        if (model->subgraphs[0].tensors) {
            free(model->subgraphs[0].tensors);
        }
        free(model->subgraphs);
    }
}

int main() {
    printf("🧪 SIMPLE STREAMING MODEL TEST\n");
    
    SimpleModel model;
    if (create_streaming_model(&model)) {
        print_model_info(&model);
        cleanup_model(&model);
        printf("\n✅ Test completed successfully!\n");
        printf("🎯 Ready to integrate into real parser!\n");
    } else {
        printf("❌ Test failed!\n");
        return 1;
    }
    
    return 0;
}