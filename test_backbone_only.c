#include <stdio.h>
#include <stdlib.h>
#include "third_party/tflite_parser_minimal.h"
#include "inc/nn/nn_model.h"

// Forward declaration
bool parse_subgraph(const uint8_t* data, uint32_t subgraph_offset, TFLiteSubgraph* subgraph);

int main() {
    printf("🎯 TEST BACKBONE MODEL PARSER\n");
    printf("==============================\n");
    
    // Test solo backbone model che funziona
    printf("\n📊 PARSING BACKBONE MODEL:\n");
    TFLiteModel* backbone = parse_tflite_model(backbone_model_data, backbone_model_data_len);
    if (backbone) {
        print_model_info(backbone);
        
        // Test accesso ai tensori
        printf("\n🔍 TESTING TENSOR ACCESS:\n");
        
        // Prova a parsare TUTTI i subgraph per trovare i dati
        printf("Testing ALL subgraphs:\n");
        for (uint32_t sg = 0; sg < backbone->num_subgraphs && sg < 6; sg++) {
            printf("\n=== TRYING SUBGRAPH %d ===\n", sg);
            
            // Calcola offset del subgraph manualmente
            uint32_t subgraphs_base = 10200; // Dal debug
            uint32_t sg_offset_rel = *((uint32_t*)(backbone_model_data + subgraphs_base + 4 + (sg * 4)));
            uint32_t sg_offset_abs = subgraphs_base + 4 + (sg * 4) + sg_offset_rel;
            
            printf("Subgraph %d: offset_rel=%d, offset_abs=%d\n", sg, sg_offset_rel, sg_offset_abs);
            
            if (sg_offset_abs < backbone_model_data_len - 32) {
                // Prova a parsare questo subgraph
                TFLiteSubgraph test_sg = {0};
                if (parse_subgraph(backbone_model_data, sg_offset_abs, &test_sg)) {
                    printf("Subgraph %d: %d tensors, %d operators, %d inputs, %d outputs\n", 
                           sg, test_sg.num_tensors, test_sg.num_operators, 
                           test_sg.num_inputs, test_sg.num_outputs);
                    
                    if (test_sg.num_tensors > 0 || test_sg.num_operators > 0) {
                        printf("  -> 🎯 FOUND NON-EMPTY SUBGRAPH %d!\n", sg);
                    }
                    
                    // Cleanup
                    free(test_sg.tensors);
                    free(test_sg.operators);
                    free(test_sg.inputs);
                    free(test_sg.outputs);
                    free(test_sg.name);
                } else {
                    printf("Subgraph %d: failed to parse\n", sg);
                }
            }
        }
        
        free_tflite_model(backbone);
        printf("✅ Backbone model test successful!\n");
        return 0;
    } else {
        printf("❌ Failed to parse backbone model\n");
        return 1;
    }
}