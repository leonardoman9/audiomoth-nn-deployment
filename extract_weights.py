#!/usr/bin/env python3
"""
Script per estrarre pesi reali e struttura dai modelli TFLite
Genera file C con array statici per AudioMoth
"""

import tensorflow as tf
import numpy as np
import os
import sys

def analyze_model_structure(interpreter):
    """Analizza la struttura completa del modello TFLite"""
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print(f"📊 TENSORS: {len(tensor_details)}")
    print(f"🔢 INPUTS: {len(input_details)}")
    print(f"🎯 OUTPUTS: {len(output_details)}")
    
    # Analyze layers
    layers = []
    for i, tensor in enumerate(tensor_details):
        print(f"  Tensor {i}: {tensor['name']} | Shape: {tensor['shape']} | Type: {tensor['dtype']}")
        if 'quantization_parameters' in tensor:
            quant = tensor['quantization_parameters']
            if 'scales' in quant and len(quant['scales']) > 0:
                print(f"    Quantization - Scale: {quant['scales']}, Zero: {quant['zero_points']}")
    
    return {
        'inputs': input_details,
        'outputs': output_details,
        'tensors': tensor_details
    }

def extract_weights_from_model(tflite_path, model_name):
    """Estrae tutti i pesi da un modello TFLite"""
    
    print(f"\n🔍 ANALYZING {model_name.upper()} MODEL: {tflite_path}")
    
    if not os.path.exists(tflite_path):
        print(f"❌ File not found: {tflite_path}")
        return None
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()
    
    # Analyze structure
    structure = analyze_model_structure(interpreter)
    
    # Extract all weights
    weights_data = {}
    
    for i, tensor in enumerate(structure['tensors']):
        try:
            # Get tensor data
            tensor_data = interpreter.get_tensor(i)
            
            if tensor_data is not None and tensor_data.size > 0:
                weights_data[f"tensor_{i}"] = {
                    'name': tensor['name'],
                    'shape': tensor['shape'], 
                    'dtype': str(tensor['dtype']),
                    'data': tensor_data,
                    'quantization': tensor.get('quantization_parameters', {})
                }
                
                print(f"✅ Extracted tensor {i}: {tensor['name']} | {tensor_data.shape} | {tensor_data.dtype}")
            
        except Exception as e:
            print(f"⚠️  Could not extract tensor {i}: {e}")
    
    return {
        'structure': structure,
        'weights': weights_data,
        'model_name': model_name
    }

def generate_c_header(model_data, output_path):
    """Genera file header C con definizioni"""
    
    model_name = model_data['model_name']
    structure = model_data['structure']
    
    header_content = f"""#ifndef {model_name.upper()}_WEIGHTS_H
#define {model_name.upper()}_WEIGHTS_H

#include <stdint.h>
#include <stddef.h>

// ================================================================
// {model_name.upper()} MODEL STRUCTURE (REAL FROM TFLITE)
// ================================================================

"""
    
    # Input/Output definitions
    inputs = structure['inputs']
    outputs = structure['outputs']
    
    for i, inp in enumerate(inputs):
        shape = inp['shape']
        header_content += f"// Input {i}: {inp['name']}\n"
        if len(shape) == 4:  # [batch, height, width, channels]
            header_content += f"#define {model_name.upper()}_INPUT_{i}_H {shape[1]}\n"
            header_content += f"#define {model_name.upper()}_INPUT_{i}_W {shape[2]}\n" 
            header_content += f"#define {model_name.upper()}_INPUT_{i}_C {shape[3]}\n"
        elif len(shape) == 2:  # [batch, features]
            header_content += f"#define {model_name.upper()}_INPUT_{i}_SIZE {shape[1]}\n"
        header_content += f"#define {model_name.upper()}_INPUT_{i}_TOTAL {np.prod(shape[1:])}\n\n"
    
    for i, out in enumerate(outputs):
        shape = out['shape']
        header_content += f"// Output {i}: {out['name']}\n"
        header_content += f"#define {model_name.upper()}_OUTPUT_{i}_SIZE {shape[1] if len(shape) > 1 else shape[0]}\n\n"
    
    # Weight declarations
    header_content += f"// Weight arrays (extern declarations)\n"
    
    weights = model_data['weights']
    for tensor_id, tensor_info in weights.items():
        if tensor_info['data'].size > 1:  # Skip scalars
            clean_name = tensor_info['name'].replace('/', '_').replace(';', '_').replace(':', '_').replace('.', '_')
            dtype = 'int8_t' if 'int8' in tensor_info['dtype'] else 'float'
            header_content += f"extern const {dtype} {model_name}_{clean_name}[{tensor_info['data'].size}];\n"
    
    header_content += f"\n#endif // {model_name.upper()}_WEIGHTS_H\n"
    
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"✅ Generated header: {output_path}")

def generate_c_source(model_data, output_path):
    """Genera file sorgente C con array di pesi"""
    
    model_name = model_data['model_name']
    weights = model_data['weights']
    
    source_content = f"""#include "{model_name}_weights.h"

// ================================================================
// {model_name.upper()} MODEL WEIGHTS (REAL FROM TFLITE)
// ================================================================

"""
    
    for tensor_id, tensor_info in weights.items():
        data = tensor_info['data']
        
        if data.size <= 1:  # Skip scalars
            continue
            
        clean_name = tensor_info['name'].replace('/', '_').replace(';', '_').replace(':', '_').replace('.', '_')
        dtype = 'int8_t' if 'int8' in tensor_info['dtype'] else 'float'
        
        source_content += f"// {tensor_info['name']} | Shape: {tensor_info['shape']}\n"
        source_content += f"const {dtype} {model_name}_{clean_name}[{data.size}] = {{\n"
        
        # Convert data to C array format
        flat_data = data.flatten()
        
        # Write in chunks of 12 elements per line
        for i in range(0, len(flat_data), 12):
            chunk = flat_data[i:i+12]
            if 'int8' in tensor_info['dtype']:
                chunk_str = ', '.join([f"{int(x):4d}" for x in chunk])
            else:
                chunk_str = ', '.join([f"{float(x):8.6f}" for x in chunk])
            
            source_content += f"    {chunk_str}"
            if i + 12 < len(flat_data):
                source_content += ","
            source_content += "\n"
        
        source_content += "};\n\n"
    
    with open(output_path, 'w') as f:
        f.write(source_content)
    
    print(f"✅ Generated source: {output_path}")

def main():
    """Main extraction function"""
    
    print("🚀 TFLITE WEIGHT EXTRACTOR FOR AUDIOMOTH")
    print("=" * 50)
    
    # Model paths
    models = [
        {
            'path': './pc-test/backbone_int8_FINAL.tflite',
            'name': 'backbone'
        },
        {
            'path': './pc-test/streaming_int8_FINAL.tflite', 
            'name': 'streaming'
        }
    ]
    
    for model_config in models:
        print(f"\n{'='*60}")
        
        # Extract model data
        model_data = extract_weights_from_model(model_config['path'], model_config['name'])
        
        if model_data is None:
            print(f"❌ Failed to extract {model_config['name']} model")
            continue
        
        # Generate C files
        header_path = f"third_party/{model_config['name']}_weights.h"
        source_path = f"third_party/{model_config['name']}_weights.c"
        
        generate_c_header(model_data, header_path)
        generate_c_source(model_data, source_path)
        
        print(f"✅ {model_config['name'].upper()} model extraction complete!")
    
    print(f"\n🎉 ALL MODELS EXTRACTED SUCCESSFULLY!")
    print("Next steps:")
    print("1. Include the generated headers in your CMSIS wrapper")
    print("2. Replace random weights with real weight arrays")
    print("3. Update model structure to match extracted topology")

if __name__ == "__main__":
    main()