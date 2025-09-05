#!/usr/bin/env python3
"""
Estrazione completa parametri quantizzazione dai modelli TFLite
"""

import tensorflow as tf
import numpy as np
import json

def extract_all_quantization_params(tflite_path, model_name):
    """Estrae TUTTI i parametri di quantizzazione per ogni tensore"""
    print(f"\n{'='*60}")
    print(f"🔧 ESTRAZIONE QUANTIZZAZIONE: {model_name}")
    print(f"{'='*60}")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    
    quantization_data = {
        'model_name': model_name,
        'file_path': tflite_path,
        'total_tensors': len(tensor_details),
        'quantized_tensors': 0,
        'tensors': []
    }
    
    print(f"📊 Analizzando {len(tensor_details)} tensori...")
    
    for i, tensor in enumerate(tensor_details):
        tensor_info = {
            'index': i,
            'name': tensor['name'],
            'shape': tensor['shape'].tolist(),
            'dtype': str(tensor['dtype']),
            'quantization': None
        }
        
        # Estrai parametri quantizzazione
        if 'quantization_parameters' in tensor:
            quant = tensor['quantization_parameters']
            
            if quant['scales'].size > 0:
                quantization_data['quantized_tensors'] += 1
                
                # Gestisci per-channel vs per-tensor quantization
                scales = quant['scales'].tolist()
                zero_points = quant['zero_points'].tolist()
                
                tensor_info['quantization'] = {
                    'scales': scales,
                    'zero_points': zero_points,
                    'quantized_dimension': quant.get('quantized_dimension', -1),
                    'is_per_channel': len(scales) > 1
                }
                
                # Print per i tensori importanti
                if any(keyword in tensor['name'].lower() for keyword in ['input', 'output', 'conv', 'dense', 'qconst']):
                    scale_str = f"{scales[0]:.8f}" if len(scales) == 1 else f"{len(scales)} scales"
                    zp_str = f"{zero_points[0]}" if len(zero_points) == 1 else f"{len(zero_points)} zps"
                    print(f"  [{i:3d}] {tensor['name'][:50]:50} Scale:{scale_str:12} ZP:{zp_str:4}")
        
        quantization_data['tensors'].append(tensor_info)
    
    print(f"\n✅ Quantized tensors: {quantization_data['quantized_tensors']}/{quantization_data['total_tensors']}")
    
    return quantization_data

def extract_weights_data(tflite_path, model_name):
    """Estrae i dati dei pesi per confronto"""
    print(f"\n🔍 ESTRAZIONE PESI: {model_name}")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    
    weights_data = {
        'model_name': model_name,
        'weights': []
    }
    
    for i, tensor in enumerate(tensor_details):
        # Cerca tensor di pesi (di solito contengono 'qconst' o sono parametri del modello)
        if 'qconst' in tensor['name'] or 'const' in tensor['name']:
            try:
                # Ottieni i dati del tensore
                tensor_data = interpreter.get_tensor(i)
                
                weight_info = {
                    'index': i,
                    'name': tensor['name'],
                    'shape': tensor['shape'].tolist(),
                    'dtype': str(tensor['dtype']),
                    'size': int(np.prod(tensor['shape'])),
                    'data_stats': {
                        'min': float(np.min(tensor_data)),
                        'max': float(np.max(tensor_data)),
                        'mean': float(np.mean(tensor_data)),
                        'std': float(np.std(tensor_data))
                    }
                }
                
                # Aggiungi primi 10 valori per debug
                if tensor_data.size <= 100:
                    weight_info['sample_values'] = tensor_data.flatten()[:10].tolist()
                
                weights_data['weights'].append(weight_info)
                
                print(f"  Peso[{i:2d}]: {tensor['name'][:40]:40} Shape:{str(tensor['shape']):15} Range:[{weight_info['data_stats']['min']:6.1f}, {weight_info['data_stats']['max']:6.1f}]")
                
            except Exception as e:
                print(f"  ❌ Errore lettura peso {tensor['name']}: {e}")
    
    print(f"✅ Estratti {len(weights_data['weights'])} tensori di pesi")
    return weights_data

def generate_cmsis_header(backbone_quant, streaming_quant, backbone_weights, streaming_weights):
    """Genera header C con tutti i parametri di quantizzazione"""
    
    header_content = """#ifndef REAL_QUANTIZATION_PARAMS_H
#define REAL_QUANTIZATION_PARAMS_H

#include <stdint.h>

// ================================================================
// PARAMETRI QUANTIZZAZIONE REALI ESTRATTI DAI TFLITE
// ================================================================

// BACKBONE MODEL QUANTIZATION
"""
    
    # Input/Output quantization del backbone
    backbone_input = None
    backbone_output = None
    
    for tensor in backbone_quant['tensors']:
        if tensor['quantization'] and 'input' in tensor['name'].lower():
            backbone_input = tensor
        elif tensor['quantization'] and 'output' in tensor['name'].lower():
            backbone_output = tensor
    
    if backbone_input:
        header_content += f"""
#define BACKBONE_INPUT_SCALE     {backbone_input['quantization']['scales'][0]:.8f}f
#define BACKBONE_INPUT_ZERO_POINT {backbone_input['quantization']['zero_points'][0]}
"""
    
    if backbone_output:
        header_content += f"""
#define BACKBONE_OUTPUT_SCALE    {backbone_output['quantization']['scales'][0]:.8f}f  
#define BACKBONE_OUTPUT_ZERO_POINT {backbone_output['quantization']['zero_points'][0]}
"""
    
    # Streaming model quantization
    header_content += """
// STREAMING MODEL QUANTIZATION
"""
    
    streaming_inputs = []
    streaming_outputs = []
    
    for tensor in streaming_quant['tensors']:
        if tensor['quantization']:
            if 'input' in tensor['name'].lower() or 'serving_default' in tensor['name']:
                streaming_inputs.append(tensor)
            elif 'output' in tensor['name'].lower() or 'PartitionedCall' in tensor['name']:
                streaming_outputs.append(tensor)
    
    for i, inp in enumerate(streaming_inputs):
        header_content += f"""
#define STREAMING_INPUT_{i}_SCALE      {inp['quantization']['scales'][0]:.8f}f
#define STREAMING_INPUT_{i}_ZERO_POINT {inp['quantization']['zero_points'][0]}
"""
    
    for i, out in enumerate(streaming_outputs):
        header_content += f"""
#define STREAMING_OUTPUT_{i}_SCALE     {out['quantization']['scales'][0]:.8f}f
#define STREAMING_OUTPUT_{i}_ZERO_POINT {out['quantization']['zero_points'][0]}
"""
    
    # Dimensioni esatte
    header_content += f"""
// DIMENSIONI ESATTE DEI MODELLI
#define BACKBONE_INPUT_SHAPE_H    {backbone_input['shape'][1] if backbone_input else 18}
#define BACKBONE_INPUT_SHAPE_W    {backbone_input['shape'][2] if backbone_input else 40}
#define BACKBONE_OUTPUT_SHAPE_H   {backbone_output['shape'][1] if backbone_output else 18}
#define BACKBONE_OUTPUT_SHAPE_C   {backbone_output['shape'][2] if backbone_output else 32}

#define STREAMING_FEATURE_SIZE    {streaming_inputs[0]['shape'][1] if streaming_inputs else 32}
#define STREAMING_HIDDEN_SIZE     {streaming_inputs[1]['shape'][1] if len(streaming_inputs) > 1 else 32}
#define STREAMING_OUTPUT_CLASSES  {streaming_outputs[1]['shape'][1] if len(streaming_outputs) > 1 else 5}

// CONVERSION FUNCTIONS
static inline int8_t float_to_quantized_backbone_input(float value) {{
    return (int8_t)(value / BACKBONE_INPUT_SCALE + BACKBONE_INPUT_ZERO_POINT);
}}

static inline float quantized_to_float_backbone_output(int8_t value) {{
    return (value - BACKBONE_OUTPUT_ZERO_POINT) * BACKBONE_OUTPUT_SCALE;
}}

static inline int8_t float_to_quantized_streaming_input_0(float value) {{
    return (int8_t)(value / STREAMING_INPUT_0_SCALE + STREAMING_INPUT_0_ZERO_POINT);
}}

static inline float quantized_to_float_streaming_output_0(int8_t value) {{
    return (value - STREAMING_OUTPUT_0_ZERO_POINT) * STREAMING_OUTPUT_0_SCALE;
}}

#endif // REAL_QUANTIZATION_PARAMS_H
"""
    
    return header_content

def main():
    """Estrazione completa parametri quantizzazione"""
    print("🔧 ESTRAZIONE PARAMETRI QUANTIZZAZIONE REALI")
    
    # Percorsi modelli
    backbone_path = "convert-and-quantize/streamable-kws/backbone_int8_FINAL.tflite"
    streaming_path = "convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite"
    
    # Estrai quantizzazione
    backbone_quant = extract_all_quantization_params(backbone_path, "BACKBONE")
    streaming_quant = extract_all_quantization_params(streaming_path, "STREAMING")
    
    # Estrai pesi  
    backbone_weights = extract_weights_data(backbone_path, "BACKBONE")
    streaming_weights = extract_weights_data(streaming_path, "STREAMING")
    
    # Salva tutto
    full_data = {
        'backbone_quantization': backbone_quant,
        'streaming_quantization': streaming_quant,
        'backbone_weights': backbone_weights,
        'streaming_weights': streaming_weights
    }
    
    with open('quantization_params.json', 'w') as f:
        json.dump(full_data, f, indent=2, default=str)
    
    # Genera header C
    header_content = generate_cmsis_header(backbone_quant, streaming_quant, backbone_weights, streaming_weights)
    
    with open('real_quantization_params.h', 'w') as f:
        f.write(header_content)
    
    print(f"\n{'='*80}")
    print(f"✅ ESTRAZIONE COMPLETATA:")
    print(f"   - quantization_params.json: Dati completi quantizzazione")
    print(f"   - real_quantization_params.h: Header C per implementazione")
    print(f"   - Backbone: {backbone_quant['quantized_tensors']} tensori quantizzati")
    print(f"   - Streaming: {streaming_quant['quantized_tensors']} tensori quantizzati")
    print(f"   - Pesi backbone: {len(backbone_weights['weights'])} tensori")
    print(f"   - Pesi streaming: {len(streaming_weights['weights'])} tensori")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()