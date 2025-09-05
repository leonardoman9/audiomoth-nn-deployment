#!/usr/bin/env python3
"""
Compara la struttura di backbone vs streaming models
"""
import tensorflow as tf

def analyze_tflite_model(model_path):
    """Analizza un modello TFLite e stampa le info dettagliate"""
    print(f"\n{'='*50}")
    print(f"ANALYZING: {model_path}")
    print(f"{'='*50}")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Basic model info
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Schema version: {"Unknown"}")
    print(f"Input tensors: {len(input_details)}")
    print(f"Output tensors: {len(output_details)}")
    
    # Input details
    for i, detail in enumerate(input_details):
        print(f"Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")
        print(f"  Quantization: scale={detail.get('quantization_parameters', {}).get('scales', 'None')}")
        print(f"  Zero point: {detail.get('quantization_parameters', {}).get('zero_points', 'None')}")
    
    # Output details 
    for i, detail in enumerate(output_details):
        print(f"Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")
        print(f"  Quantization: scale={detail.get('quantization_parameters', {}).get('scales', 'None')}")
        print(f"  Zero point: {detail.get('quantization_parameters', {}).get('zero_points', 'None')}")
    
    # Tensors totali
    tensor_details = interpreter.get_tensor_details()
    print(f"Total tensors: {len(tensor_details)}")
    
    # Operations
    ops = []
    for detail in tensor_details:
        if 'op_name' in detail:
            ops.append(detail['op_name'])
    
    unique_ops = set(ops)
    print(f"Unique operations: {len(unique_ops)}")
    print(f"Operations: {sorted(unique_ops)}")
    
    # File size analysis
    import os
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size} bytes ({file_size/1024:.1f} KB)")
    
    return {
        'input_details': input_details,
        'output_details': output_details,
        'tensor_count': len(tensor_details),
        'file_size': file_size,
        'operations': unique_ops
    }

if __name__ == "__main__":
    # Analyze both models
    backbone_info = analyze_tflite_model("backbone_int8_FINAL.tflite")
    streaming_info = analyze_tflite_model("streaming_int8_FINAL.tflite")
    
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    print(f"Backbone tensors: {backbone_info['tensor_count']}")
    print(f"Streaming tensors: {streaming_info['tensor_count']}")
    print(f"Backbone size: {backbone_info['file_size']} bytes")
    print(f"Streaming size: {streaming_info['file_size']} bytes")
    
    # Check for differences in operations
    backbone_ops = backbone_info['operations']
    streaming_ops = streaming_info['operations']
    
    common_ops = backbone_ops & streaming_ops
    backbone_only = backbone_ops - streaming_ops
    streaming_only = streaming_ops - backbone_ops
    
    print(f"Common operations: {len(common_ops)}")
    print(f"Backbone-only operations: {backbone_only}")
    print(f"Streaming-only operations: {streaming_only}")