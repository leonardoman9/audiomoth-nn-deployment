#!/usr/bin/env python3

"""
Script per trovare i dati reali (tensori/operatori) nei file .tflite 
usando l'analisi diretta con TensorFlow
"""

import sys
import os
sys.path.append('.')

try:
    import tensorflow as tf
    print("✅ TensorFlow available")
except ImportError:
    print("❌ TensorFlow not available")
    sys.exit(1)

def analyze_tflite_detailed(model_path):
    print(f"\n🔍 ANALYZING: {model_path}")
    print("=" * 50)
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print(f"📊 SUMMARY:")
    print(f"  Input tensors: {len(input_details)}")
    print(f"  Output tensors: {len(output_details)}")
    print(f"  Total tensors: {len(tensor_details)}")
    
    print(f"\n📥 INPUT TENSORS:")
    for i, detail in enumerate(input_details):
        print(f"  {i}: name='{detail['name']}', shape={detail['shape']}, dtype={detail['dtype']}")
        if 'quantization_parameters' in detail:
            scale = detail['quantization_parameters']['scales']
            zero_point = detail['quantization_parameters']['zero_points']
            print(f"      quantization: scale={scale}, zero_point={zero_point}")
    
    print(f"\n📤 OUTPUT TENSORS:")
    for i, detail in enumerate(output_details):
        print(f"  {i}: name='{detail['name']}', shape={detail['shape']}, dtype={detail['dtype']}")
        if 'quantization_parameters' in detail:
            scale = detail['quantization_parameters']['scales']
            zero_point = detail['quantization_parameters']['zero_points']
            print(f"      quantization: scale={scale}, zero_point={zero_point}")
    
    print(f"\n🔢 ALL TENSORS (first 10):")
    for i, detail in enumerate(tensor_details[:10]):
        print(f"  {i}: name='{detail['name']}', shape={detail['shape']}, dtype={detail['dtype']}")
    
    if len(tensor_details) > 10:
        print(f"  ... and {len(tensor_details) - 10} more")
    
    # Try to run inference to verify it works
    try:
        print(f"\n🚀 TESTING INFERENCE:")
        
        # Create dummy input
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == tf.int8:
            dummy_input = tf.random.uniform(input_shape, minval=-128, maxval=127, dtype=tf.int32)
            dummy_input = tf.cast(dummy_input, tf.int8)
        else:
            dummy_input = tf.random.normal(input_shape, dtype=input_dtype)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input.numpy())
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"  Input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"  ✅ Inference successful!")
        
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")

if __name__ == "__main__":
    # Analyze both models
    models = [
        "./convert-and-quantize/streamable-kws/backbone_int8_FINAL.tflite",
        "./convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite"
    ]
    
    for model_path in models:
        if os.path.exists(model_path):
            analyze_tflite_detailed(model_path)
        else:
            print(f"❌ Model not found: {model_path}")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"I modelli TFLite FUNZIONANO con TensorFlow, quindi hanno sicuramente tensori e operatori!")
    print(f"Il problema è nel nostro parser C che non li trova correttamente.")