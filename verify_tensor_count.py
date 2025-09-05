#!/usr/bin/env python3
"""
Verify exact tensor count in streaming model
"""

import tensorflow as tf
import numpy as np
import os

def analyze_streaming_model_exact():
    """Analyze streaming model with exact tensor counting"""
    
    model_path = "./convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite"
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: {model_path} not found")
        return
    
    print("🔍 EXACT STREAMING MODEL ANALYSIS")
    print("=" * 50)
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get all details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print(f"📊 TENSOR COUNTS:")
    print(f"  Input tensors: {len(input_details)}")
    print(f"  Output tensors: {len(output_details)}")
    print(f"  TOTAL tensors: {len(tensor_details)}")
    
    print(f"\n📋 INPUT DETAILS:")
    for i, inp in enumerate(input_details):
        print(f"  [{i}] {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
        print(f"      index={inp['index']}, scale={inp['quantization_parameters']['scales']}")
    
    print(f"\n📋 OUTPUT DETAILS:")
    for i, out in enumerate(output_details):
        print(f"  [{i}] {out['name']}: shape={out['shape']}, dtype={out['dtype']}")
        print(f"      index={out['index']}, scale={out['quantization_parameters']['scales']}")
    
    print(f"\n📋 ALL TENSOR DETAILS:")
    weight_tensors = 0
    activation_tensors = 0
    
    for i, tensor in enumerate(tensor_details):
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        quant = tensor['quantization_parameters']
        
        # Try to get tensor data to see if it's a weight
        try:
            data = interpreter.get_tensor(tensor['index'])
            if data is not None and data.size > 0:
                # This is likely a weight tensor
                weight_tensors += 1
                tensor_type = "WEIGHT"
                print(f"  [{i:2d}] {tensor_type:6s} {name}")
                print(f"       Shape: {shape}, Type: {dtype}")
                print(f"       Data: min={data.min()}, max={data.max()}, std={data.std():.4f}")
                print(f"       Scale: {quant['scales']}, Zero: {quant['zero_points']}")
            else:
                activation_tensors += 1
                tensor_type = "ACTIV"
                print(f"  [{i:2d}] {tensor_type:6s} {name}")
                print(f"       Shape: {shape}, Type: {dtype}")
                print(f"       Scale: {quant['scales']}, Zero: {quant['zero_points']}")
        except:
            activation_tensors += 1
            tensor_type = "ACTIV"
            print(f"  [{i:2d}] {tensor_type:6s} {name}")
            print(f"       Shape: {shape}, Type: {dtype}")
            print(f"       Scale: {quant['scales']}, Zero: {quant['zero_points']}")
        
        print()
    
    print(f"\n📈 FINAL SUMMARY:")
    print(f"  Total tensors: {len(tensor_details)}")
    print(f"  Weight tensors: {weight_tensors}")
    print(f"  Activation tensors: {activation_tensors}")
    
    # Check model operations
    try:
        # Try to get operation details if possible
        print(f"\n🔧 OPERATION ANALYSIS:")
        
        # Run a dummy inference to see data flow
        # Set up dummy inputs
        for inp in input_details:
            dummy_input = np.zeros(inp['shape'], dtype=inp['dtype'].type)
            interpreter.set_tensor(inp['index'], dummy_input)
        
        # Run inference
        interpreter.invoke()
        
        # Check outputs
        print(f"  Inference successful with {len(input_details)} inputs → {len(output_details)} outputs")
        
        for i, out in enumerate(output_details):
            output_data = interpreter.get_tensor(out['index'])
            print(f"  Output[{i}]: shape={output_data.shape}, range=[{output_data.min()}, {output_data.max()}]")
            
    except Exception as e:
        print(f"  Could not run inference: {e}")
    
    return len(tensor_details), weight_tensors, activation_tensors

def main():
    total, weights, activations = analyze_streaming_model_exact()
    
    print(f"\n🎯 VERIFICATION RESULT:")
    print(f"  The streaming model has EXACTLY {total} tensors")
    print(f"  - {weights} weight/constant tensors")
    print(f"  - {activations} activation/intermediate tensors")
    
    if total != 47:
        print(f"  ⚠️  WARNING: Expected 47, but found {total}!")
    else:
        print(f"  ✅ Confirmed: 47 tensors as expected")

if __name__ == "__main__":
    main()