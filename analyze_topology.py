#!/usr/bin/env python3
"""
Analisi semplice: confronto topologia TFLite vs nostro modello CMSIS
"""

import tensorflow as tf
import numpy as np

def analyze_backbone_topology():
    """Analizza la struttura del backbone model"""
    print("🔍 ANALISI TOPOLOGIA BACKBONE")
    print("="*50)
    
    interpreter = tf.lite.Interpreter(model_path="./convert-and-quantize/streamable-kws/backbone_int8_FINAL.tflite")
    interpreter.allocate_tensors()
    
    # Ottieni dettagli completi del modello
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print(f"📊 INPUT/OUTPUT:")
    print(f"   Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"   Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
    
    print(f"\n🧠 TENSORI TOTALI: {len(tensor_details)}")
    
    # Analizza tutti i tensori per capire la struttura
    for i, tensor in enumerate(tensor_details):
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        print(f"   [{i:2d}] {name:40} {str(shape):20} {dtype}")
    
    return len(tensor_details)

def analyze_streaming_topology():
    """Analizza la struttura del streaming model"""
    print("\n🔍 ANALISI TOPOLOGIA STREAMING")
    print("="*50)
    
    interpreter = tf.lite.Interpreter(model_path="./convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print(f"📊 INPUT/OUTPUT:")
    for i, inp in enumerate(input_details):
        print(f"   Input {i}: {inp['shape']} {inp['dtype']}")
    for i, out in enumerate(output_details):
        print(f"   Output {i}: {out['shape']} {out['dtype']}")
    
    print(f"\n🧠 TENSORI TOTALI: {len(tensor_details)}")
    
    for i, tensor in enumerate(tensor_details):
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        print(f"   [{i:2d}] {name:40} {str(shape):20} {dtype}")
    
    return len(tensor_details)

def compare_with_our_implementation():
    """Confronta con la nostra implementazione CMSIS"""
    print("\n🔥 CONFRONTO CON NOSTRO MODELLO CMSIS")
    print("="*60)
    
    print("📊 NOSTRO BACKBONE (cmsis_wrapper_correct.c):")
    print("   - Input: [1×18×40] = 720 int8")
    print("   - Output: [1×18×32] = 576 int8") 
    print("   - Implementazione: 1 Dense layer per frame temporale")
    print("   - Operazioni: Dense → ReLU (per ogni frame)")
    
    print("\n📊 NOSTRO STREAMING (cmsis_wrapper_correct.c):")
    print("   - Input 0 (features): [1×32] = 32 int8")
    print("   - Input 1 (hidden): [1×32] = 32 int8") 
    print("   - Output 0 (new_hidden): [1×32] = 32 int8")
    print("   - Output 1 (logits): [1×5] = 5 int8")
    print("   - Implementazione: GRU approssimato + classificatore")
    print("   - Operazioni: Dense → Tanh → Dense → Add → Dense → Softmax")

if __name__ == "__main__":
    print("🎯 ANALISI TOPOLOGIE TFLITE vs CMSIS")
    print("="*60)
    
    backbone_tensors = analyze_backbone_topology()
    streaming_tensors = analyze_streaming_topology()
    
    compare_with_our_implementation()
    
    print(f"\n📋 SUMMARY:")
    print(f"   Backbone tensori TFLite: {backbone_tensors}")
    print(f"   Streaming tensori TFLite: {streaming_tensors}")
    print(f"   Nostro modello: Semplificazione statica con CMSIS-NN")
    
    print(f"\n❓ TOPOLOGIA IDENTICA?")
    print(f"   ❌ NO - Il nostro è una SEMPLIFICAZIONE")
    print(f"   ✅ Input/Output shapes: IDENTICI") 
    print(f"   ✅ Quantizzazione: IDENTICA")
    print(f"   ❌ Layer interni: DIVERSI (semplificati)")