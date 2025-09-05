#!/usr/bin/env python3
"""
Analisi semplificata dei modelli TFLite per AudioMoth NN deployment
"""

import tensorflow as tf
import numpy as np
import json

def analyze_simple(tflite_path, model_name):
    """Analisi semplificata senza schema complesso"""
    print(f"\n{'='*60}")
    print(f"📊 ANALISI {model_name} MODEL")
    print(f"File: {tflite_path}")
    print(f"{'='*60}")
    
    # Carica interprete
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Input/Output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n🎯 INPUT:")
    for inp in input_details:
        quant = inp.get('quantization_parameters', {})
        scale = quant.get('scales', [])
        zero_point = quant.get('zero_points', [])
        print(f"  Name: {inp['name']}")
        print(f"  Shape: {inp['shape']}")
        print(f"  Type: {inp['dtype']}")
        if len(scale) > 0:
            print(f"  Quantization: scale={scale[0]:.8f}, zero_point={zero_point[0]}")
    
    print(f"\n🎯 OUTPUT:")
    for out in output_details:
        quant = out.get('quantization_parameters', {})
        scale = quant.get('scales', [])
        zero_point = quant.get('zero_points', [])
        print(f"  Name: {out['name']}")
        print(f"  Shape: {out['shape']}")
        print(f"  Type: {out['dtype']}")
        if len(scale) > 0:
            print(f"  Quantization: scale={scale[0]:.8f}, zero_point={zero_point[0]}")
    
    # Tensor details per quantizzazione
    tensor_details = interpreter.get_tensor_details()
    
    print(f"\n🔧 QUANTIZATION SUMMARY:")
    print(f"  Total tensors: {len(tensor_details)}")
    
    # Conta tensor con quantizzazione
    quant_tensors = 0
    scales = []
    zero_points = []
    
    for tensor in tensor_details:
        if 'quantization_parameters' in tensor:
            quant = tensor['quantization_parameters']
            if quant['scales'].size > 0:
                quant_tensors += 1
                scales.extend(quant['scales'])
                zero_points.extend(quant['zero_points'])
    
    print(f"  Quantized tensors: {quant_tensors}")
    if scales:
        print(f"  Scale range: {min(scales):.8f} - {max(scales):.8f}")
        print(f"  Zero-point range: {min(zero_points)} - {max(zero_points)}")
    
    return {
        'model_name': model_name,
        'input_details': input_details,
        'output_details': output_details,
        'tensor_count': len(tensor_details),
        'quantized_count': quant_tensors
    }

def compare_topologies():
    """Confronta con il modello Python"""
    print(f"\n{'='*80}")
    print(f"🧠 CONFRONTO CON MODELLO PYTHON ORIGINALE")
    print(f"{'='*80}")
    
    print("""
    🔍 MODELLO PYTHON - Improved_Phi_GRU_ATT:
    
    INPUT: Audio waveform [batch, time_samples]
       ↓
    SPECTROGRAM: 
       - Tipo: mel/linear_stft/linear_triangular/combined_log_linear
       - n_fft=400, hop_length=160  
       - Output: [batch, freq_bins, time_frames]
       - AmplitudeToDB + Normalizzazione
       ↓
    BACKBONE (MatchboxNet):
       - Input: [batch, features, time]
       - MatchboxNetSkip con separable convolutions
       - Output: [batch, 32, time_frames] 
       ↓
    STREAMING:
       - Permute a [batch, time, 32]
       - GRU(input_size=32, hidden_size=32, bidirectional=False)
       - Projection: Linear(32, 32)
       - AttentionLayer(32) 
       - Classification: Linear(32, num_classes)
       ↓
    OUTPUT: [batch, num_classes] logits
    """)

def main():
    """Analisi completa"""
    print("🚀 ANALISI SEMPLIFICATA AUDIOMOTH-NN MODELS")
    
    # Percorsi modelli
    backbone_path = "convert-and-quantize/streamable-kws/backbone_int8_FINAL.tflite"
    streaming_path = "convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite"
    
    # Analizza entrambi
    backbone_analysis = analyze_simple(backbone_path, "BACKBONE")
    streaming_analysis = analyze_simple(streaming_path, "STREAMING")
    
    # Confronto topologia
    compare_topologies()
    
    # Riassunto CRITICO  
    print(f"\n{'='*80}")
    print(f"🎯 DIFFERENZE CRITICHE SCOPERTE:")
    print(f"{'='*80}")
    
    print(f"📐 BACKBONE TFLite:")
    print(f"   Input shape: {backbone_analysis['input_details'][0]['shape']}")  
    print(f"   Output shape: {backbone_analysis['output_details'][0]['shape']}")
    print(f"   ❌ PROBLEMA: Input [1,18,40] NON è spettrogramma raw!")
    print(f"   ❌ PROBLEMA: Output [1,18,32] NON è [1,32,time]!")
    
    print(f"\n📐 STREAMING TFLite:")
    print(f"   Input shape: {streaming_analysis['input_details'][0]['shape']}")
    print(f"   Output shape: {streaming_analysis['output_details'][0]['shape']}")
    print(f"   ❌ PROBLEMA: Diverso dal GRU atteso!")
    
    print(f"\n💡 CONCLUSIONI:")
    print(f"   1. I modelli TFLite sono PRE-PROCESSATI")
    print(f"   2. Input backbone: spettrogramma già calcolato [18,40]")
    print(f"   3. Output backbone: features intermedie [18,32]")
    print(f"   4. La nostra implementazione attuale è SBAGLIATA")
    print(f"   5. Dobbiamo replicare ESATTAMENTE questa topologia!")
    
    # Salva analisi
    analysis = {
        'backbone': backbone_analysis,
        'streaming': streaming_analysis,
        'critical_findings': {
            'backbone_input_shape': backbone_analysis['input_details'][0]['shape'].tolist(),
            'backbone_output_shape': backbone_analysis['output_details'][0]['shape'].tolist(),
            'streaming_input_shape': streaming_analysis['input_details'][0]['shape'].tolist(),
            'streaming_output_shape': streaming_analysis['output_details'][0]['shape'].tolist(),
            'key_issue': 'Models expect preprocessed spectrogram input, not raw audio'
        }
    }
    
    with open('model_analysis_simple.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n✅ Analisi salvata in 'model_analysis_simple.json'")

if __name__ == "__main__":
    main()