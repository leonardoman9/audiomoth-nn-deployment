#!/usr/bin/env python3
"""
🔍 STEP 1: Analizzare dimensioni attuali del modello TFLite
per capire quanto possiamo ridurre mantenendo la stessa architettura
"""

import tensorflow as tf
import numpy as np
import os

def analyze_model_weights(model_path, model_name):
    """Analizza i pesi del modello per capire dove ridurre"""
    print(f"\n🔍 ANALISI DETTAGLIATA: {model_name}")
    print("="*60)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    
    total_params = 0
    weight_breakdown = {}
    
    for tensor in tensor_details:
        if 'pseudo_qconst' in tensor['name'] and tensor['shape'] is not None:
            shape = tensor['shape']
            if len(shape) > 0:  # Skip scalar constants
                size = np.prod(shape)
                total_params += size
                
                # Categorizza i pesi
                if len(shape) == 4:  # Conv weights [out, h, w, in]
                    category = f"Conv weights {shape}"
                elif len(shape) == 2:  # Dense weights [out, in]  
                    category = f"Dense weights {shape}"
                elif len(shape) == 1:  # Bias
                    category = f"Bias {shape}"
                else:
                    category = f"Other {shape}"
                
                if category not in weight_breakdown:
                    weight_breakdown[category] = 0
                weight_breakdown[category] += size
    
    print(f"📊 PARAMETRI TOTALI: {total_params:,}")
    print(f"💾 MEMORIA PESI: {total_params * 1} bytes = {total_params/1024:.1f} KB")
    
    print(f"\n🔧 BREAKDOWN PER CATEGORIA:")
    for category, count in sorted(weight_breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100
        print(f"   {category:40} → {count:6,} params ({percentage:5.1f}%)")
    
    return total_params, weight_breakdown

def analyze_operations(model_path, model_name):
    """Analizza le operazioni per capire la complessità"""
    print(f"\n⚙️ ANALISI OPERAZIONI: {model_name}")
    print("="*60)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    
    operations = {}
    for tensor in tensor_details:
        name = tensor['name']
        
        # Categorizza le operazioni dal nome
        if 'conv1d' in name:
            op_type = 'Conv1D'
        elif 'matmul' in name.lower():
            op_type = 'MatMul/Dense'
        elif 'relu' in name:
            op_type = 'ReLU/Activation'
        elif 'sigmoid' in name:
            op_type = 'Sigmoid'
        elif 'tanh' in name:
            op_type = 'Tanh'
        elif 'softmax' in name:
            op_type = 'Softmax'
        elif 'add' in name.lower():
            op_type = 'Add/Residual'
        elif 'mul' in name.lower():
            op_type = 'Multiply'
        elif 'mean' in name:
            op_type = 'Mean/Pooling'
        elif 'pad' in name:
            op_type = 'Padding'
        else:
            op_type = 'Other'
        
        if op_type not in operations:
            operations[op_type] = 0
        operations[op_type] += 1
    
    print(f"📊 OPERAZIONI TOTALI: {len(tensor_details)}")
    print(f"\n🔧 BREAKDOWN PER TIPO:")
    for op_type, count in sorted(operations.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(tensor_details)) * 100
        print(f"   {op_type:20} → {count:3d} operazioni ({percentage:5.1f}%)")
    
    return operations

def estimate_reduction_potential(backbone_params, streaming_params, backbone_ops, streaming_ops):
    """Stima il potenziale di riduzione"""
    print(f"\n💡 STIMA POTENZIALE DI RIDUZIONE")
    print("="*60)
    
    # Calcola riduzioni possibili
    reductions = {
        "50% hidden size": {
            "backbone": backbone_params * 0.25,  # 40→20 = 50% riduzione, ma è quadratica per Dense
            "streaming": streaming_params * 0.25,  # 32→16 = 50% riduzione, quadratica per MatMul
            "description": "Hidden: backbone 40→20, streaming 32→16"
        },
        "75% hidden size": {
            "backbone": backbone_params * 0.56,  # 40→30, meno drastico  
            "streaming": streaming_params * 0.56,  # 32→24
            "description": "Hidden: backbone 40→30, streaming 32→24"
        },
        "Half MatchboxNet blocks": {
            "backbone": backbone_params * 0.5,  # Metà dei layer
            "streaming": streaming_params,  # Streaming invariato
            "description": "MatchboxNet: metà dei blocchi Conv1D"
        },
        "Combined reduction": {
            "backbone": backbone_params * 0.25 * 0.5,  # Hidden + blocks
            "streaming": streaming_params * 0.25,
            "description": "Entrambe le riduzioni combinate"
        }
    }
    
    print(f"📊 SITUAZIONE ATTUALE:")
    print(f"   Backbone: {backbone_params:,} params")
    print(f"   Streaming: {streaming_params:,} params")
    print(f"   TOTALE: {backbone_params + streaming_params:,} params")
    
    for reduction_name, reduction in reductions.items():
        new_backbone = int(reduction["backbone"])
        new_streaming = int(reduction["streaming"])
        total_new = new_backbone + new_streaming
        total_original = backbone_params + streaming_params
        savings = ((total_original - total_new) / total_original) * 100
        
        print(f"\n🎯 {reduction_name}:")
        print(f"   Backbone: {backbone_params:,} → {new_backbone:,}")
        print(f"   Streaming: {streaming_params:,} → {new_streaming:,}")
        print(f"   TOTALE: {total_original:,} → {total_new:,} (-{savings:.0f}%)")
        print(f"   Descrizione: {reduction['description']}")
        
        # Stima memoria TFLM
        weights_kb = total_new / 1024
        tflm_overhead = 80  # KB stimati per TFLM
        arena_kb = max(30, weights_kb * 3)  # Arena proporzionale ai pesi
        total_memory = weights_kb + tflm_overhead + arena_kb
        
        print(f"   💾 Memoria stimata: {total_memory:.0f}KB", end="")
        if total_memory < 250:
            print(" ✅ ENTRA in AudioMoth!")
        else:
            print(" ❌ Troppo grande")

def suggest_model_modifications():
    """Suggerisce modifiche concrete al modello"""
    print(f"\n🚀 MODIFICHE SUGGERITE AL MODELLO")
    print("="*60)
    
    print(f"📝 Per il training script:")
    print(f"1. HIDDEN SIZES:")
    print(f"   backbone_hidden = 16  # invece di 32")
    print(f"   streaming_hidden = 16  # invece di 32")
    print(f"   output_features = 16  # invece di 32")
    
    print(f"\n2. MATCHBOXNET BLOCKS:")
    print(f"   num_blocks = 3  # invece di 6-8")
    print(f"   channels = [40, 8, 16]  # invece di [40, 16, 32]")
    
    print(f"\n3. CONV KERNELS:")
    print(f"   kernel_sizes = [3, 3, 3]  # invece di [3, 5, 7]")
    print(f"   reduce_se_blocks = True  # Semplificare attention")
    
    print(f"\n4. QUANTIZATION:")
    print(f"   # Mantenere int8, è già efficiente")
    print(f"   representative_dataset_size = 100  # Meno campioni")
    
    print(f"\n🎯 RISULTATO ATTESO:")
    print(f"   • STESSA architettura (MatchboxNet + GRU)")
    print(f"   • ~70-80% meno parametri")  
    print(f"   • ~150-180KB memoria totale")
    print(f"   • Accuracy: ~85-90% del modello originale")

if __name__ == "__main__":
    print("🎯 ANALISI MODELLO TFLITE PER RIDUZIONE")
    print("="*80)
    
    # Paths ai modelli
    backbone_path = "./convert-and-quantize/streamable-kws/backbone_int8_FINAL.tflite"
    streaming_path = "./convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite"
    
    # Verifica che i file esistano
    if not os.path.exists(backbone_path):
        print(f"❌ File non trovato: {backbone_path}")
        exit(1)
    if not os.path.exists(streaming_path):
        print(f"❌ File non trovato: {streaming_path}")
        exit(1)
    
    # Analizza entrambi i modelli
    backbone_params, backbone_weights = analyze_model_weights(backbone_path, "BACKBONE")
    backbone_ops = analyze_operations(backbone_path, "BACKBONE")
    
    streaming_params, streaming_weights = analyze_model_weights(streaming_path, "STREAMING")
    streaming_ops = analyze_operations(streaming_path, "STREAMING")
    
    # Stima riduzioni
    estimate_reduction_potential(backbone_params, streaming_params, backbone_ops, streaming_ops)
    
    # Suggerimenti
    suggest_model_modifications()
    
    print(f"\n✅ ANALISI COMPLETATA!")
    print(f"🎯 Prossimo step: modificare il training script con le dimensioni ridotte")