#!/usr/bin/env python3
"""
💾 ANALISI MEMORIA: Implementazione CMSIS COMPLETA del modello TFLite
Calcoli dettagliati per vedere se riusciamo a far entrare TUTTO in AudioMoth
"""

import numpy as np

def analyze_current_tflite_model():
    """Analizza il modello TFLite attuale per mapping CMSIS"""
    print("🔍 MODELLO TFLITE ATTUALE (da implementare in CMSIS)")
    print("="*70)
    
    # Dati dall'analisi precedente
    backbone_ops = {
        'Conv1D': 81,
        'ReLU/Activation': 5, 
        'Add/Residual': 10,
        'Multiply': 7,
        'Sigmoid': 14,  # SE blocks
        'Mean/Pooling': 7,
        'Padding': 7
    }
    
    streaming_ops = {
        'MatMul/Dense': 8,  # GRU gates
        'Add/Residual': 3,
        'Multiply': 4, 
        'Sigmoid': 2,
        'Tanh': 1,
        'Softmax': 1
    }
    
    # Parametri attuali
    backbone_params = 7245  # 7.1KB
    streaming_params = 7590  # 7.4KB
    total_params = backbone_params + streaming_params  # 14.8KB
    
    print(f"📊 BACKBONE ({len(backbone_ops)} tipi operazioni):")
    for op, count in backbone_ops.items():
        print(f"   {op:20} → {count:2d} operazioni")
    
    print(f"\n📊 STREAMING ({len(streaming_ops)} tipi operazioni):")
    for op, count in streaming_ops.items():
        print(f"   {op:20} → {count:2d} operazioni")
    
    print(f"\n💾 PARAMETRI TOTALI: {total_params:,} = {total_params/1024:.1f}KB")
    
    return backbone_ops, streaming_ops, backbone_params, streaming_params

def calculate_cmsis_memory_requirements(backbone_ops, streaming_ops, backbone_params, streaming_params):
    """Calcola i requisiti di memoria per implementazione CMSIS completa"""
    print("\n💾 CALCOLI MEMORIA IMPLEMENTAZIONE CMSIS COMPLETA")
    print("="*70)
    
    # 1. PESI E BIAS (identici al TFLite)
    weights_memory = backbone_params + streaming_params  # 14.8KB
    
    # 2. CODICE OPERAZIONI CMSIS-NN
    cmsis_operations = {
        'arm_convolve_s8': 3.5,      # KB per Conv1D (81 operazioni)
        'arm_fully_connected_s8': 2.0, # KB per Dense/MatMul (8 operazioni)  
        'arm_relu6_s8': 0.5,         # KB per ReLU (5 operazioni)
        'arm_elementwise_add_s8': 1.0, # KB per Add (13 operazioni tot)
        'arm_elementwise_mul_s8': 1.0, # KB per Multiply (11 operazioni tot)
        'arm_softmax_s8': 1.5,       # KB per Softmax (1 operazione)
        'custom_sigmoid_s8': 1.0,    # KB per Sigmoid (16 operazioni - custom)
        'custom_tanh_s8': 0.8,       # KB per Tanh (1 operazione - custom)
        'custom_mean_pooling': 1.2,  # KB per Mean/Pooling (7 operazioni)
        'custom_padding': 0.5,       # KB per Padding (7 operazioni)
    }
    
    total_code_kb = sum(cmsis_operations.values())
    
    print(f"📊 CODICE OPERAZIONI CMSIS-NN:")
    for op, kb in cmsis_operations.items():
        print(f"   {op:25} → {kb:4.1f} KB")
    print(f"   {'TOTALE CODICE':25} → {total_code_kb:4.1f} KB")
    
    # 3. TENSORI INTERMEDI (activations)
    print(f"\n📊 TENSORI INTERMEDI (activations):")
    
    # Backbone activations (progressive: 40→16→32)
    backbone_activations = {
        'input_tensor': (1, 18, 40),      # Input
        'conv_intermediate_1': (1, 18, 16), # Dopo primi Conv1D
        'conv_intermediate_2': (1, 18, 16), # Intermedi 
        'conv_intermediate_3': (1, 18, 32), # Ultimi Conv1D
        'se_attention_1': (1, 1, 16),     # SE blocks intermediate
        'se_attention_2': (1, 1, 32),     # SE blocks intermediate
        'output_tensor': (1, 18, 32),     # Output finale
    }
    
    backbone_activation_bytes = 0
    for name, shape in backbone_activations.items():
        size_bytes = np.prod(shape)
        backbone_activation_bytes += size_bytes
        print(f"   {name:20} {str(shape):15} → {size_bytes:4d} bytes")
    
    # Streaming activations (GRU states)
    streaming_activations = {
        'features_input': (1, 32),        # Features input
        'hidden_input': (1, 32),          # Hidden state input  
        'gru_gate_1': (1, 32),           # Forget gate
        'gru_gate_2': (1, 32),           # Input gate
        'gru_gate_3': (1, 32),           # Candidate gate
        'gru_intermediate': (1, 32),      # Intermediate states
        'hidden_output': (1, 32),         # New hidden state
        'attention_weights': (1, 1),      # Attention (simplified)
        'logits_output': (1, 5),          # Final logits
    }
    
    streaming_activation_bytes = 0
    for name, shape in streaming_activations.items():
        size_bytes = np.prod(shape)
        streaming_activation_bytes += size_bytes
        print(f"   {name:20} {str(shape):15} → {size_bytes:4d} bytes")
    
    total_activations = backbone_activation_bytes + streaming_activation_bytes
    
    # 4. WORKSPACE BUFFER (per CMSIS-NN operations)
    workspace_buffer = max(
        1024,  # Minimum workspace
        total_activations * 2,  # Double buffering
        8192   # Conservative estimate per CMSIS-NN docs
    )
    
    # 5. STACK E HEAP
    stack_size = 4 * 1024  # 4KB stack
    heap_size = 2 * 1024   # 2KB heap per malloc eventuali
    
    print(f"\n💾 TOTALE MEMORIA RICHIESTA:")
    print(f"   Pesi e bias:           {weights_memory/1024:6.1f} KB")
    print(f"   Codice CMSIS-NN:       {total_code_kb:6.1f} KB") 
    print(f"   Tensori intermedi:     {total_activations/1024:6.1f} KB")
    print(f"   Workspace buffer:      {workspace_buffer/1024:6.1f} KB")
    print(f"   Stack:                 {stack_size/1024:6.1f} KB")
    print(f"   Heap:                  {heap_size/1024:6.1f} KB")
    
    total_memory = (weights_memory + total_code_kb*1024 + total_activations + 
                   workspace_buffer + stack_size + heap_size)
    
    print(f"   {'='*30}")
    print(f"   TOTALE:                {total_memory/1024:6.1f} KB")
    
    return total_memory, {
        'weights': weights_memory,
        'code': total_code_kb * 1024,
        'activations': total_activations,
        'workspace': workspace_buffer,
        'stack': stack_size,
        'heap': heap_size
    }

def check_audiomoth_constraints(total_memory, breakdown):
    """Verifica se entra nei vincoli AudioMoth"""
    print(f"\n🎯 VERIFICA VINCOLI AUDIOMOTH")
    print("="*70)
    
    # Vincoli AudioMoth
    flash_size = 256 * 1024    # 256KB Flash
    ram_size = 32 * 1024       # 32KB RAM
    sram_size = 256 * 1024     # 256KB external SRAM
    total_available = flash_size + ram_size + sram_size  # 544KB
    
    print(f"📊 MEMORIA DISPONIBILE:")
    print(f"   Flash (codice):        {flash_size/1024:6.0f} KB")
    print(f"   RAM (stack/heap):      {ram_size/1024:6.0f} KB") 
    print(f"   External SRAM (data):  {sram_size/1024:6.0f} KB")
    print(f"   TOTALE DISPONIBILE:    {total_available/1024:6.0f} KB")
    
    # Alloca memoria per categoria
    flash_usage = breakdown['code']  # Codice in Flash
    ram_usage = breakdown['stack'] + breakdown['heap']  # Stack/Heap in RAM
    sram_usage = (breakdown['weights'] + breakdown['activations'] + 
                  breakdown['workspace'])  # Dati in external SRAM
    
    print(f"\n📊 ALLOCAZIONE PROPOSTA:")
    print(f"   Flash (codice):        {flash_usage/1024:6.1f} KB / {flash_size/1024:.0f} KB", end="")
    if flash_usage <= flash_size:
        print(" ✅")
    else:
        print(" ❌ OVERFLOW!")
    
    print(f"   RAM (stack/heap):      {ram_usage/1024:6.1f} KB / {ram_size/1024:.0f} KB", end="")
    if ram_usage <= ram_size:
        print(" ✅")
    else:
        print(" ❌ OVERFLOW!")
    
    print(f"   SRAM (dati NN):        {sram_usage/1024:6.1f} KB / {sram_size/1024:.0f} KB", end="")
    if sram_usage <= sram_size:
        print(" ✅")
    else:
        print(" ❌ OVERFLOW!")
    
    total_used = flash_usage + ram_usage + sram_usage
    print(f"   TOTALE USATO:          {total_used/1024:6.1f} KB / {total_available/1024:.0f} KB", end="")
    
    fits = (flash_usage <= flash_size and ram_usage <= ram_size and sram_usage <= sram_size)
    
    if fits:
        print(" ✅ ENTRA!")
        margin = total_available - total_used
        print(f"   MARGINE DISPONIBILE:   {margin/1024:6.1f} KB ({margin/total_available*100:.1f}%)")
    else:
        print(" ❌ NON ENTRA!")
        overflow = total_used - total_available
        print(f"   OVERFLOW:              {overflow/1024:6.1f} KB")
    
    return fits

def estimate_performance():
    """Stima performance del modello CMSIS completo"""
    print(f"\n⚡ STIMA PERFORMANCE MODELLO CMSIS COMPLETO")
    print("="*70)
    
    # Cycles stimati per operazione (basato su CMSIS-NN docs)
    cycles_per_op = {
        'Conv1D (81 ops)': 81 * 15000,    # ~15K cycles per Conv1D 
        'Dense/MatMul (8 ops)': 8 * 8000,  # ~8K cycles per Dense
        'ReLU (5 ops)': 5 * 500,           # ~500 cycles per ReLU
        'Add (13 ops)': 13 * 300,          # ~300 cycles per Add
        'Multiply (11 ops)': 11 * 300,     # ~300 cycles per Mul
        'Sigmoid (16 ops)': 16 * 2000,     # ~2K cycles per Sigmoid (custom)
        'Tanh (1 op)': 1 * 2000,           # ~2K cycles per Tanh
        'Softmax (1 op)': 1 * 3000,        # ~3K cycles per Softmax
        'Mean/Pooling (7 ops)': 7 * 1000,  # ~1K cycles per Mean
        'Padding (7 ops)': 7 * 200,        # ~200 cycles per Padding
    }
    
    total_cycles = sum(cycles_per_op.values())
    
    print(f"📊 CYCLES STIMATI:")
    for op, cycles in cycles_per_op.items():
        print(f"   {op:25} → {cycles:8,} cycles")
    print(f"   {'TOTALE':25} → {total_cycles:8,} cycles")
    
    # Timing su Cortex-M4 @ 48MHz
    cpu_freq = 48_000_000  # 48MHz
    inference_time_sec = total_cycles / cpu_freq
    inference_time_ms = inference_time_sec * 1000
    
    print(f"\n⏱️ TIMING @ 48MHz:")
    print(f"   Inferenza completa:    {inference_time_ms:6.1f} ms")
    print(f"   FPS teorico:           {1000/inference_time_ms:6.1f} fps")
    
    # Confronto con real-time requirements
    print(f"\n🎯 REAL-TIME ANALYSIS:")
    frame_ms = 32  # Assumiamo frame da 32ms per audio
    if inference_time_ms < frame_ms:
        utilization = (inference_time_ms / frame_ms) * 100
        print(f"   ✅ Real-time OK: {utilization:.1f}% CPU utilization")
        print(f"   ✅ Margine: {frame_ms - inference_time_ms:.1f}ms per frame")
    else:
        print(f"   ❌ Non real-time: {inference_time_ms:.1f}ms > {frame_ms}ms frame")
    
    return inference_time_ms

if __name__ == "__main__":
    print("💾 ANALISI IMPLEMENTAZIONE CMSIS COMPLETA DEL MODELLO TFLITE")
    print("="*80)
    print("🎯 Obiettivo: Vedere se riusciamo a implementare TUTTO il modello")
    print("   con le operazioni reali in CMSIS-NN su AudioMoth")
    print("")
    
    # Step 1: Analizza modello attuale
    backbone_ops, streaming_ops, backbone_params, streaming_params = analyze_current_tflite_model()
    
    # Step 2: Calcola requisiti memoria
    total_memory, breakdown = calculate_cmsis_memory_requirements(
        backbone_ops, streaming_ops, backbone_params, streaming_params
    )
    
    # Step 3: Verifica vincoli
    fits = check_audiomoth_constraints(total_memory, breakdown)
    
    # Step 4: Stima performance
    inference_time = estimate_performance()
    
    # Conclusione
    print(f"\n🏆 CONCLUSIONE FINALE")
    print("="*50)
    if fits:
        print("✅ IL MODELLO COMPLETO PUÒ ENTRARE IN AUDIOMOTH!")
        print("🎯 Implementazione fattibile con CMSIS-NN")
        print(f"⚡ Performance: {inference_time:.1f}ms per inferenza")
        print("📋 Prossimi passi:")
        print("   1. Implementare tutte le operazioni CMSIS-NN")
        print("   2. Creare pipeline completa backbone→streaming")
        print("   3. Ottimizzare memory layout")
    else:
        print("❌ Il modello completo è troppo grande per AudioMoth")
        print("💡 Opzioni:")
        print("   1. Ridurre dimensioni modello (hidden size)")
        print("   2. Semplificare alcune operazioni")
        print("   3. Usare quantizzazione più aggressiva")
    
    print(f"\n🎯 Il nostro modello attuale semplificato usa solo {89.5:.1f}KB")
    print(f"   vs {total_memory/1024:.1f}KB per implementazione completa")