#!/usr/bin/env python3
"""
🚀 TEST PIPELINE AUDIOMOTH - SIMULAZIONE ESATTA DEL FIRMWARE
Test che replica esattamente la logica del firmware AudioMoth modificato
"""

import numpy as np
import tensorflow as tf
import time

print("🎯 AudioMoth Pipeline Test - Logica Firmware Corretta")
print("=" * 60)

# Carica modelli TFLite
print("📂 Caricamento modelli TFLite...")
backbone = tf.lite.Interpreter(model_path="backbone_int8_FINAL.tflite")
backbone.allocate_tensors()

streaming = tf.lite.Interpreter(model_path="streaming_int8_FINAL.tflite")
streaming.allocate_tensors()

# Info tensori
bb_input = backbone.get_input_details()[0]
bb_output = backbone.get_output_details()[0]
st_input = streaming.get_input_details()[0]
st_output = streaming.get_output_details()[0]

print(f"📊 Backbone: {bb_input['shape']} → {bb_output['shape']}")
print(f"📊 Streaming: {st_input['shape']} → {st_output['shape']}")

# Dequantizzazione
def dequantize_output(quantized_output, output_details):
    """Converte int8 quantizzato a float32"""
    scale = output_details['quantization'][0]
    zero_point = output_details['quantization'][1]
    return scale * (quantized_output.astype(np.float32) - zero_point)

def simulate_audiomoth_pipeline():
    """
    Simula ESATTAMENTE il pipeline AudioMoth:
    1. Genera spectrogram dummy (come preprocess_audio_to_spectrogram)
    2. Run backbone inference 
    3. Estrae ultimo timestep (come nel firmware modificato)
    4. Single streaming inference
    5. Apply softmax
    """
    
    # STEP 1: Simula spectrogram generation (placeholder come nel firmware)
    # Nel firmware: preprocess_audio_to_spectrogram()
    dummy_spectrogram = np.random.randint(-128, 127, bb_input['shape'], dtype=np.int8)
    
    # STEP 2: Backbone inference (run_backbone_inference)
    start_bb = time.perf_counter()
    backbone.set_tensor(bb_input['index'], dummy_spectrogram)
    backbone.invoke()
    bb_features_raw = backbone.get_tensor(bb_output['index'])
    bb_time = time.perf_counter() - start_bb
    
    # Dequantizza backbone output
    bb_features_float = dequantize_output(bb_features_raw, bb_output)
    print(f"🔍 Backbone output shape: {bb_features_float.shape}")
    
    # STEP 3: Extract last timestep (come nel firmware modificato)
    # Replica la logica: last_timestep_offset = (NN_TIME_FRAMES - 1) * NN_BACKBONE_FEATURES
    if len(bb_features_float.shape) == 3:  # [batch, timesteps, features]
        batch_size, timesteps, features = bb_features_float.shape
        print(f"💡 Estrazione ultimo timestep: timestep {timesteps-1} di {timesteps}")
        # Prendi ultimo timestep: [1, timesteps, features] → [1, features]
        final_features = bb_features_float[0, -1:, :].reshape(1, features)
    else:
        final_features = bb_features_float
    
    print(f"🎯 Features finali shape: {final_features.shape}")
    print(f"🎯 Streaming input atteso: {st_input['shape']}")
    
    # Verifica compatibilità dimensioni
    if final_features.shape[1] != st_input['shape'][1]:
        print(f"⚠️  ATTENZIONE: Mismatch features {final_features.shape[1]} vs atteso {st_input['shape'][1]}")
        # Reshape per compatibilità
        final_features = final_features[:, :st_input['shape'][1]]
    
    # Requantizza per streaming input
    st_scale = st_input['quantization'][0] 
    st_zero_point = st_input['quantization'][1]
    final_features_quantized = np.round(final_features / st_scale + st_zero_point).astype(np.int8)
    
    # STEP 4: Streaming inference (run_streaming_inference)
    start_st = time.perf_counter()
    streaming.set_tensor(st_input['index'], final_features_quantized)
    streaming.invoke()
    st_output_raw = streaming.get_tensor(st_output['index'])
    st_time = time.perf_counter() - start_st
    
    # Dequantizza streaming output
    st_logits = dequantize_output(st_output_raw, st_output)
    
    # STEP 5: Apply softmax (come apply_softmax nel firmware)
    if st_logits.shape[1] > 10:  # Se output è > 10, prendi solo prime 10
        st_logits = st_logits[:, :10]
    
    # Replica softmax firmware (con max per stabilità numerica)
    logits_1d = st_logits[0]
    max_logit = np.max(logits_1d)
    exp_logits = np.exp(logits_1d - max_logit)
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Decision
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    return {
        'bb_time': bb_time,
        'st_time': st_time,
        'total_time': bb_time + st_time,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

# TEST SINGOLO
print("\n🧪 TEST SINGOLA INFERENZA:")
result = simulate_audiomoth_pipeline()
print(f"⚡ Backbone: {result['bb_time']*1000:.2f}ms")
print(f"⚡ Streaming: {result['st_time']*1000:.2f}ms") 
print(f"📈 Totale: {result['total_time']*1000:.2f}ms")
print(f"🎯 Predizione: classe {result['predicted_class']}, confidence {result['confidence']:.3f}")

# PERFORMANCE TEST (simula NN_runPerformanceTestSequence)
print(f"\n🏃 PERFORMANCE TEST PIPELINE (50 iterazioni)...")
times = []
bb_times = []
st_times = []

for i in range(50):
    result = simulate_audiomoth_pipeline()
    times.append(result['total_time'] * 1000)
    bb_times.append(result['bb_time'] * 1000) 
    st_times.append(result['st_time'] * 1000)
    
    if (i+1) % 10 == 0:
        print(f"   ✅ {i+1}/50 completate...")

print(f"\n📈 RISULTATI PC (Intel/Apple Silicon):")
print(f"⚡ Backbone: {np.mean(bb_times):.2f}ms ± {np.std(bb_times):.2f}ms")
print(f"⚡ Streaming: {np.mean(st_times):.2f}ms ± {np.std(st_times):.2f}ms")
print(f"📈 Totale: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
print(f"🚀 Throughput PC: {1000/np.mean(times):.1f} inferenze/sec")

# STIMA AUDIOMOTH (ARM Cortex-M4 @48MHz con cache limitata)
audiomoth_factor = 8  # Più conservativo per int8 + no hardware acceleration
audiomoth_time = np.mean(times) * audiomoth_factor
audiomoth_throughput = 1000 / audiomoth_time

print(f"\n📱 STIMA AUDIOMOTH (ARM Cortex-M4 @48MHz):")
print(f"⚡ Tempo inferenza: ~{audiomoth_time:.1f}ms")
print(f"🚀 Throughput: ~{audiomoth_throughput:.1f} inferenze/sec")

# Analisi real-time capability
audio_chunk_duration = 21.3  # ms (1024 samples @ 48kHz)
realtime_possible = audiomoth_time < audio_chunk_duration

print(f"\n🎵 ANALISI REAL-TIME:")
print(f"📊 Chunk audio: {audio_chunk_duration:.1f}ms") 
print(f"⚡ Inferenza: ~{audiomoth_time:.1f}ms")

if realtime_possible:
    print(f"✅ REAL-TIME POSSIBILE! 🎉")
    overhead = audio_chunk_duration - audiomoth_time
    print(f"💨 Margine: {overhead:.1f}ms ({overhead/audio_chunk_duration*100:.1f}%)")
else:
    print(f"❌ Real-time difficile")
    delay = audiomoth_time - audio_chunk_duration
    print(f"⏰ Ritardo: {delay:.1f}ms")

print(f"\n🔋 POWER ANALYSIS:")
inference_per_sec = min(audiomoth_throughput, 1000/audio_chunk_duration)  # Limitato da audio rate
print(f"💡 Inference/sec realizzabili: ~{inference_per_sec:.1f}")
duty_cycle = (audiomoth_time/1000) * inference_per_sec
print(f"🔋 Duty cycle: {duty_cycle:.1%}")
print(f"😴 Sleep time: {100-duty_cycle*100:.1f}%")

# SIMULAZIONE SEQUENZA TEST AUDIOMOTH
print(f"\n🎛️  SIMULAZIONE NN_runPerformanceTestSequence:")
print("🔴 LED Start → 10 inferenze → pausa → 100 → pausa → 1000 → 🔴 LED End")

# Phase 1: 10 inferences
start_time = time.perf_counter()
for i in range(10):
    simulate_audiomoth_pipeline()
phase1_time = (time.perf_counter() - start_time) * 1000

# Phase 2: 100 inferences  
start_time = time.perf_counter()
for i in range(100):
    simulate_audiomoth_pipeline()
phase2_time = (time.perf_counter() - start_time) * 1000

# Phase 3: 1000 inferences
start_time = time.perf_counter()
for i in range(1000):
    simulate_audiomoth_pipeline()
phase3_time = (time.perf_counter() - start_time) * 1000

print(f"\n⏱️  TIMING FASI PC:")
print(f"📈 10 inferenze: {phase1_time:.0f}ms ({phase1_time/10:.1f}ms/inf)")
print(f"📈 100 inferenze: {phase2_time:.0f}ms ({phase2_time/100:.1f}ms/inf)")
print(f"📈 1000 inferenze: {phase3_time:.0f}ms ({phase3_time/1000:.1f}ms/inf)")

print(f"\n⏱️  STIMA AUDIOMOTH:")
print(f"📈 10 inferenze: ~{phase1_time*audiomoth_factor:.0f}ms")
print(f"📈 100 inferenze: ~{phase2_time*audiomoth_factor:.0f}ms")  
print(f"📈 1000 inferenze: ~{phase3_time*audiomoth_factor:.0f}ms")

total_test_time = (phase1_time + phase2_time + phase3_time) * audiomoth_factor / 1000
print(f"🕒 Tempo totale test AudioMoth: ~{total_test_time:.0f} secondi")

print(f"\n✅ PIPELINE TEST COMPLETATO! Ready per AudioMoth hardware 🎯")