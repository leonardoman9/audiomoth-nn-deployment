#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import time

print("🚀 AudioMoth NN Performance Test")
print("=" * 40)

# Carica modelli
backbone = tf.lite.Interpreter("backbone_int8_FINAL.tflite")
streaming = tf.lite.Interpreter("streaming_int8_FINAL.tflite") 
backbone.allocate_tensors()
streaming.allocate_tensors()

# Info modelli
bb_input = backbone.get_input_details()[0]
bb_output = backbone.get_output_details()[0]
st_input = streaming.get_input_details()[0]
st_output = streaming.get_output_details()[0]

print(f"📊 Backbone: {bb_input['shape']} → {bb_output['shape']}")
print(f"📊 Streaming: {st_input['shape']} → {st_output['shape']}")

# Funzione per dequantizzare output int8
def dequantize_output(quantized_output, output_details):
    scale = output_details['quantization_parameters']['scales'][0]
    zero_point = output_details['quantization_parameters']['zero_points'][0]
    return (quantized_output.astype(np.float32) - zero_point) * scale

# Test performance
print(f"\n🏃 Performance Test (50 iterazioni)...")
times = []
bb_times = []
st_times = []

for i in range(50):
    # Dummy audio input 
    dummy_audio = np.random.randint(-128, 127, bb_input['shape'], dtype=np.int8)
    
    # Backbone inference
    start = time.perf_counter()
    backbone.set_tensor(bb_input['index'], dummy_audio)
    backbone.invoke()
    bb_features = backbone.get_tensor(bb_output['index'])
    bb_time = time.perf_counter() - start
    
    # Reshape per streaming (prendi ultimo timestep)
    bb_features_stream = bb_features[0, -1:, :].reshape(1, -1)
    
    # Streaming inference
    start = time.perf_counter()
    streaming.set_tensor(st_input['index'], bb_features_stream)
    streaming.invoke()
    st_output_data = streaming.get_tensor(st_output['index'])
    st_time = time.perf_counter() - start
    
    total_time = bb_time + st_time
    times.append(total_time * 1000)
    bb_times.append(bb_time * 1000)
    st_times.append(st_time * 1000)
    
    # Test output primo giro
    if i == 0:
        # Dequantizza per analisi
        st_float = dequantize_output(st_output_data, st_output)
        probs = tf.nn.softmax(st_float[0]).numpy()
        predicted = np.argmax(probs)
        confidence = probs[predicted]
        print(f"🎯 Test output: classe {predicted}, confidence {confidence:.3f}")
    
    if (i+1) % 10 == 0:
        print(f"   ✅ {i+1}/50 completate...")

print(f"\n📈 RISULTATI PC (Intel/Apple Silicon):")
print(f"⚡ Backbone: {np.mean(bb_times):.2f}ms ± {np.std(bb_times):.2f}ms")
print(f"⚡ Streaming: {np.mean(st_times):.2f}ms ± {np.std(st_times):.2f}ms")
print(f"📈 Totale: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
print(f"🚀 Throughput: {1000/np.mean(times):.1f} inferenze/sec")

# Stima AudioMoth (ARM Cortex-M4)
audiomoth_factor = 5  # Cortex-M4 è ~5x più lento di PC moderno
audiomoth_time = np.mean(times) * audiomoth_factor
audiomoth_throughput = 1000 / audiomoth_time

print(f"\n📱 STIMA AUDIOMOTH (ARM Cortex-M4):")
print(f"⚡ Tempo inferenza: ~{audiomoth_time:.0f}ms")
print(f"🚀 Throughput: ~{audiomoth_throughput:.1f} inferenze/sec")

# Analisi realtime capability
audio_chunk_duration = 21.3  # ms (1024 samples @ 48kHz)
realtime_possible = audiomoth_time < audio_chunk_duration

print(f"\n🎵 ANALISI REAL-TIME:")
print(f"📊 Chunk audio: {audio_chunk_duration:.1f}ms")
print(f"⚡ Inferenza: ~{audiomoth_time:.0f}ms")
if realtime_possible:
    print(f"✅ REAL-TIME POSSIBILE! 🎉")
    overhead = audio_chunk_duration - audiomoth_time
    print(f"💨 Margine: {overhead:.1f}ms")
else:
    print(f"❌ Real-time difficile")
    delay = audiomoth_time - audio_chunk_duration
    print(f"⏰ Ritardo: {delay:.1f}ms")

print(f"\n🔋 POWER ANALYSIS:")
print(f"💡 Inference/sec: ~{audiomoth_throughput:.1f}")
print(f"🔋 Duty cycle: {(audiomoth_time/1000)*audiomoth_throughput:.1%}")
print(f"😴 Sleep time: {100-(audiomoth_time/1000)*audiomoth_throughput*100:.1f}%")
