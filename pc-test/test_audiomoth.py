#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import time

print("🚀 Test AudioMoth NN Models")

# Carica modelli
print("📂 Caricamento modelli...")
backbone = tf.lite.Interpreter("backbone_int8_FINAL.tflite")
streaming = tf.lite.Interpreter("streaming_int8_FINAL.tflite") 

backbone.allocate_tensors()
streaming.allocate_tensors()

# Info modelli
bb_input = backbone.get_input_details()[0]
bb_output = backbone.get_output_details()[0]
st_input = streaming.get_input_details()[0]
st_output = streaming.get_output_details()[0]

print(f"📊 Backbone input: {bb_input['shape']}")
print(f"📊 Streaming input: {st_input['shape']}")

# Test con dati dummy
print("\n🧪 Test con dati dummy...")

# Generate dummy data
dummy_audio = np.random.randint(-128, 127, bb_input['shape'], dtype=np.int8)

# Backbone inference
start = time.perf_counter()
backbone.set_tensor(bb_input['index'], dummy_audio)
backbone.invoke()
bb_features = backbone.get_tensor(bb_output['index'])
bb_time = time.perf_counter() - start

# Streaming inference
start = time.perf_counter()
streaming.set_tensor(st_input['index'], bb_features)
streaming.invoke()
st_output_data = streaming.get_tensor(st_output['index'])
st_time = time.perf_counter() - start

# Results
probs = tf.nn.softmax(st_output_data[0]).numpy()
predicted = np.argmax(probs)
confidence = probs[predicted]

print(f"✅ Backbone: {bb_time*1000:.2f}ms")
print(f"✅ Streaming: {st_time*1000:.2f}ms") 
print(f"✅ Totale: {(bb_time+st_time)*1000:.2f}ms")
print(f"🎯 Predizione: classe {predicted}, confidence {confidence:.3f}")

# Performance test
print(f"\n🏃 Performance test (20 iterazioni)...")
times = []
for i in range(20):
    dummy_audio = np.random.randint(-128, 127, bb_input['shape'], dtype=np.int8)
    
    start = time.perf_counter()
    backbone.set_tensor(bb_input['index'], dummy_audio)
    backbone.invoke()
    bb_features = backbone.get_tensor(bb_output['index'])
    
    streaming.set_tensor(st_input['index'], bb_features)
    streaming.invoke()
    st_output_data = streaming.get_tensor(st_output['index'])
    total_time = time.perf_counter() - start
    
    times.append(total_time * 1000)
    
    if (i+1) % 5 == 0:
        print(f"   {i+1}/20 completate...")

print(f"📈 Media: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
print(f"🚀 Throughput: {1000/np.mean(times):.1f} inferenze/sec")
print(f"⚡ Su AudioMoth (ARM Cortex-M4): ~{np.mean(times)*5:.0f}ms stimati")
