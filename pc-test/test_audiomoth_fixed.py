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

# Info modelli dettagliate
bb_input = backbone.get_input_details()[0]
bb_output = backbone.get_output_details()[0]
st_input = streaming.get_input_details()[0]
st_output = streaming.get_output_details()[0]

print(f"📊 Backbone input: {bb_input['shape']} ({bb_input['dtype']})")
print(f"📊 Backbone output: {bb_output['shape']} ({bb_output['dtype']})")
print(f"📊 Streaming input: {st_input['shape']} ({st_input['dtype']})")
print(f"📊 Streaming output: {st_output['shape']} ({st_output['dtype']})")

# Test dimensioni
print("\n🔍 Test compatibilità dimensioni...")
dummy_audio = np.random.randint(-128, 127, bb_input['shape'], dtype=np.int8)

backbone.set_tensor(bb_input['index'], dummy_audio)
backbone.invoke()
bb_features = backbone.get_tensor(bb_output['index'])

print(f"💡 Backbone output effettivo: {bb_features.shape}")
print(f"💡 Streaming input atteso: {st_input['shape']}")

# Fix dimensioni se necessario
if bb_features.shape != tuple(st_input['shape']):
    print("⚙️  Reshape necessario...")
    # Se backbone output è [1, timesteps, features] e streaming vuole [1, features]
    if len(bb_features.shape) == 3 and len(st_input['shape']) == 2:
        # Prendiamo l'ultimo timestep
        bb_features_reshaped = bb_features[0, -1:, :].reshape(1, -1)
        print(f"   Reshape a: {bb_features_reshaped.shape}")
    else:
        bb_features_reshaped = bb_features.reshape(st_input['shape'])
        print(f"   Reshape a: {bb_features_reshaped.shape}")
    bb_features = bb_features_reshaped

# Test con dati dummy
print("\n🧪 Test inferenza completa...")

# Backbone inference
start = time.perf_counter()
backbone.set_tensor(bb_input['index'], dummy_audio)
backbone.invoke()
bb_features = backbone.get_tensor(bb_output['index'])
bb_time = time.perf_counter() - start

# Fix dimensioni per streaming
if bb_features.shape != tuple(st_input['shape']):
    if len(bb_features.shape) == 3 and len(st_input['shape']) == 2:
        bb_features = bb_features[0, -1:, :].reshape(1, -1)
    else:
        bb_features = bb_features.reshape(st_input['shape'])

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
print(f"📊 Output shape: {st_output_data.shape}")

# Performance test
print(f"\n🏃 Performance test (20 iterazioni)...")
times = []
bb_times = []
st_times = []

for i in range(20):
    dummy_audio = np.random.randint(-128, 127, bb_input['shape'], dtype=np.int8)
    
    # Backbone
    start = time.perf_counter()
    backbone.set_tensor(bb_input['index'], dummy_audio)
    backbone.invoke()
    bb_features = backbone.get_tensor(bb_output['index'])
    bb_time = time.perf_counter() - start
    
    # Fix dimensioni
    if bb_features.shape != tuple(st_input['shape']):
        if len(bb_features.shape) == 3 and len(st_input['shape']) == 2:
            bb_features = bb_features[0, -1:, :].reshape(1, -1)
        else:
            bb_features = bb_features.reshape(st_input['shape'])
    
    # Streaming
    start = time.perf_counter()
    streaming.set_tensor(st_input['index'], bb_features)
    streaming.invoke()
    st_output_data = streaming.get_tensor(st_output['index'])
    st_time = time.perf_counter() - start
    
    total_time = bb_time + st_time
    times.append(total_time * 1000)
    bb_times.append(bb_time * 1000)
    st_times.append(st_time * 1000)
    
    if (i+1) % 5 == 0:
        print(f"   {i+1}/20 completate...")

print(f"\n📈 RISULTATI PERFORMANCE:")
print(f"⚡ Backbone: {np.mean(bb_times):.2f}ms ± {np.std(bb_times):.2f}ms")
print(f"⚡ Streaming: {np.mean(st_times):.2f}ms ± {np.std(st_times):.2f}ms")
print(f"📈 Totale: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
print(f"🚀 Throughput PC: {1000/np.mean(times):.1f} inferenze/sec")
print(f"⚡ AudioMoth stimato: ~{np.mean(times)*4:.0f}ms (4x più lento)")
print(f"🎯 AudioMoth throughput: ~{1000/(np.mean(times)*4):.1f} inferenze/sec")
