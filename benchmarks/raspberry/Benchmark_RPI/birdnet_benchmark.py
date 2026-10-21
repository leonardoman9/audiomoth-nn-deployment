import numpy as np
import time
import tflite_runtime.interpreter as tflite

# Configuration
MODEL_PATH = "audio-model-int8.tflite"
WARMUP_TRIALS = 30
BENCHMARK_TRIALS = 150

# BirdNET audio processing parameters
SAMPLE_RATE = 48000
CHUNK_DURATION = 3.0  # 3 seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

def load_model(model_path):
    """Load the TensorFlow Lite model."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model loaded: {model_path}")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    return interpreter, input_details, output_details

def generate_random_audio(input_shape):
    """Generate random audio data."""
    return np.random.uniform(-1.0, 1.0, input_shape).astype(np.float32)

def run_single_inference(interpreter, input_details, output_details, audio_data):
    """Run one inference and return the time taken."""
    # Set input
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    
    # Run inference
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    
    # Get output (optional, for completeness)
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return end_time - start_time

def run_benchmark():
    """Run the complete benchmark."""
    print("BirdNET Benchmark - Simple Version")
    print("=" * 50)
    
    # Load model
    interpreter, input_details, output_details = load_model(MODEL_PATH)
    input_shape = input_details[0]['shape']
    
    print(f"\nConfiguration:")
    print(f"  Audio duration: {CHUNK_DURATION}s")
    print(f"  Sample rate: {SAMPLE_RATE}Hz") 
    print(f"  Input shape: {input_shape}")
    print(f"  Warmup trials: {WARMUP_TRIALS}")
    print(f"  Benchmark trials: {BENCHMARK_TRIALS}")
    
    # Warmup phase
    print(f"\nRunning {WARMUP_TRIALS} warmup trials...")
    for i in range(WARMUP_TRIALS):
        dummy_audio = generate_random_audio(input_shape)
        run_single_inference(interpreter, input_details, output_details, dummy_audio)
        
        if (i + 1) % 10 == 0:
            print(f"  Warmup {i + 1}/{WARMUP_TRIALS}")
    
    # Benchmark phase
    print(f"\nRunning {BENCHMARK_TRIALS} benchmark trials...")
    inference_times = []
    
    for i in range(BENCHMARK_TRIALS):
        dummy_audio = generate_random_audio(input_shape)
        inference_time = run_single_inference(interpreter, input_details, output_details, dummy_audio)
        inference_times.append(inference_time)
        
        if (i + 1) % 25 == 0:
            print(f"  Trial {i + 1}/{BENCHMARK_TRIALS}")
    
    # Calculate results
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    # Real-time factor
    real_time_factor = CHUNK_DURATION / avg_time
    
    # Print results
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Audio duration: {CHUNK_DURATION:.1f}s")
    print(f"Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"Min inference time: {min_time*1000:.2f}ms")
    print(f"Max inference time: {max_time*1000:.2f}ms")
    print(f"Real-time factor: {real_time_factor:.3f}x")
    print()
    
    if real_time_factor > 1.0:
        print("✅ Faster than real-time!")
    else:
        print("❌ Slower than real-time")
    
    print(f"Can process {1/avg_time:.1f} inferences per second")

if __name__ == "__main__":
    run_benchmark()