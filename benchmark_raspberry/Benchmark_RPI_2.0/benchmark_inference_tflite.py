import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Configuration
BACKBONE_PATH = "models/backbone_int8_FINAL.tflite"
STREAMING_PATH = "models/streaming_int8_FINAL.tflite"
WARMUP_TRIALS = 30
BENCHMARK_TRIALS = 150

# Audio processing parameters
BACKBONE_WINDOW_MS = 200
TARGET_DURATION_MS = 3000
NUM_WINDOWS = TARGET_DURATION_MS // BACKBONE_WINDOW_MS  # 15 windows

def create_models():
    """Initialize TensorFlow Lite interpreters."""
    backbone_interpreter = tflite.Interpreter(model_path=BACKBONE_PATH)
    backbone_interpreter.allocate_tensors()
    
    streaming_interpreter = tflite.Interpreter(model_path=STREAMING_PATH)
    streaming_interpreter.allocate_tensors()
    
    return backbone_interpreter, streaming_interpreter

def get_model_info(backbone_interpreter, streaming_interpreter):
    """Get input/output information from both models."""
    # Backbone model info
    backbone_input_details = backbone_interpreter.get_input_details()
    backbone_output_details = backbone_interpreter.get_output_details()
    
    backbone_input_index = backbone_input_details[0]['index']
    backbone_output_index = backbone_output_details[0]['index']
    backbone_input_shape = backbone_input_details[0]['shape']
    backbone_input_dtype = backbone_input_details[0]['dtype']

    # Streaming model info
    streaming_input_details = streaming_interpreter.get_input_details()
    streaming_output_details = streaming_interpreter.get_output_details()
    
    feature_input_index = None
    hidden_state_input_index = None
    hidden_state_shape = None
    hidden_state_dtype = None

    for inp in streaming_input_details:
        name = inp['name'].lower()
        if 'feature' in name or 'frame' in name:
            feature_input_index = inp['index']
        elif 'hidden' in name:
            hidden_state_input_index = inp['index']
            hidden_state_shape = tuple(inp['shape'])
            hidden_state_dtype = inp['dtype']
    
    streaming_output_indices = [out['index'] for out in streaming_output_details]
    
    # Get output details to understand what each output represents
    output_details = []
    for out in streaming_output_details:
        output_details.append({
            'index': out['index'],
            'name': out['name'],
            'shape': tuple(out['shape']),
            'dtype': out['dtype']
        })

    return {
        'backbone_input_index': backbone_input_index,
        'backbone_output_index': backbone_output_index,
        'backbone_input_shape': backbone_input_shape,
        'backbone_input_dtype': backbone_input_dtype,
        'feature_input_index': feature_input_index,
        'hidden_state_input_index': hidden_state_input_index,
        'hidden_state_shape': hidden_state_shape,
        'hidden_state_dtype': hidden_state_dtype,
        'streaming_output_indices': streaming_output_indices,
        'streaming_output_details': output_details
    }

def process_single_trial(backbone_interpreter, streaming_interpreter, model_info):
    """Process one complete 3-second audio trial."""
    backbone_input_shape = tuple(model_info['backbone_input_shape'])
    backbone_input_dtype = model_info['backbone_input_dtype']
    
    total_inference_time = 0.0
    all_outputs = []

    # Initialize hidden state based on what the model expects as input
    current_hidden_state = None
    hidden_shape = model_info['hidden_state_shape']
    hidden_dtype = model_info['hidden_state_dtype']

    if hidden_shape is not None and hidden_dtype is not None:
        current_hidden_state = np.zeros(hidden_shape, dtype=hidden_dtype)
    else:
        current_hidden_state = None

    # Process each 200ms window
    for window_idx in range(NUM_WINDOWS):
        # Generate dummy input for backbone
        if backbone_input_dtype == np.int8:
            dummy_input = np.random.randint(
                low=-128, high=127, size=backbone_input_shape, dtype=np.int8
            )
        else:
            dummy_input = np.random.uniform(
                low=-1.0, high=1.0, size=backbone_input_shape
            ).astype(backbone_input_dtype)

        start_time = time.perf_counter()
        
        # Backbone inference
        backbone_interpreter.set_tensor(model_info['backbone_input_index'], dummy_input)
        backbone_interpreter.invoke()
        backbone_output = backbone_interpreter.get_tensor(model_info['backbone_output_index'])

        num_frames = backbone_output.shape[1]
        window_outputs = []

        for frame_idx in range(num_frames):
            frame = backbone_output[:, frame_idx, :]  # Shape: (1, 32)

            # Ensure frame dtype matches expected streaming input
            if frame.dtype != np.int8:
                frame = frame.astype(np.int8)

            streaming_interpreter.set_tensor(model_info['feature_input_index'], frame)

            if current_hidden_state is not None:
                # Always use the expected shape and dtype for input
                streaming_interpreter.set_tensor(model_info['hidden_state_input_index'], current_hidden_state)

            streaming_interpreter.invoke()

            streaming_outputs = [
                streaming_interpreter.get_tensor(output_idx)
                for output_idx in model_info['streaming_output_indices']
            ]

            # Save primary output (assuming output 0 is the main prediction)
            window_outputs.append(streaming_outputs[0])

            # Update hidden state from output 1
            if len(streaming_outputs) > 1:
                # The model outputs (1, 5) but expects (1, 32) as input
                # This suggests we need to transform the hidden state
                output_hidden_state = streaming_outputs[1]
                
                if output_hidden_state.shape == (1, 5) and model_info['hidden_state_shape'] == (1, 32):
                    # Transform the (1, 5) output to (1, 32) input for next iteration
                    # This could be padding, projection, or some other transformation
                    # For now, let's pad with zeros or use a simple transformation
                    transformed_hidden = np.zeros((1, 32), dtype=model_info['hidden_state_dtype'])
                    transformed_hidden[0, :5] = output_hidden_state[0]  # Copy first 5 elements
                    current_hidden_state = transformed_hidden
                else:
                    # If shapes match, use directly
                    current_hidden_state = output_hidden_state

        end_time = time.perf_counter()
        total_inference_time += (end_time - start_time)
        all_outputs.extend(window_outputs)

    return total_inference_time, len(all_outputs)

def run_benchmark():
    """Run the complete benchmark."""
    print("Initializing models...")
    backbone_interpreter, streaming_interpreter = create_models()
    model_info = get_model_info(backbone_interpreter, streaming_interpreter)

    # Print model details for debugging
    print(f"\nModel details:")
    print(f"Backbone input shape: {model_info['backbone_input_shape']}")
    print(f"Backbone input dtype: {model_info['backbone_input_dtype']}")
    print(f"Hidden state input shape: {model_info['hidden_state_shape']}")
    print(f"Hidden state input dtype: {model_info['hidden_state_dtype']}")
    print("Streaming outputs:")
    for i, out in enumerate(model_info['streaming_output_details']):
        print(f"  Output {i}: {out['name']}, shape: {out['shape']}, dtype: {out['dtype']}")

    print(f"\nBenchmark configuration:")
    print(f"  Target duration: {TARGET_DURATION_MS}ms")
    print(f"  Backbone window size: {BACKBONE_WINDOW_MS}ms")
    print(f"  Number of windows: {NUM_WINDOWS}")
    print(f"  Warmup trials: {WARMUP_TRIALS}")
    print(f"  Benchmark trials: {BENCHMARK_TRIALS}\n")

    print("Running warmup trials...")
    for i in range(WARMUP_TRIALS):
        process_single_trial(backbone_interpreter, streaming_interpreter, model_info)
        if (i + 1) % 5 == 0:
            print(f"  Warmup {i + 1}/{WARMUP_TRIALS} completed")

    print(f"\nRunning {BENCHMARK_TRIALS} benchmark trials...")
    inference_times = []

    for i in range(BENCHMARK_TRIALS):
        inference_time, num_outputs = process_single_trial(backbone_interpreter, streaming_interpreter, model_info)
        inference_times.append(inference_time)

        if (i + 1) % 20 == 0:
            print(f"  Trial {i + 1}/{BENCHMARK_TRIALS} completed")

    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    audio_duration_seconds = TARGET_DURATION_MS / 1000.0
    real_time_factor = audio_duration_seconds / avg_inference_time

    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Audio duration: {audio_duration_seconds:.1f}s")
    print(f"Average inference time: {avg_inference_time*1000:.2f}ms ± {std_inference_time*1000:.2f}ms")
    print(f"Real-time factor: {real_time_factor:.3f}x\n")

    if real_time_factor > 1.0:
        print("✅ Model runs faster than real-time!")
    else:
        print("❌ Model runs slower than real-time")

    print(f"Min inference time: {min(inference_times)*1000:.2f}ms")
    print(f"Max inference time: {max(inference_times)*1000:.2f}ms")

if __name__ == "__main__":
    run_benchmark()