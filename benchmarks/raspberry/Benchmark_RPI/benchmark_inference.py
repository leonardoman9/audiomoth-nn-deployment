import numpy as np
import onnxruntime as ort
import time

# Configuration
BACKBONE_PATH = "models/backbone.onnx"
STREAMING_PATH = "models/streaming.onnx"
WARMUP_TRIALS = 30
BENCHMARK_TRIALS = 150

# Audio processing parameters
BACKBONE_WINDOW_MS = 200  # Each backbone input represents 200ms
TARGET_DURATION_MS = 3000  # 3 seconds
NUM_WINDOWS = TARGET_DURATION_MS // BACKBONE_WINDOW_MS  # 15 windows for 3 seconds

def create_models():
    """Initialize ONNX Runtime sessions."""
    backbone_session = ort.InferenceSession(BACKBONE_PATH)
    streaming_session = ort.InferenceSession(STREAMING_PATH)
    return backbone_session, streaming_session

def get_model_info(backbone_session, streaming_session):
    """Get input/output information from both models."""
    backbone_inputs = backbone_session.get_inputs()
    backbone_outputs = backbone_session.get_outputs()
    streaming_inputs = streaming_session.get_inputs()

    # Extract input/output names and shapes
    backbone_input_name = backbone_inputs[0].name
    backbone_output_name = backbone_outputs[0].name
    backbone_input_shape = [dim if dim is not None else 1 for dim in backbone_inputs[0].shape]

    # Find streaming model input names
    feature_input_name = None
    hidden_state_input_name = None
    hidden_state_shape = None

    for inp in streaming_inputs:
        if 'feature' in inp.name.lower() or 'frame' in inp.name.lower():
            feature_input_name = inp.name
        elif 'hidden' in inp.name.lower():
            hidden_state_input_name = inp.name
            hidden_state_shape = [dim if dim is not None else 1 for dim in inp.shape]

    return {
        'backbone_input_name': backbone_input_name,
        'backbone_output_name': backbone_output_name,
        'backbone_input_shape': backbone_input_shape,
        'feature_input_name': feature_input_name,
        'hidden_state_input_name': hidden_state_input_name,
        'hidden_state_shape': hidden_state_shape
    }

def process_single_trial(backbone_session, streaming_session, model_info):
    """Process one complete 3-second audio trial."""
    # Create dummy input for backbone (representing 3 seconds split into windows)
    backbone_input_shape = model_info['backbone_input_shape']

    total_inference_time = 0.0
    all_outputs = []

    # Initialize hidden state
    if model_info['hidden_state_input_name'] and model_info['hidden_state_shape']:
        current_hidden_state = np.zeros(model_info['hidden_state_shape'], dtype=np.float32)
    else:
        current_hidden_state = None

    # Process each 200ms window
    for window_idx in range(NUM_WINDOWS):
        # Create dummy spectrogram input for this 200ms window
        dummy_input = np.random.uniform(
            low=-1.0, high=1.0, size=backbone_input_shape
        ).astype(np.float32)

        # Backbone inference
        start_time = time.perf_counter()
        backbone_output = backbone_session.run(
            [model_info['backbone_output_name']],
            {model_info['backbone_input_name']: dummy_input}
        )[0]

        # Process each frame from backbone output
        num_frames = backbone_output.shape[1]  # Should be 18 frames
        window_outputs = []

        for frame_idx in range(num_frames):
            frame = backbone_output[:, frame_idx, :]  # Shape: (1, 32)

            # Prepare streaming model inputs
            streaming_inputs_dict = {model_info['feature_input_name']: frame}
            if current_hidden_state is not None:
                streaming_inputs_dict[model_info['hidden_state_input_name']] = current_hidden_state

            # Streaming inference
            streaming_outputs = streaming_session.run(None, streaming_inputs_dict)

            # Store output and update hidden state
            window_outputs.append(streaming_outputs[0])
            if len(streaming_outputs) > 1 and current_hidden_state is not None:
                current_hidden_state = streaming_outputs[1]

        end_time = time.perf_counter()
        total_inference_time += (end_time - start_time)
        all_outputs.extend(window_outputs)

    return total_inference_time, len(all_outputs)

def run_benchmark():
    """Run the complete benchmark."""
    print("Initializing models...")
    backbone_session, streaming_session = create_models()
    model_info = get_model_info(backbone_session, streaming_session)

    print(f"Benchmark configuration:")
    print(f"  Target duration: {TARGET_DURATION_MS}ms")
    print(f"  Backbone window size: {BACKBONE_WINDOW_MS}ms")
    print(f"  Number of windows: {NUM_WINDOWS}")
    print(f"  Warmup trials: {WARMUP_TRIALS}")
    print(f"  Benchmark trials: {BENCHMARK_TRIALS}")
    print()

    # Warmup phase
    print("Running warmup trials...")
    for i in range(WARMUP_TRIALS):
        process_single_trial(backbone_session, streaming_session, model_info)
        if (i + 1) % 5 == 0:
            print(f"  Warmup {i + 1}/{WARMUP_TRIALS} completed")

    # Benchmark phase
    print(f"\nRunning {BENCHMARK_TRIALS} benchmark trials...")
    inference_times = []

    for i in range(BENCHMARK_TRIALS):
        inference_time, num_outputs = process_single_trial(backbone_session, streaming_session, model_info)
        inference_times.append(inference_time)

        if (i + 1) % 20 == 0:
            print(f"  Trial {i + 1}/{BENCHMARK_TRIALS} completed")

    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    # Real-time factor calculation
    audio_duration_seconds = TARGET_DURATION_MS / 1000.0
    real_time_factor = audio_duration_seconds/avg_inference_time 

    # Results
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Audio duration: {audio_duration_seconds:.1f}s")
    print(f"Average inference time: {avg_inference_time*1000:.2f}ms ± {std_inference_time*1000:.2f}ms")
    print(f"Real-time factor: {real_time_factor:.3f}x")
    print()

    if real_time_factor > 1.0:
        print("✅ Model runs faster than real-time!")
    else:
        print("❌ Model runs slower than real-time")

    print(f"Min inference time: {min(inference_times)*1000:.2f}ms")
    print(f"Max inference time: {max(inference_times)*1000:.2f}ms")

if __name__ == "__main__":
    run_benchmark()
