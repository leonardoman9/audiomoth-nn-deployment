import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Configuration
BACKBONE_PATH = "models/backbone_int8_FINAL.tflite"
STREAMING_PATH = "models/streaming_int8_FINAL.tflite"
WARMUP_TRIALS = 30
BENCHMARK_TRIALS = 150

# Audio processing parameters
SAMPLE_RATE = 32000  # 32kHz
BACKBONE_WINDOW_MS = 200
TARGET_DURATION_MS = 3000
NUM_WINDOWS = TARGET_DURATION_MS // BACKBONE_WINDOW_MS  # 15 windows

# STFT parameters
N_FFT = 512
HOP_LENGTH = 320
N_MEL_BINS = 40

# Calculate samples for 200ms window
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * BACKBONE_WINDOW_MS / 1000)

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
    }

def generate_audio_window():
    """Generate realistic audio data for a 200ms window."""
    samples = np.random.uniform(low=-1.0, high=1.0, size=SAMPLES_PER_WINDOW)
    
    # Apply some basic audio-like properties
    t = np.linspace(0, BACKBONE_WINDOW_MS / 1000, SAMPLES_PER_WINDOW, endpoint=False)
    
    # Add some random frequency components
    for freq in [100, 500, 1000, 2000, 4000, 8000]:
        if np.random.random() > 0.3:
            amplitude = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)
            samples += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Apply window function
    window = np.hanning(SAMPLES_PER_WINDOW)
    samples *= window
    
    return samples

def real_stft_processing(audio_window):
    """
    Actual STFT implementation without external dependencies.
    Returns log-mel spectrogram features.
    """
    # Calculate number of frames
    num_frames = 1 + (len(audio_window) - N_FFT) // HOP_LENGTH
    
    # Create mel filterbank
    def create_mel_filterbank():
        # Mel frequency conversion formulas
        def hz_to_mel(freq):
            return 2595 * np.log10(1 + freq / 700.0)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595.0) - 1)
        
        # Frequency bins for FFT
        fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
        
        # Mel frequency range (human hearing range)
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(SAMPLE_RATE / 2)
        mel_points = np.linspace(low_mel, high_mel, N_MEL_BINS + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Create filterbank
        filterbank = np.zeros((N_MEL_BINS, N_FFT // 2 + 1))
        
        for i in range(1, N_MEL_BINS + 1):
            left = hz_points[i - 1]
            center = hz_points[i]
            right = hz_points[i + 1]
            
            # Create triangular filter
            for j, freq in enumerate(fft_freqs):
                if freq < left:
                    continue
                elif freq < center:
                    filterbank[i - 1, j] = (freq - left) / (center - left)
                elif freq < right:
                    filterbank[i - 1, j] = (right - freq) / (right - center)
                else:
                    break
        
        # Normalize filters
        filterbank = filterbank / np.maximum(np.sum(filterbank, axis=1, keepdims=True), 1e-8)
        return filterbank

    # Compute STFT
    stft_result = np.zeros((num_frames, N_FFT // 2 + 1), dtype=np.complex64)
    
    for frame_idx in range(num_frames):
        start = frame_idx * HOP_LENGTH
        end = start + N_FFT
        
        if end > len(audio_window):
            # Zero-pad the last frame
            frame = np.pad(audio_window[start:], (0, end - len(audio_window)))
        else:
            frame = audio_window[start:end]
        
        # Apply window function
        window = np.hanning(N_FFT)
        frame = frame * window
        
        # FFT
        fft_result = np.fft.rfft(frame, n=N_FFT)
        stft_result[frame_idx, :] = fft_result

    # Magnitude spectrum
    magnitude_spectrum = np.abs(stft_result)
    
    # Apply mel filterbank
    mel_filterbank = create_mel_filterbank()
    mel_spectrogram = np.dot(magnitude_spectrum, mel_filterbank.T)
    
    # Log scaling (avoid log(0))
    log_mel = np.log(mel_spectrogram + 1e-8)
    
    # Reshape to match model input (1, num_frames, N_MEL_BINS)
    return log_mel[np.newaxis, :, :]

def process_single_trial(backbone_interpreter, streaming_interpreter, model_info):
    """Process one complete 3-second audio trial."""
    backbone_input_shape = tuple(model_info['backbone_input_shape'])
    backbone_input_dtype = model_info['backbone_input_dtype']
    
    total_inference_time = 0.0

    # Initialize hidden state
    current_hidden_state = None
    hidden_shape = model_info['hidden_state_shape']
    hidden_dtype = model_info['hidden_state_dtype']

    if hidden_shape is not None and hidden_dtype is not None:
        current_hidden_state = np.zeros(hidden_shape, dtype=hidden_dtype)

    # Process each 200ms window
    for window_idx in range(NUM_WINDOWS):
        # Generate realistic audio data
        audio_window = generate_audio_window()
        
        # Compute actual STFT features
        stft_features = real_stft_processing(audio_window)
        
        # Ensure the features match the expected input shape
        expected_frames = backbone_input_shape[1]
        expected_bins = backbone_input_shape[2]
        current_frames = stft_features.shape[1]
        
        if current_frames < expected_frames:
            # Pad with zeros
            pad_width = ((0, 0), (0, expected_frames - current_frames), (0, 0))
            stft_features = np.pad(stft_features, pad_width, mode='constant')
        elif current_frames > expected_frames:
            # Truncate
            stft_features = stft_features[:, :expected_frames, :]
        
        # Convert to the expected data type (int8 quantization)
        if backbone_input_dtype == np.int8:
            # Normalize and quantize
            stft_features = (stft_features - np.mean(stft_features)) / np.std(stft_features)
            stft_features = np.clip(stft_features * 127, -128, 127).astype(np.int8)
        else:
            stft_features = stft_features.astype(backbone_input_dtype)

        start_time = time.perf_counter()
        
        # Backbone inference
        backbone_interpreter.set_tensor(model_info['backbone_input_index'], stft_features)
        backbone_interpreter.invoke()
        backbone_output = backbone_interpreter.get_tensor(model_info['backbone_output_index'])

        num_frames = backbone_output.shape[1]

        for frame_idx in range(num_frames):
            frame = backbone_output[:, frame_idx, :]  # Shape: (1, 32)

            # Ensure frame dtype matches expected streaming input
            if frame.dtype != np.int8:
                frame = frame.astype(np.int8)

            streaming_interpreter.set_tensor(model_info['feature_input_index'], frame)

            if current_hidden_state is not None:
                streaming_interpreter.set_tensor(model_info['hidden_state_input_index'], current_hidden_state)

            streaming_interpreter.invoke()

            streaming_outputs = [
                streaming_interpreter.get_tensor(output_idx)
                for output_idx in model_info['streaming_output_indices']
            ]

            # Update hidden state from output 1
            if len(streaming_outputs) > 1:
                output_hidden_state = streaming_outputs[1]
                
                if output_hidden_state.shape == (1, 5) and model_info['hidden_state_shape'] == (1, 32):
                    # Transform the (1, 5) output to (1, 32) input
                    transformed_hidden = np.zeros((1, 32), dtype=model_info['hidden_state_dtype'])
                    transformed_hidden[0, :5] = output_hidden_state[0]
                    current_hidden_state = transformed_hidden
                else:
                    current_hidden_state = output_hidden_state

        end_time = time.perf_counter()
        total_inference_time += (end_time - start_time)

    return total_inference_time, num_frames * NUM_WINDOWS

def run_benchmark():
    """Run the complete benchmark."""
    print("Initializing models...")
    backbone_interpreter, streaming_interpreter = create_models()
    model_info = get_model_info(backbone_interpreter, streaming_interpreter)

    print(f"\nModel and processing details:")
    print(f"Sample rate: {SAMPLE_RATE}Hz")
    print(f"STFT parameters: n_fft={N_FFT}, hop_length={HOP_LENGTH}")
    print(f"Mel bins: {N_MEL_BINS}")
    print(f"Samples per {BACKBONE_WINDOW_MS}ms window: {SAMPLES_PER_WINDOW}")
    print(f"Backbone input shape: {model_info['backbone_input_shape']}")
    print(f"Backbone input dtype: {model_info['backbone_input_dtype']}")
    print(f"Hidden state input shape: {model_info['hidden_state_shape']}")
    print(f"Hidden state input dtype: {model_info['hidden_state_dtype']}")

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