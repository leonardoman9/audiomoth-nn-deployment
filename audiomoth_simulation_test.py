#!/usr/bin/env python3
"""
AudioMoth NN Simulation Test
Simula l'inferenza AudioMoth con dati dummy per testare performance
"""

import numpy as np
import tensorflow as tf
import time
import argparse
from pathlib import Path

# Configurazione che simula AudioMoth
AUDIOMOTH_CONFIG = {
    'sample_rate': 48000,
    'audio_chunk_size': 1024,  # Simula buffer AudioMoth
    'backbone_input_shape': None,  # Da determinare dal modello
    'streaming_input_shape': None,  # Da determinare dal modello
    'num_classes': 10,
    'confidence_threshold': 0.7,
}

class AudioMothSimulator:
    def __init__(self, backbone_model_path, streaming_model_path):
        print("🚀 Inizializzazione AudioMoth Simulator...")
        
        # Carica modelli TFLite
        self.backbone_interpreter = tf.lite.Interpreter(model_path=backbone_model_path)
        self.streaming_interpreter = tf.lite.Interpreter(model_path=streaming_model_path)
        
        # Alloca tensori
        self.backbone_interpreter.allocate_tensors()
        self.streaming_interpreter.allocate_tensors()
        
        # Ottieni informazioni input/output
        self._setup_model_info()
        
        # Stato streaming GRU (simula memoria AudioMoth)
        self.gru_state = None
        
        print(f"✅ Backbone input: {self.backbone_input_shape}")
        print(f"✅ Streaming input: {self.streaming_input_shape}")
        print(f"✅ Modelli caricati e pronti!")
    
    def _setup_model_info(self):
        """Setup informazioni modelli"""
        # Backbone
        backbone_input = self.backbone_interpreter.get_input_details()[0]
        backbone_output = self.backbone_interpreter.get_output_details()[0]
        self.backbone_input_shape = backbone_input['shape']
        self.backbone_output_shape = backbone_output['shape']
        self.backbone_input_index = backbone_input['index']
        self.backbone_output_index = backbone_output['index']
        
        # Streaming
        streaming_input = self.streaming_interpreter.get_input_details()[0]
        streaming_output = self.streaming_interpreter.get_output_details()[0]
        self.streaming_input_shape = streaming_input['shape']
        self.streaming_output_shape = streaming_output['shape']
        self.streaming_input_index = streaming_input['index']
        self.streaming_output_index = streaming_output['index']
        
        # Inizializza stato GRU se necessario
        streaming_inputs = self.streaming_interpreter.get_input_details()
        if len(streaming_inputs) > 1:
            # Probabilmente ha stato GRU
            print("🔄 Modello streaming ha stato GRU")
    
    def generate_dummy_audio_chunk(self):
        """Genera chunk audio dummy (simula input AudioMoth)"""
        # Simula spectrogram o features audio
        if len(self.backbone_input_shape) == 4:  # [batch, height, width, channels]
            dummy_data = np.random.randint(-128, 127, 
                                         self.backbone_input_shape, 
                                         dtype=np.int8)
        else:
            dummy_data = np.random.randint(-128, 127, 
                                         self.backbone_input_shape, 
                                         dtype=np.int8)
        return dummy_data
    
    def reset_streaming_state(self):
        """Reset stato GRU (simula NN_ResetStreamState)"""
        self.gru_state = None
        print("🔄 Reset stato streaming GRU")
    
    def run_backbone_inference(self, audio_features):
        """Esegue inferenza backbone (simula parte 1 di NN_ProcessAudio)"""
        start_time = time.perf_counter()
        
        # Set input
        self.backbone_interpreter.set_tensor(self.backbone_input_index, audio_features)
        
        # Run inference
        self.backbone_interpreter.invoke()
        
        # Get output
        backbone_output = self.backbone_interpreter.get_tensor(self.backbone_output_index)
        
        inference_time = time.perf_counter() - start_time
        return backbone_output, inference_time
    
    def run_streaming_inference(self, backbone_features):
        """Esegue inferenza streaming (simula parte 2 di NN_ProcessAudio)"""
        start_time = time.perf_counter()
        
        # Set input features
        self.streaming_interpreter.set_tensor(self.streaming_input_index, backbone_features)
        
        # Set GRU state se presente
        # TODO: Gestire stato GRU se il modello lo richiede
        
        # Run inference
        self.streaming_interpreter.invoke()
        
        # Get output
        streaming_output = self.streaming_interpreter.get_tensor(self.streaming_output_index)
        
        inference_time = time.perf_counter() - start_time
        return streaming_output, inference_time
    
    def process_audio_chunk(self, verbose=False):
        """
        Simula NN_ProcessAudio completo:
        1. Generate dummy audio
        2. Backbone inference
        3. Streaming inference
        4. Decision logic
        """
        # Step 1: Generate dummy input (simula audio chunk)
        audio_features = self.generate_dummy_audio_chunk()
        
        if verbose:
            print(f"📊 Input shape: {audio_features.shape}")
        
        # Step 2: Backbone inference
        backbone_output, backbone_time = self.run_backbone_inference(audio_features)
        
        # Step 3: Streaming inference
        streaming_output, streaming_time = self.run_streaming_inference(backbone_output)
        
        # Step 4: Process output (simula finalize_decision)
        probabilities = tf.nn.softmax(streaming_output[0]).numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        total_time = backbone_time + streaming_time
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'backbone_time_ms': backbone_time * 1000,
            'streaming_time_ms': streaming_time * 1000,
            'total_time_ms': total_time * 1000,
            'detection': confidence > AUDIOMOTH_CONFIG['confidence_threshold']
        }
        
        if verbose:
            print(f"🎯 Classe: {predicted_class}, Confidence: {confidence:.3f}")
            print(f"⏱️  Backbone: {backbone_time*1000:.2f}ms, Streaming: {streaming_time*1000:.2f}ms")
        
        return result

def run_performance_test(simulator, num_iterations=100):
    """Esegue test performance completo"""
    print(f"\n🧪 TEST PERFORMANCE ({num_iterations} iterazioni)")
    print("=" * 50)
    
    results = []
    detections = 0
    
    # Reset streaming state
    simulator.reset_streaming_state()
    
    for i in range(num_iterations):
        result = simulator.process_audio_chunk(verbose=(i < 5))  # Verbose per prime 5
        results.append(result)
        
        if result['detection']:
            detections += 1
        
        if (i + 1) % 10 == 0:
            print(f"✅ Processate {i+1}/{num_iterations} iterazioni...")
    
    # Analisi risultati
    backbone_times = [r['backbone_time_ms'] for r in results]
    streaming_times = [r['streaming_time_ms'] for r in results]
    total_times = [r['total_time_ms'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    print(f"\n📊 RISULTATI PERFORMANCE:")
    print(f"🎯 Detections: {detections}/{num_iterations} ({detections/num_iterations*100:.1f}%)")
    print(f"⏱️  Backbone: {np.mean(backbone_times):.2f}ms ± {np.std(backbone_times):.2f}ms")
    print(f"⏱️  Streaming: {np.mean(streaming_times):.2f}ms ± {np.std(streaming_times):.2f}ms")
    print(f"⏱️  Totale: {np.mean(total_times):.2f}ms ± {np.std(total_times):.2f}ms")
    print(f"📈 Confidence media: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
    print(f"🚀 Throughput: {1000/np.mean(total_times):.1f} inferenze/sec")

def main():
    parser = argparse.ArgumentParser(description="AudioMoth NN Simulation Test")
    parser.add_argument("--backbone", default="backbone_int8_FINAL.tflite", 
                       help="Path to backbone TFLite model")
    parser.add_argument("--streaming", default="streaming_int8_FINAL.tflite",
                       help="Path to streaming TFLite model") 
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of test iterations")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Verifica esistenza modelli
    if not Path(args.backbone).exists():
        print(f"❌ Modello backbone non trovato: {args.backbone}")
        return 1
    
    if not Path(args.streaming).exists():
        print(f"❌ Modello streaming non trovato: {args.streaming}")
        return 1
    
    try:
        # Inizializza simulator
        simulator = AudioMothSimulator(args.backbone, args.streaming)
        
        # Test singolo
        print("\n🧪 TEST SINGOLO:")
        result = simulator.process_audio_chunk(verbose=True)
        print(f"✅ Test singolo completato!")
        
        # Test performance
        run_performance_test(simulator, args.iterations)
        
        print(f"\n🎉 Test completato con successo!")
        return 0
        
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())