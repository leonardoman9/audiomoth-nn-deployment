#!/usr/bin/env python3
"""
Validazione numerica: il nostro modello produce lo stesso output del TFLite?
"""

import tensorflow as tf
import numpy as np

def validate_backbone_model():
    """Testa se il nostro backbone produce lo stesso output del TFLite originale"""
    print("🔍 VALIDAZIONE BACKBONE MODEL")
    print("="*50)
    
    # Carica il modello TFLite originale
    interpreter = tf.lite.Interpreter(model_path="./convert-and-quantize/streamable-kws/backbone_int8_FINAL.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 TFLite Original:")
    print(f"   Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"   Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
    
    # Crea input di test (stesso che usiamo nel nostro modello)
    input_shape = input_details[0]['shape']
    test_input = np.zeros(input_shape, dtype=np.int8)
    
    # Riempi con pattern di test
    for i in range(input_shape[1]):  # 18 frames
        for j in range(input_shape[2]):  # 40 features
            test_input[0, i, j] = (i * 40 + j) % 256 - 128  # Pattern ripetibile
    
    print(f"🎯 Test input shape: {test_input.shape}")
    print(f"   Range: [{test_input.min()}, {test_input.max()}]")
    
    # Esegui inferenza TFLite originale
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"📊 TFLite output shape: {tflite_output.shape}")
    print(f"   Range: [{tflite_output.min()}, {tflite_output.max()}]")
    
    # Salva input e output per comparazione con il nostro modello
    np.save('test_input_backbone.npy', test_input)
    np.save('expected_output_backbone.npy', tflite_output)
    
    # Genera C array per il main.c
    generate_c_test_data(test_input, tflite_output, "backbone")
    
    return test_input, tflite_output

def validate_streaming_model():
    """Testa il modello streaming"""
    print("\n🔍 VALIDAZIONE STREAMING MODEL")
    print("="*50)
    
    interpreter = tf.lite.Interpreter(model_path="./convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 TFLite Original:")
    for i, inp in enumerate(input_details):
        print(f"   Input {i}: {inp['shape']} {inp['dtype']}")
    for i, out in enumerate(output_details):
        print(f"   Output {i}: {out['shape']} {out['dtype']}")
    
    # Input di test
    feature_input = np.random.randint(-50, 50, input_details[0]['shape'], dtype=np.int8)
    hidden_input = np.random.randint(-30, 30, input_details[1]['shape'], dtype=np.int8)
    
    # Inferenza TFLite
    interpreter.set_tensor(input_details[0]['index'], feature_input)
    interpreter.set_tensor(input_details[1]['index'], hidden_input)
    interpreter.invoke()
    
    hidden_output = interpreter.get_tensor(output_details[0]['index'])
    logits_output = interpreter.get_tensor(output_details[1]['index'])
    
    print(f"📊 Outputs:")
    print(f"   Hidden: {hidden_output.shape}, range [{hidden_output.min()}, {hidden_output.max()}]")
    print(f"   Logits: {logits_output.shape}, range [{logits_output.min()}, {logits_output.max()}]")
    
    # Salva per comparazione
    np.save('test_feature_input.npy', feature_input)
    np.save('test_hidden_input.npy', hidden_input)
    np.save('expected_hidden_output.npy', hidden_output)
    np.save('expected_logits_output.npy', logits_output)
    
    generate_c_test_data_streaming(feature_input, hidden_input, hidden_output, logits_output)
    
    return feature_input, hidden_input, hidden_output, logits_output

def generate_c_test_data(input_data, expected_output, model_name):
    """Genera array C per test nel firmware"""
    input_flat = input_data.flatten()
    output_flat = expected_output.flatten()
    
    c_code = f"""
// Test data per validazione {model_name} model
static const int8_t test_input_{model_name}[{len(input_flat)}] = {{
"""
    
    for i, val in enumerate(input_flat):
        if i % 16 == 0:
            c_code += "\n    "
        c_code += f"{val:4d},"
    
    c_code += f"""
}};

static const int8_t expected_output_{model_name}[{len(output_flat)}] = {{
"""
    
    for i, val in enumerate(output_flat):
        if i % 16 == 0:
            c_code += "\n    "
        c_code += f"{val:4d},"
    
    c_code += "\n};\n"
    
    with open(f'test_data_{model_name}.h', 'w') as f:
        f.write(c_code)
    
    print(f"💾 Generato test_data_{model_name}.h")

def generate_c_test_data_streaming(features, hidden_in, hidden_out, logits):
    """Genera test data per streaming model"""
    c_code = f"""
// Test data per validazione streaming model
static const int8_t test_features[{features.size}] = {{
"""
    for i, val in enumerate(features.flatten()):
        if i % 16 == 0:
            c_code += "\n    "
        c_code += f"{val:4d},"
    
    c_code += f"""
}};

static const int8_t test_hidden_input[{hidden_in.size}] = {{
"""
    for i, val in enumerate(hidden_in.flatten()):
        if i % 16 == 0:
            c_code += "\n    "
        c_code += f"{val:4d},"
    
    c_code += f"""
}};

static const int8_t expected_hidden_output[{hidden_out.size}] = {{
"""
    for i, val in enumerate(hidden_out.flatten()):
        if i % 16 == 0:
            c_code += "\n    "
        c_code += f"{val:4d},"
    
    c_code += f"""
}};

static const int8_t expected_logits[{logits.size}] = {{
"""
    for i, val in enumerate(logits.flatten()):
        if i % 16 == 0:
            c_code += "\n    "
        c_code += f"{val:4d},"
    
    c_code += "\n};\n"
    
    with open('test_data_streaming.h', 'w') as f:
        f.write(c_code)
    
    print(f"💾 Generato test_data_streaming.h")

if __name__ == "__main__":
    print("🎯 VALIDAZIONE MODELLI TFLITE vs NOSTRO CMSIS")
    print("="*60)
    
    # Testa entrambi i modelli
    backbone_input, backbone_expected = validate_backbone_model()
    streaming_inputs = validate_streaming_model()
    
    print("\n✅ VALIDATION DATA GENERATED!")
    print("🔥 Ora possiamo testare se il nostro modello produce gli stessi output!")