#!/bin/bash

echo "=== ONNX to TFLite Conversion Script ==="
echo "Input directory: /workspace/input"
echo "Output directory: /workspace/output"

# Check if input files exist
BACKBONE_ONNX="/workspace/input/cnn_backbone_simplified.onnx"
STREAMING_ONNX="/workspace/input/streaming_processor_simplified.onnx"

if [ ! -f "$BACKBONE_ONNX" ]; then
    echo "ERROR: Backbone ONNX file not found at $BACKBONE_ONNX"
    exit 1
fi

if [ ! -f "$STREAMING_ONNX" ]; then
    echo "ERROR: Streaming ONNX file not found at $STREAMING_ONNX"
    exit 1
fi

echo "✓ Found both ONNX input files"

# Convert Backbone CNN Model
echo ""
echo "--- Converting Backbone CNN Model ---"

# TENTATIVO 1: Con dati di calibrazione (sintassi JSON)
echo "🔄 Tentativo 1: Quantizzazione con calibrazione (JSON)..."
onnx2tf \
    -i "$BACKBONE_ONNX" \
    -o "/workspace/output/backbone_model_attempt1" \
    --output_quant_dtype int8 \
    --quant_type per-channel \
    -cind '{"spectrogram": ["/workspace/calibration_data/backbone_cal_data.npy", 0.0, 1.0]}'

if [ $? -eq 0 ]; then
    echo "✅ Tentativo 1 RIUSCITO!"
    find /workspace/output/backbone_model_attempt1 -name "*.tflite" -exec cp {} /workspace/output/backbone_int8.tflite \;
    echo "✓ Backbone TFLite model saved as: /workspace/output/backbone_int8.tflite"

else
    echo "❌ Tentativo 1 fallito, provo approccio alternativo..."
    
    # TENTATIVO 2: Senza calibrazione, solo quantizzazione
    echo "🔄 Tentativo 2: Quantizzazione senza calibrazione..."
    onnx2tf \
        -i "$BACKBONE_ONNX" \
        -o "/workspace/output/backbone_model_attempt2" \
        --output_quant_dtype int8 \
        --quant_type per-channel
    
    if [ $? -eq 0 ]; then
        echo "✅ Tentativo 2 RIUSCITO!"
        find /workspace/output/backbone_model_attempt2 -name "*.tflite" -exec cp {} /workspace/output/backbone_int8.tflite \;
        echo "✓ Backbone TFLite model saved as: /workspace/output/backbone_int8.tflite"
            # Tieni SavedModel per post-processing Python, rimuovi solo i .tflite extra
    find /workspace/output/backbone_model_attempt2 -name "*.tflite" ! -name "*_float32.tflite" ! -name "*_float16.tflite" -delete 2>/dev/null || true
    else
        echo "❌ Anche il Tentativo 2 è fallito"
        
        # TENTATIVO 3: Solo conversione float32, quantizzazione dopo
        echo "🔄 Tentativo 3: Conversione float32 (fallback)..."
        onnx2tf \
            -i "$BACKBONE_ONNX" \
            -o "/workspace/output/backbone_model_float32"
        
        if [ $? -eq 0 ]; then
            echo "⚠️  Tentativo 3: Conversione float32 riuscita (da quantizzare manualmente)"
            find /workspace/output/backbone_model_float32 -name "*.tflite" -exec cp {} /workspace/output/backbone_float32.tflite \;

        else
            echo "💥 TUTTI i tentativi sono falliti per il backbone model"
        fi
    fi
fi

# Convert Streaming GRU Model
echo ""
echo "--- Converting Streaming GRU Model ---"

# TENTATIVO 1: Con dati di calibrazione (sintassi JSON per multi-input)
echo "🔄 Tentativo 1: Quantizzazione streaming con calibrazione (JSON)..."
onnx2tf \
    -i "$STREAMING_ONNX" \
    -o "/workspace/output/streaming_model_attempt1" \
    --output_quant_dtype int8 \
    --quant_type per-channel \
    -cind '{"feature_frame": ["/workspace/calibration_data/streaming_feature_cal_data.npy", 0.0, 1.0], "hidden_state": ["/workspace/calibration_data/streaming_hidden_cal_data.npy", 0.0, 1.0]}'

if [ $? -eq 0 ]; then
    echo "✅ Tentativo 1 RIUSCITO!"
    find /workspace/output/streaming_model_attempt1 -name "*.tflite" -exec cp {} /workspace/output/streaming_int8.tflite \;
    echo "✓ Streaming TFLite model saved as: /workspace/output/streaming_int8.tflite"

else
    echo "❌ Tentativo 1 fallito, provo approccio alternativo..."
    
    # TENTATIVO 2: Senza calibrazione, solo quantizzazione
    echo "🔄 Tentativo 2: Quantizzazione senza calibrazione..."
    onnx2tf \
        -i "$STREAMING_ONNX" \
        -o "/workspace/output/streaming_model_attempt2" \
        --output_quant_dtype int8 \
        --quant_type per-channel
    
    if [ $? -eq 0 ]; then
        echo "✅ Tentativo 2 RIUSCITO!"
        find /workspace/output/streaming_model_attempt2 -name "*.tflite" -exec cp {} /workspace/output/streaming_int8.tflite \;
        echo "✓ Streaming TFLite model saved as: /workspace/output/streaming_int8.tflite"
        # Tieni SavedModel per post-processing Python, rimuovi solo i .tflite extra
        find /workspace/output/streaming_model_attempt2 -name "*.tflite" ! -name "*_float32.tflite" ! -name "*_float16.tflite" -delete 2>/dev/null || true
    else
        echo "❌ Anche il Tentativo 2 è fallito"
        
        # TENTATIVO 3: Solo conversione float32
        echo "🔄 Tentativo 3: Conversione float32 (fallback)..."
        onnx2tf \
            -i "$STREAMING_ONNX" \
            -o "/workspace/output/streaming_model_float32"
        
        if [ $? -eq 0 ]; then
            echo "⚠️  Tentativo 3: Conversione float32 riuscita (da quantizzare manualmente)"
            find /workspace/output/streaming_model_float32 -name "*.tflite" -exec cp {} /workspace/output/streaming_float32.tflite \;

        else
            echo "💥 TUTTI i tentativi sono falliti per il streaming model"
        fi
    fi
fi

echo ""
echo "=== Conversion Summary ==="
echo "Output files:"
ls -lh /workspace/output/*.tflite 2>/dev/null || echo "No .tflite files generated"

echo ""
echo "=== Conversion Complete ==="