#!/bin/bash

echo "=== Docker ONNX to TFLite Conversion ==="

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/logs/export_nfft_128_all_classes/2025-08-19_15-33-20_simplified"
OUTPUT_DIR="$SCRIPT_DIR/tflite_models"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if input files exist
if [ ! -f "$INPUT_DIR/cnn_backbone_simplified.onnx" ] || [ ! -f "$INPUT_DIR/streaming_processor_simplified.onnx" ]; then
    echo "ERROR: ONNX input files not found in $INPUT_DIR"
    echo "Expected files:"
    echo "  - cnn_backbone_simplified.onnx"
    echo "  - streaming_processor_simplified.onnx"
    exit 1
fi

echo "✓ Found input ONNX files"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Build Docker image
echo ""
echo "--- Building Docker Image ---"
docker build -t onnx2tflite-converter "$SCRIPT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

echo "✓ Docker image built successfully"

# Run conversion in Docker container
echo ""
echo "--- Running Conversion in Docker Container ---"
docker run --rm \
    -v "$INPUT_DIR:/workspace/input:ro" \
    -v "$SCRIPT_DIR/calibration_data:/workspace/calibration_data:ro" \
    -v "$OUTPUT_DIR:/workspace/output" \
    onnx2tflite-converter

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 CONVERSION COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Generated TFLite models:"
    ls -lh "$OUTPUT_DIR"/*.tflite 2>/dev/null || echo "No .tflite files found"
    echo ""
    echo "Models are ready for integration into AudioMoth firmware!"
else
    echo ""
    echo "❌ CONVERSION FAILED"
    echo "Check the Docker container logs above for details."
    exit 1
fi