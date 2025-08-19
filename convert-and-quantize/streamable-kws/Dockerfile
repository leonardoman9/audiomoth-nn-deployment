# Use Python base image that supports ARM64
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for ONNX to TFLite conversion
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    tf_keras \
    onnx==1.15.0 \
    onnx_graphsurgeon \
    psutil \
    ai-edge-litert \
    sng4onnx \
    onnx2tf

# Create input and output directories
RUN mkdir -p /workspace/input /workspace/output

# Copy conversion script
COPY convert_models.sh /workspace/convert_models.sh
RUN chmod +x /workspace/convert_models.sh

# Set entrypoint
ENTRYPOINT ["/workspace/convert_models.sh"]