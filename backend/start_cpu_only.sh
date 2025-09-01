#!/bin/bash

# Script to start the backend with CPU-only configuration
# This avoids CUDA memory issues

echo "🔧 Starting backend with CPU-only configuration..."

# Set environment variables to force CPU usage
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_DISABLE_CUDA=1
export TORCH_USE_CUDA_DSA=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export NEMO_CACHE_DIR="/tmp/nemo_cache"

# Clear any existing CUDA processes
echo "🧹 Clearing CUDA processes..."
pkill -f python || true
sleep 2

# Activate virtual environment
source venv/bin/activate

# Start the server
echo "🚀 Starting server with CPU-only mode..."
python3 main.py
