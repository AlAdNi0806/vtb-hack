#!/bin/bash

# Setup script for Ubuntu environment
echo "Setting up Real-time Speech-to-Text backend for Ubuntu..."

# Update package list
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    git

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install additional dependencies for NeMo
echo "Installing additional NeMo dependencies..."
pip install Cython
pip install nemo_toolkit[all]

echo "Setup complete!"
echo ""
echo "🎉 Installation finished!"
echo "📋 Available models:"
echo "   - Parakeet 1.1B (multilingual) - Primary model"
echo "   - Whisper Tiny (English) - Fallback model"
echo ""
echo "🚀 To start the server:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "💡 The first run will download the Parakeet model (~4GB)"
