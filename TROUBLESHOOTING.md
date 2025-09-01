# Troubleshooting Guide

## Common Issues and Solutions

### 1. CUDA Out of Memory Error

**Symptoms:**
- "CUDA out of memory" error when loading Parakeet model
- GPU memory exhausted
- Falls back to RealtimeSTT model

**Solutions:**

#### Option A: Use CPU-Only Startup Script (Recommended)
```bash
cd backend
chmod +x start_cpu_only.sh
./start_cpu_only.sh
```

#### Option B: Use the Safe Startup Script
```bash
cd backend
python3 start_backend_safe.py
```

#### Option C: Set Environment Variables Manually
```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_DISABLE_CUDA=1
export TORCH_USE_CUDA_DSA=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
python3 main.py
```

#### Option D: Install CPU-only PyTorch
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Backend Crashes with "Aborted (core dumped)"

**Symptoms:**
- Server starts but crashes when processing audio
- Error messages about CUDA libraries
- Memory leaks and semaphore warnings

**Solutions:**
- Use the CPU-only startup script above
- Ensure all CUDA processes are killed before starting

### 2. Parakeet Model Authentication Error

**Error:** `401 Client Error` or `Repository Not Found`

**Solution:** The parakeet model requires authentication or isn't publicly available. The code now falls back to the reliable Whisper tiny model automatically.

### 3. RealtimeSTT Installation Issues

**Error:** Package installation fails

**Solutions:**
```bash
# Install system dependencies first
sudo apt update
sudo apt install -y python3-dev portaudio19-dev libasound2-dev libsndfile1-dev

# Install with specific versions
pip install RealtimeSTT==0.3.104
```

### 4. WebSocket Connection Issues

**Error:** Frontend can't connect to backend

**Solutions:**
1. Check if backend is running: `curl http://localhost:8000/health`
2. Verify firewall settings
3. Check the IP address in frontend configuration
4. Ensure CORS is properly configured

### 5. Microphone Permission Issues

**Error:** Browser doesn't request microphone permission

**Solutions:**
1. Use HTTPS or localhost (required for microphone access)
2. Check browser settings for microphone permissions
3. Try a different browser
4. Ensure the site isn't blocked

### 6. No Audio Transcription

**Possible Causes:**
- Model not loaded properly
- Audio format issues
- Network connectivity problems

**Solutions:**
1. Check backend logs for model loading errors
2. Test with the backend test script: `python3 test_server.py`
3. Verify audio is being captured in browser console
4. Try speaking louder and clearer

### 7. High CPU Usage

**Solutions:**
1. Use the tiny model instead of base: Edit `audio_processor.py` and change `model="base"` to `model="tiny"`
2. Increase chunk duration to reduce processing frequency
3. Limit the number of concurrent connections

### 8. Memory Leaks

**Solutions:**
1. Use the safe startup script which includes proper cleanup
2. Restart the backend periodically if running for extended periods
3. Monitor memory usage with `htop` or `top`

## Performance Optimization

### For Low-End Hardware:
```python
# In audio_processor.py, use these settings:
model="tiny.en"  # Smallest English-only model
compute_type="int8"  # Lower precision
device="cpu"  # Force CPU usage
```

### For Better Performance:
```python
# In audio_processor.py, use these settings:
model="base"  # Better accuracy
compute_type="float16"  # Higher precision
device="auto"  # Let it choose best device
```

## Testing Commands

### Test Backend Health:
```bash
curl http://localhost:8000/health
```

### Test WebSocket Connection:
```bash
cd backend
python3 test_server.py
```

### Test Frontend Build:
```bash
cd frontend
npm run build
```

### Check System Resources:
```bash
# Monitor CPU and memory
htop

# Check audio devices
arecord -l

# Check network connections
netstat -tulpn | grep :8000
```

## Log Analysis

### Backend Logs:
- Look for "Models initialized successfully" - confirms model loading
- Check for WebSocket connection messages
- Monitor for memory warnings

### Frontend Logs:
- Open browser developer tools (F12)
- Check Console tab for JavaScript errors
- Monitor Network tab for WebSocket connections

## Environment Setup Verification

### Check Python Environment:
```bash
python3 --version  # Should be 3.8+
pip list | grep -E "(fastapi|uvicorn|RealtimeSTT)"
```

### Check Node.js Environment:
```bash
node --version  # Should be 18+
npm --version
```

### Check System Audio:
```bash
# Test microphone
arecord -d 5 test.wav
aplay test.wav

# Check audio devices
cat /proc/asound/cards
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs** - Both backend terminal and browser console
2. **Try the safe startup script** - `python3 start_backend_safe.py`
3. **Test with minimal configuration** - Use tiny model and CPU-only
4. **Verify system requirements** - Ubuntu 20.04+, Python 3.8+, Node.js 18+
5. **Check network connectivity** - Ensure ports 3000 and 8000 are available

## Quick Reset

If everything is broken, try this complete reset:

```bash
# Stop all processes
pkill -f "python.*main.py"
pkill -f "npm.*dev"

# Clean Python environment
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Clean Node.js environment
cd ../frontend
rm -rf node_modules package-lock.json
npm install

# Start fresh
cd ..
./start_app.sh
```
