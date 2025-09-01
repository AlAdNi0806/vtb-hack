#!/usr/bin/env python3
"""
Safe startup script for the backend server with better error handling
and environment configuration to prevent CUDA and memory issues.
"""

import os
import sys
import logging
import signal
import multiprocessing

# Set environment variables before importing any ML libraries
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA completely
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
os.environ["NUMBA_DISABLE_CUDA"] = "1"  # Disable NUMBA CUDA
os.environ["TORCH_USE_CUDA_DSA"] = "0"  # Disable CUDA DSA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Limit CUDA memory
os.environ["NEMO_CACHE_DIR"] = "/tmp/nemo_cache"  # Set NeMo cache directory

# Configure multiprocessing to avoid semaphore leaks
multiprocessing.set_start_method('spawn', force=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        import websockets
        import numpy
        logger.info("✓ Core dependencies available")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False

def check_realtimestt():
    """Check if RealtimeSTT is working"""
    try:
        from RealtimeSTT import AudioToTextRecorder
        logger.info("✓ RealtimeSTT import successful")
        
        # Try to create a minimal recorder to test
        recorder = AudioToTextRecorder(
            model="tiny.en",
            device="cpu",
            compute_type="int8",
            use_microphone=False,
            level=logging.CRITICAL
        )
        logger.info("✓ RealtimeSTT initialization successful")
        return True
    except Exception as e:
        logger.error(f"✗ RealtimeSTT check failed: {e}")
        return False

def main():
    """Main startup function"""
    logger.info("🚀 Starting backend server with safe configuration...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    
    # Check RealtimeSTT
    if not check_realtimestt():
        logger.error("❌ RealtimeSTT check failed")
        logger.info("💡 Try installing with: pip install RealtimeSTT --upgrade")
        sys.exit(1)
    
    try:
        # Import and start the main application
        from main import app
        import uvicorn
        
        logger.info("✅ All checks passed, starting server...")
        
        # Start the server with safe configuration
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            workers=1,  # Single worker to avoid multiprocessing issues
            loop="asyncio"
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
