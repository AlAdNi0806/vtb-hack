#!/usr/bin/env python3
"""
Piper TTS WebSocket Server (FIXED MODEL PATH)
================================================
Fixed version that properly locates the Russian voice model.

Key Fixes:
- Uses absolute path resolution for model files
- Checks if model exists before starting
- Provides clear setup instructions when model is missing
- Works with both relative and absolute paths

Setup Instructions:
1. Install Piper TTS: pip install piper-tts
2. Download Russian model files:
   - https://huggingface.co/rhasspy/piper-voices/resolve/tacotron2/v1/ru_RU/ru_RU-embed-ruslan-medium.onnx
   - https://huggingface.co/rhasspy/piper-voices/resolve/tacotron2/v1/ru_RU/ru_RU-embed-ruslan-medium.onnx.json
3. Place both files in the same directory as this script (or update PIPER_MODEL path)
"""

import asyncio
import websockets
import subprocess
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('piper-websocket')

# Configuration - FIXED MODEL PATH HANDLING
# Place your model files in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_MODEL = os.path.join(SCRIPT_DIR, "ru_RU-embed-ruslan-medium.onnx")  # Absolute path
PIPER_PATH = "piper"  # Path to piper executable
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8765  # WebSocket port
PCM_CHUNK_SIZE = 4096  # Size of audio chunks to stream

def verify_model():
    """Verify model files exist and are accessible"""
    # Check if model file exists
    if not os.path.exists(PIPER_MODEL):
        logger.error(f"Model file not found: {PIPER_MODEL}")
        logger.error("Please download the Russian model files:")
        logger.error("1. https://huggingface.co/rhasspy/piper-voices/resolve/tacotron2/v1/ru_RU/ru_RU-embed-ruslan-medium.onnx")
        logger.error("2. https://huggingface.co/rhasspy/piper-voices/resolve/tacotron2/v1/ru_RU/ru_RU-embed-ruslan-medium.onnx.json")
        logger.error(f"Place both files in: {SCRIPT_DIR}")
        return False
    
    # Check if JSON metadata file exists
    json_path = PIPER_MODEL.replace(".onnx", ".onnx.json")
    if not os.path.exists(json_path):
        logger.error(f"Model metadata not found: {json_path}")
        logger.error("Please download the matching .json file for the model")
        return False
    
    return True

async def process_text(websocket, text):
    """Process text through Piper and stream audio back"""
    try:
        # Start Piper process with raw output
        process = await asyncio.create_subprocess_exec(
            PIPER_PATH,
            "--model", PIPER_MODEL,
            "--output-raw",  # Output raw PCM audio instead of WAV
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"Processing text: '{text}'")
        
        # Send text to Piper
        process.stdin.write(text.encode() + b'\n')
        await process.stdin.drain()
        process.stdin.close()
        
        # Stream audio back in chunks
        while True:
            chunk = await process.stdout.read(PCM_CHUNK_SIZE)
            if not chunk:
                break
            await websocket.send(chunk)
        
        # Wait for process to complete
        await process.wait()
        
        if process.returncode != 0:
            error = (await process.stderr.read()).decode()
            logger.error(f"Piper error: {error}")
            await websocket.send(f"ERROR: Piper exited with code {process.returncode}".encode())
            
    except FileNotFoundError:
        error_msg = "ERROR: Piper executable not found. Please install Piper TTS and ensure it's in your PATH."
        logger.error(error_msg)
        await websocket.send(error_msg.encode())
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        logger.exception("Unexpected error")
        await websocket.send(error_msg.encode())

async def handler(websocket, path):
    """WebSocket connection handler"""
    client_ip = websocket.remote_address[0]
    logger.info(f"New connection from {client_ip}")
    
    try:
        async for message in websocket:
            # Process incoming text message
            await process_text(websocket, message)
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed: {e}")
    except Exception as e:
        logger.exception("Connection error")
        try:
            await websocket.send(f"ERROR: {str(e)}".encode())
        except:
            pass

async def main():
    """Start the WebSocket server"""
    try:
        # Verify model files exist
        if not verify_model():
            logger.error("Server cannot start without valid model files")
            return
            
        # Verify Piper is available
        process = await asyncio.create_subprocess_exec(
            PIPER_PATH, "--help",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await process.communicate()
        
        if process.returncode != 0:
            logger.error("Piper executable not found or not working. Please check your installation.")
            return
            
        # Start server
        server = await websockets.serve(
            handler,
            HOST,
            PORT,
            ping_interval=None,  # Disable pings for uninterrupted audio streaming
            max_size=None        # Allow large messages
        )
        
        logger.info(f"Piper WebSocket server running at ws://{HOST}:{PORT}")
        logger.info(f"Using model: {PIPER_MODEL}")
        logger.info("Connect with a WebSocket client to send text and receive PCM audio")
        
        # Keep server running
        await server.wait_closed()
        
    except Exception as e:
        logger.exception("Failed to start server")
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    asyncio.run(main())