#!/usr/bin/env python3
"""
Piper TTS WebSocket Server
==========================
A single-file WebSocket server that converts text to Russian speech using Piper TTS.

Features:
- Accepts text via WebSocket connection
- Streams back raw PCM audio in real-time (16-bit mono PCM) [[6]]
- Supports Russian language with appropriate model
- Simple implementation requiring only standard libraries

Setup Requirements:
1. Install Piper TTS: https://pypi.org/project/piper-tts/
2. Download a Russian model (e.g., 'ru_RU-embed-ruslan-medium.onnx')
   - Models can be found at: https://huggingface.co/rhasspy/piper-voices/tree/tacotron2
3. Place this script in your project directory

Usage:
1. Save this file as piper_websocket.py
2. Make executable: chmod +x piper_websocket.py
3. Run: ./piper_websocket.py
4. Connect to ws://localhost:8765 with a WebSocket client

Client Notes:
- The audio returned is raw 16-bit mono PCM, NOT WAV format [[4]]
- Sample rate matches the voice model (typically 22050 Hz)
- You'll need to handle audio playback with correct parameters on the client side
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

# Configuration - CHANGE THESE VALUES ACCORDING TO YOUR SETUP
PIPER_MODEL = "ru_RU-embed-ruslan-medium.onnx"  # Path to your Russian model file
PIPER_PATH = "piper"  # Path to piper executable (use full path if not in PATH)
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8765  # WebSocket port
PCM_CHUNK_SIZE = 4096  # Size of audio chunks to stream

async def process_text(websocket, text):
    """Process text through Piper and stream audio back"""
    try:
        # Start Piper process with raw output
        process = await asyncio.create_subprocess_exec(
            PIPER_PATH,
            "--model", PIPER_MODEL,
            "--output-raw",  # Output raw PCM audio instead of WAV [[6]]
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

async def handler(websocket):
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
        print("\nSetup instructions:")
        print("1. Install Piper: pip install piper-tts")
        print("2. Download a Russian model (e.g., ru_RU-embed-ruslan-medium.onnx)")
        print("3. Update PIPER_MODEL in this script to point to your model file")
        print("4. Ensure Piper is in your PATH or update PIPER_PATH")

if __name__ == "__main__":
    asyncio.run(main())