#!/usr/bin/env python3
"""
Piper TTS WebSocket Client
==========================
A simple client that queries the Piper TTS WebSocket server and saves the result as a WAV file.

Usage:
1. Make sure the Piper WebSocket server is running
2. Run this script with your text and output filename:
   ./piper_client.py "Привет мир" output.wav

Requirements:
- websockets package (install with: pip install websockets)
"""

import asyncio
import websockets
import wave
import sys
import argparse

# Audio parameters - DEFAULT values (can be overridden)
DEFAULT_SAMPLE_RATE = 22050  # Common for Piper models
SAMPLE_WIDTH = 2     # 16-bit audio (2 bytes per sample)
CHANNELS = 1         # Mono audio

async def query_piper(text, output_file, sample_rate=DEFAULT_SAMPLE_RATE):
    """Query the Piper WebSocket server and save response as WAV"""
    try:
        # Connect to the WebSocket server
        async with websockets.connect('ws://192.168.0.176:8765') as websocket:
            print(f"Connected to Piper TTS server. Sending text: '{text}'")
            
            # Send text to the server
            await websocket.send(text)
            
            # Collect all audio chunks
            audio_data = b''
            while True:
                try:
                    # Set a timeout to detect end of transmission
                    chunk = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    if isinstance(chunk, str) and chunk.startswith("ERROR"):
                        print(f"Server error: {chunk}")
                        return False
                    audio_data += chunk
                except asyncio.TimeoutError:
                    # No more data received - end of transmission
                    break
            
            if not audio_data:
                print("Error: No audio data received from server")
                return False
                
            # Save as WAV file
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            print(f"Successfully saved audio to {output_file}")
            print(f"Audio parameters: {sample_rate} Hz, {SAMPLE_WIDTH*8}-bit, {CHANNELS} channel(s)")
            return True
            
    except ConnectionRefusedError:
        print("Error: Could not connect to Piper WebSocket server. Is it running?")
        print("Start the server with: python piper_websocket.py")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Query Piper TTS server and save as WAV')
    parser.add_argument('text', type=str, help='Text to convert to speech')
    parser.add_argument('output', type=str, help='Output WAV file name')
    parser.add_argument('--rate', type=int, default=DEFAULT_SAMPLE_RATE, 
                        help=f'Audio sample rate (default: {DEFAULT_SAMPLE_RATE})')
    
    args = parser.parse_args()
    
    # Run the async query with the specified sample rate
    success = asyncio.get_event_loop().run_until_complete(
        query_piper(args.text, args.output, args.rate)
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())