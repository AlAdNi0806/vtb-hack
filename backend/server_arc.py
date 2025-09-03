import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures
import webrtcvad  # For voice activity detection
from collections import deque
import time

# --- Configuration ---
MODEL_NAME = "stt_ru_fastconformer_hybrid_large_pc"  # Changed to model with punctuation and capitalization for better sentence end prediction
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # Frame size for VAD (10, 20, or 30 ms)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Frame size in samples
VAD_AGGRESSIVENESS = 2  # 0-3, with 3 being the most aggressive
SILENCE_TIMEOUT = 1.5  # Seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to consider it valid speech
PARTIAL_INTERVAL = 0.5  # Interval in seconds to send partial transcripts
MIN_PARTIAL_DURATION = 0.5  # Minimum audio duration in seconds for partial transcription
MIN_PARTIAL_BYTES = int(SAMPLE_RATE * 2 * MIN_PARTIAL_DURATION)  # 16-bit audio

# --- Global Resources ---
print("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecHybridRNNTCTCModel.from_pretrained(MODEL_NAME)  # Use hybrid model for P&C
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
print("Model loaded and ready.")

# --------------------------------------------------------------
def transcribe_audio(pcm_bytes: bytes) -> str:
    """
    Transcribes audio directly from memory.
    This function is designed to be run in a separate thread.
    """
    if not pcm_bytes:
        return ""

    # Convert bytes to a NumPy array of 16-bit integers
    pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    
    # Convert to a normalized float32 NumPy array, as expected by the model
    waveform_np = pcm_i16.astype(np.float32) / 32768.0

    try:
        # Run inference in a no_grad context for efficiency
        with torch.no_grad():
            hyps = asr_model.transcribe([waveform_np], batch_size=1)
        
        # Check if we got a valid, non-empty result
        if hyps and len(hyps) > 0:
            hypothesis = hyps[0]
            
            if hasattr(hypothesis, 'text'):
                text = hypothesis.text
                print(f"Transcription result: '{text}'")
                return text
            elif isinstance(hypothesis, str):
                print(f"Transcription result: '{hypothesis}'")
                return hypothesis

        return ""

    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# --------------------------------------------------------------
class SpeechBuffer:
    """Buffer that handles voice activity detection and speech segmentation"""
    
    def __init__(self, sample_rate, frame_duration_ms, silence_timeout, min_speech_duration):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        
        self.buffer = bytearray()
        self.speech_start_time = None
        self.last_voice_time = None
        self.is_speaking = False
        
    def add_audio(self, audio_data):
        """Add audio data to the buffer"""
        self.buffer.extend(audio_data)
        
    def process_frames(self):
        """Process frames for voice activity detection"""
        results = []
        frame_size_bytes = self.frame_size * 2  # 16-bit audio
        
        # Process all complete frames in the buffer
        while len(self.buffer) >= frame_size_bytes:
            frame = self.buffer[:frame_size_bytes]
            self.buffer = self.buffer[frame_size_bytes:]
            
            # Check if frame contains speech
            is_speech = vad.is_speech(frame, self.sample_rate)
            current_time = time.time()
            
            if is_speech:
                if not self.is_speaking:
                    # Speech just started
                    self.is_speaking = True
                    self.speech_start_time = current_time
                
                self.last_voice_time = current_time
                
            else:
                if self.is_speaking:
                    # Check if we've had enough silence to consider speech ended
                    if current_time - self.last_voice_time > self.silence_timeout:
                        # Speech has ended
                        self.is_speaking = False
                        # Check if the speech was long enough to be valid
                        if self.last_voice_time - self.speech_start_time >= self.min_speech_duration:
                            results.append(("speech_end", None))
            
            # If we're speaking, add the frame to the speech buffer
            if self.is_speaking:
                results.append(("frame", frame))
        
        return results
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = bytearray()
        self.is_speaking = False
        self.speech_start_time = None
        self.last_voice_time = None

# --------------------------------------------------------------
async def recognize(websocket):
    """Handles a single client connection."""
    print("Client connected.")
    loop = asyncio.get_running_loop()
    
    # Initialize speech buffer
    speech_buffer = SpeechBuffer(
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        silence_timeout=SILENCE_TIMEOUT,
        min_speech_duration=MIN_SPEECH_DURATION
    )
    
    current_speech = bytearray()
    last_partial_time = time.time()
    last_transcript = ""
    
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                # Add audio to buffer and process for VAD
                speech_buffer.add_audio(msg)
                events = speech_buffer.process_frames()
                
                for event_type, data in events:
                    if event_type == "frame":
                        # Add frame to current speech segment
                        current_speech.extend(data)
                    elif event_type == "speech_end":
                        # Process the completed speech segment
                        if current_speech:
                            audio_data = bytes(current_speech)
                            current_speech.clear()
                            
                            # Transcribe the speech segment
                            text = await loop.run_in_executor(
                                pool, partial(transcribe_audio, audio_data)
                            )
                            
                            if text:
                                await websocket.send(json.dumps({
                                    "transcript": text,
                                    "is_final": True
                                }))
                            last_transcript = text  # Update last transcript after final

                # Check for partial transcription if still speaking
                if speech_buffer.is_speaking:
                    current_time = time.time()
                    if current_time - last_partial_time >= PARTIAL_INTERVAL and len(current_speech) >= MIN_PARTIAL_BYTES:
                        audio_data = bytes(current_speech)
                        text = await loop.run_in_executor(
                            pool, partial(transcribe_audio, audio_data)
                        )
                        
                        if text and text != last_transcript:
                            print(f"Partial transcription: '{text}'")
                            await websocket.send(json.dumps({
                                "transcript": text,
                                "is_final": False
                            }))
                            last_transcript = text
                        
                        last_partial_time = current_time
            
            # Handle JSON messages for control
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e.reason} (code: {e.code})")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Process any remaining speech when the client disconnects
        if current_speech:
            print("Processing remaining speech...")
            text = await loop.run_in_executor(
                pool, partial(transcribe_audio, bytes(current_speech))
            )
            if text:
                await websocket.send(json.dumps({
                    "transcript": text,
                    "is_final": True,
                    "is_complete": True
                }))
        print("Client disconnected.")

# --------------------------------------------------------------
async def main():
    """Starts the WebSocket server."""
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"ASR WebSocket server started on ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())