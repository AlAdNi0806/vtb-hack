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
MODEL_NAME = "nvidia/stt_ru_conformer_transducer_large"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # Frame size for VAD (10, 20, or 30 ms)
VAD_AGGRESSIVENESS = 2  # 0-3, with 3 being the most aggressive
SILENCE_TIMEOUT = 1.0  # Seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 0.25 # Minimum speech duration to consider it valid speech

# NEW: How often to run streaming transcription (in seconds)
# A smaller value is more "real-time" but more computationally expensive.
TRANSCRIPTION_INTERVAL = 0.5 
TRANSCRIPTION_CHUNK_SAMPLES = int(SAMPLE_RATE * TRANSCRIPTION_INTERVAL)
TRANSCRIPTION_CHUNK_BYTES = TRANSCRIPTION_CHUNK_SAMPLES * 2 # 16-bit audio

# --- Global Resources ---
print("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
# Set model to evaluation mode
asr_model.eval()
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
print("Model loaded and ready.")

# --------------------------------------------------------------
# NEW: Function dedicated to streaming transcription
def transcribe_stream_chunk(audio_chunk_bytes: bytes, state, context) -> (str, tuple, tuple):
    """
    Transcribes a single chunk of audio in a streaming fashion.
    
    Args:
        audio_chunk_bytes: The raw bytes of the audio chunk.
        state: The previous decoder state from the ASR model.
        context: The previous acoustic encoder context from the ASR model.

    Returns:
        A tuple containing (transcribed_text, new_state, new_context).
    """
    if not audio_chunk_bytes:
        return "", state, context

    # Convert bytes to a normalized float32 NumPy array
    pcm_i16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
    waveform_np = pcm_i16.astype(np.float32) / 32768.0
    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)

    try:
        with torch.no_grad():
            # Use the model's streaming transcription method
            hypotheses, next_state, next_context = asr_model.transcribe_stream(
                audio=waveform_tensor,
                state=state,
                context=context
            )
        if hypotheses and hypotheses[0]:
            return hypotheses[0], next_state, next_context
        return "", next_state, next_context
        
    except Exception as e:
        print(f"Streaming transcription error: {e}")
        return "", state, context


# --------------------------------------------------------------
class SpeechBuffer:
    """Buffer that handles voice activity detection and speech segmentation. (Largely unchanged)"""
    
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
                results.append(("frame", frame)) # Pass the frame for transcription buffer
                
            else: # Not speech
                if self.is_speaking:
                    # Check if we've had enough silence to consider speech ended
                    if current_time - self.last_voice_time > self.silence_timeout:
                        self.is_speaking = False
                        # Check if the speech was long enough to be valid
                        if self.last_voice_time - self.speech_start_time >= self.min_speech_duration:
                            results.append(("speech_end", None))
        
        return results
        
    def clear(self):
        """Clear the buffer"""
        self.buffer = bytearray()
        self.is_speaking = False
        self.speech_start_time = None
        self.last_voice_time = None

# --------------------------------------------------------------
# REVISED: Main handler with streaming logic
async def recognize(websocket):
    """Handles a single client connection with real-time streaming."""
    print("Client connected.")
    loop = asyncio.get_running_loop()
    
    # Initialize speech buffer for VAD
    speech_buffer = SpeechBuffer(
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        silence_timeout=SILENCE_TIMEOUT,
        min_speech_duration=MIN_SPEECH_DURATION
    )
    
    # Buffer for accumulating audio for the ASR model
    transcription_buffer = bytearray()
    
    # State for the streaming ASR model
    asr_state = None
    asr_context = None

    try:
        async for msg in websocket:
            if not isinstance(msg, bytes):
                continue

            # 1. Add audio to VAD buffer and detect speech events
            speech_buffer.add_audio(msg)
            events = speech_buffer.process_frames()
            
            for event_type, data in events:
                if event_type == "frame":
                    # If it's a speech frame, add it to our transcription buffer
                    transcription_buffer.extend(data)
                    
                    # 2. If transcription buffer is full, process it for an interim result
                    if len(transcription_buffer) >= TRANSCRIPTION_CHUNK_BYTES:
                        audio_chunk = bytes(transcription_buffer)
                        transcription_buffer.clear()

                        # Run streaming transcription in a separate thread
                        text, asr_state, asr_context = await loop.run_in_executor(
                            pool, partial(transcribe_stream_chunk, audio_chunk, asr_state, asr_context)
                        )

                        if text:
                            await websocket.send(json.dumps({
                                "transcript": text,
                                "is_final": False
                            }))

                elif event_type == "speech_end":
                    # 3. VAD detected end of speech. Process remaining audio.
                    if transcription_buffer:
                        audio_chunk = bytes(transcription_buffer)
                        transcription_buffer.clear()
                        
                        text, _, _ = await loop.run_in_executor(
                            pool, partial(transcribe_stream_chunk, audio_chunk, asr_state, asr_context)
                        )
                    else:
                        # If buffer is empty, we still might have a final hypothesis
                        text, _, _ = await loop.run_in_executor(
                            pool, partial(transcribe_stream_chunk, b'', asr_state, asr_context)
                        )

                    if text:
                        print(f"Final Transcript: '{text}'")
                        await websocket.send(json.dumps({
                            "transcript": text,
                            "is_final": True
                        }))

                    # 4. Reset ASR state for the next utterance
                    asr_state, asr_context = None, None

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e.reason} (code: {e.code})")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Client disconnected.")

# --------------------------------------------------------------
async def main():
    """Starts the WebSocket server."""
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"ASR WebSocket server started on ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())