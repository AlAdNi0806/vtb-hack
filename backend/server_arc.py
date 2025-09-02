import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures
import time

# --- Configuration ---
MODEL_NAME = "nvidia/stt_ru_fastconformer_hybrid_large_pc"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 3  # Fallback chunk size if needed
CHUNK_SIZE_BYTES = CHUNK_DURATION_SECONDS * SAMPLE_RATE * 2  # 16-bit PCM
PAUSE_THRESHOLD_MS = 500  # ms of silence to trigger finalization
PARTIAL_INTERVAL_MS = 300  # Send partials every 300ms during speech
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
VAD_WINDOW_SIZE_BYTES = 480  # ~30ms window at 16kHz, 16-bit (16000*0.03*2)

# --- Global Resources ---
print("Loading NeMo ASR model with punctuation...")
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
print("Model loaded and ready.")

# --- Simple Energy-Based VAD ---
def is_silence(pcm_bytes: bytes, threshold: float = SILENCE_THRESHOLD) -> bool:
    if len(pcm_bytes) == 0:
        return True
    pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    waveform_np = pcm_i16.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(waveform_np ** 2))
    return rms < threshold

# --- Transcription Function ---
def transcribe_audio(pcm_bytes: bytes) -> str:
    """
    Transcribes audio from memory using the hybrid model (includes punctuation).
    """
    if not pcm_bytes:
        return ""

    pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    waveform_np = pcm_i16.astype(np.float32) / 32768.0

    try:
        with torch.no_grad():
            hyps = asr_model.transcribe([waveform_np], batch_size=1)
        
        if hyps and len(hyps) > 0:
            hypothesis = hyps[0]
            if hasattr(hypothesis, 'text'):
                return hypothesis.text
            elif isinstance(hypothesis, str):
                return hypothesis
        return ""

    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# --- WebSocket Handler ---
async def recognize(websocket):
    """Handles a single client connection with VAD for drafts and finals."""
    print("Client connected.")
    loop = asyncio.get_running_loop()
    audio_buffer = bytearray()  # Accumulate speech audio
    silence_duration_ms = 0.0
    last_partial_time = time.time() * 1000
    current_transcript = ""  # For accumulating raw text if needed
    
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                # Process VAD in small windows
                for i in range(0, len(msg), VAD_WINDOW_SIZE_BYTES):
                    window = msg[i:i + VAD_WINDOW_SIZE_BYTES]
                    window_duration_ms = (len(window) / (SAMPLE_RATE * 2)) * 1000
                    
                    if is_silence(window):
                        silence_duration_ms += window_duration_ms
                    else:
                        silence_duration_ms = 0.0
                        audio_buffer.extend(window)  # Only add speech audio

                # Send partial if interval elapsed and there's audio
                current_time = time.time() * 1000
                if current_time - last_partial_time >= PARTIAL_INTERVAL_MS and audio_buffer:
                    partial_text = await loop.run_in_executor(
                        pool, partial(transcribe_audio, bytes(audio_buffer))
                    )
                    if partial_text and partial_text != current_transcript:
                        current_transcript = partial_text
                        await websocket.send(json.dumps({"partial": partial_text}))
                    last_partial_time = current_time

                # If silence exceeds threshold, finalize the segment
                if silence_duration_ms > PAUSE_THRESHOLD_MS and audio_buffer:
                    # Transcribe the full buffer (model adds punctuation)
                    final_text = await loop.run_in_executor(
                        pool, partial(transcribe_audio, bytes(audio_buffer))
                    )
                    if final_text:
                        await websocket.send(json.dumps({"final": final_text}))
                    
                    # Reset for next utterance
                    audio_buffer.clear()
                    silence_duration_ms = 0.0
                    current_transcript = ""
                    last_partial_time = current_time
            
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e.reason} (code: {e.code})")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Finalize any remaining audio
        if audio_buffer:
            final_text = await loop.run_in_executor(
                pool, partial(transcribe_audio, bytes(audio_buffer))
            )
            if final_text:
                await websocket.send(json.dumps({"final": final_text}))
        print("Client disconnected.")

# --- Server Startup ---
async def main():
    """Starts the WebSocket server."""
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"ASR WebSocket server started on ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())