import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures

# --- Configuration ---
MODEL_NAME = "nvidia/stt_ru_fastconformer_hybrid_large_pc"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 3  # Process audio in 3-second chunks
CHUNK_SIZE_BYTES = CHUNK_DURATION_SECONDS * SAMPLE_RATE * 2  # 16-bit PCM

# --- Global Resources ---
print("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
print("Model loaded and ready.")

# --------------------------------------------------------------
def transcribe_chunk(pcm_bytes: bytes) -> str:
    """
    Transcribes an audio chunk directly from memory.
    This function is designed to be run in a separate thread.
    """
    if not pcm_bytes:
        return ""

    # Convert bytes to a NumPy array of 16-bit integers
    pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    
    # Convert to a normalized float32 NumPy array, as expected by the model
    waveform_np = pcm_i16.astype(np.float32) / 32768.0

    try:
        # Use the model's ability to transcribe from a NumPy array in memory
        with torch.no_grad():
            hyps = asr_model.transcribe([waveform_np], batch_size=1)
        
        if hyps and len(hyps) > 0:
            hypothesis = hyps[0]
            print(f"Transcription result: '{hypothesis}'")
            return hypothesis
        return ""

    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# --------------------------------------------------------------
async def recognize(websocket):
    """Handles a single client connection."""
    print("Client connected.")
    loop = asyncio.get_running_loop()
    buffer = bytearray()
    
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                buffer.extend(msg)

                # If we have a full chunk, process it
                if len(buffer) >= CHUNK_SIZE_BYTES:
                    current_chunk = bytes(buffer)
                    buffer.clear() # Clear buffer to start accumulating the next chunk

                    # Run the CPU-bound ASR in the thread pool
                    text = await loop.run_in_executor(
                        pool, partial(transcribe_chunk, current_chunk)
                    )
                    
                    if text:
                        await websocket.send(json.dumps({"transcript": text}))
            
            # (Optional) Handle JSON messages for control, e.g., stop
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e.reason} (code: {e.code})")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Process any remaining audio in the buffer when the client disconnects
        if buffer:
            print("Processing remaining audio chunk...")
            text = await loop.run_in_executor(
                pool, partial(transcribe_chunk, bytes(buffer))
            )
            if text:
                await websocket.send(json.dumps({"final_transcript": text}))
        print("Client disconnected.")

# --------------------------------------------------------------
async def main():
    """Starts the WebSocket server."""
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"ASR WebSocket server started on ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())