import asyncio
import websockets
import json
import torch
import nemo.collections.asr as nemo_asr
import numpy as np
import time

# Configuration
MODEL_NAME        = "nvidia/parakeet-tdt-0.6b-v3"
SERVER_HOST       = "0.0.0.0"
SERVER_PORT       = 8765
PARTIAL_INTERVAL   = 0.4      # seconds between partial emits
SILENCE_TIMEOUT    = 0.8      # how long silence means end-of-utterance
SAMPLE_RATE        = 16000
MIN_CHUNK_SAMPLES  = 8000     # 0.5s worth of audio

print(f"Loading model {MODEL_NAME}…")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
print("Model loaded.")

# Convert PCM16 → float32 numpy
def audio_bytes_to_np(pcm_bytes: bytes) -> np.ndarray:
    pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm_i16.astype(np.float32) / 32768.0

async def recognize(websocket):
    print("Client connected.")
    buffer = bytearray()
    last_partial_t = 0
    last_voice_t = time.time()
    is_speaking = False

    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                buffer.extend(msg)

                # Only process if buffer has enough samples
                if len(buffer) >= MIN_CHUNK_SAMPLES * 2:
                    audio = bytes(buffer)
                    buffer.clear()

                    waveform = audio_bytes_to_np(audio)

                    # Run ASR
                    output = asr_model.transcribe([waveform], timestamps=False)[0]
                    text = output if isinstance(output, str) else output.text

                    now = time.time()
                    silence_elapsed = now - last_voice_t

                    if text.strip():
                        last_voice_t = now
                        is_speaking = True

                        # --- Decide partial vs final ---
                        if text.endswith((".", "?", "!")):
                            # Treat as final if it looks like sentence boundary
                            await websocket.send(json.dumps({
                                "transcript": text,
                                "is_final": True
                            }))
                            is_speaking = False
                        elif now - last_partial_t > PARTIAL_INTERVAL:
                            await websocket.send(json.dumps({
                                "transcript": text,
                                "is_final": False
                            }))
                            last_partial_t = now

                    # If silent long enough after speech → finalize
                    if is_speaking and silence_elapsed > SILENCE_TIMEOUT:
                        await websocket.send(json.dumps({
                            "transcript": text,
                            "is_final": True
                        }))
                        is_speaking = False

            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print("Connection handler failed:", e)
    finally:
        print("Client disconnected.")


async def main():
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"Server running at ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
