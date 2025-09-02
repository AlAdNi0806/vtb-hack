import asyncio
import websockets
import json
import torch
import torchaudio
import io
import nemo.collections.asr as nemo_asr
from functools import partial
import numpy as np

# Load model once
MODEL_NAME = "nvidia/stt_ru_fastconformer_hybrid_large_pc"
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)

# Thread pool for CPU-bound ASR
pool = None   # will be set to concurrent.futures.ThreadPoolExecutor()

# --------------------------------------------------------------
def transcribe_chunk(pcm_bytes: bytes) -> str:
    pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    waveform = torch.from_numpy(pcm_i16.astype(np.float32) / 32768.0)
    waveform = waveform.unsqueeze(0)          # (1, time)   OK
    # *** NEW LINE ***  make sure it is 2-D
    waveform = waveform.squeeze(1) if waveform.dim() == 3 else waveform

    with torch.no_grad():
        hyps = asr_model.transcribe([waveform], batch_size=1)
    return hyps[0][0]

# --------------------------------------------------------------
async def recognize(websocket):
    print("Client connected")
    loop = asyncio.get_running_loop()
    buffer = bytearray()

    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                buffer.extend(msg)
                # Run ASR in thread so we don’t block the loop
                text = await loop.run_in_executor(
                    pool, partial(transcribe_chunk, bytes(buffer))
                )
                await websocket.send(json.dumps({"transcript": text}))
            else:
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print("Client disconnected")

# --------------------------------------------------------------
async def main():
    import concurrent.futures
    global pool
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async with websockets.serve(recognize, "0.0.0.0", 8765):
        print("ASR WebSocket server on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())