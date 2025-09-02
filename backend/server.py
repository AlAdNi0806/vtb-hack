import asyncio
import websockets
import json
import torch
import torchaudio
import io
import nemo.collections.asr as nemo_asr
from functools import partial

# Load model once
MODEL_NAME = "nvidia/stt_ru_fastconformer_hybrid_large_pc"
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)

# Thread pool for CPU-bound ASR
pool = None   # will be set to concurrent.futures.ThreadPoolExecutor()

# --------------------------------------------------------------
def transcribe_chunk(buffer: bytes) -> str:
    """
    Convert raw bytes to tensor and run NeMo ASR.
    buffer: raw little-endian 16-bit PCM @ 16 kHz, mono
    """
    # 1. bytes -> torch tensor
    waveform, sr = torchaudio.load(io.BytesIO(buffer))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # 2. NeMo wants (B, T) float tensor
    waveform = waveform.squeeze(0)

    # 3. run inference
    with torch.no_grad():
        hyps = asr_model.transcribe([waveform], batch_size=1)
    # hyps -> ([text, ...], )  or ([text, ...], [time_stamps])
    return hyps[0][0]

# --------------------------------------------------------------
async def recognize(websocket, path):
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

    async with websockets.serve(recognize, "localhost", 8765):
        print("ASR WebSocket server on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())