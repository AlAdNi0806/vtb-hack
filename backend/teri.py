# server.py
import asyncio
import base64
import json
import math
import os
from typing import AsyncGenerator

import torch
import torchaudio
import websockets
from chatterbox.tts import ChatterboxTTS
  # or ChatterboxTTS

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8765"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000  # match model.sr after load
CHUNK_MS = 200  # send ~200ms chunks (tweak)

# Load model once
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SAMPLE_RATE = getattr(model, "sr", SAMPLE_RATE)

# Generator that yields chunks (bytes) from full wav tensor
def wav_to_chunks_bytes(wav_tensor: torch.Tensor, sr: int, chunk_ms: int) -> AsyncGenerator[bytes, None]:
    # wav_tensor: (1, N) float32 tensor in [-1,1]
    num_samples_per_chunk = int(sr * (chunk_ms / 1000.0))
    total = wav_tensor.shape[-1]
    idx = 0
    while idx < total:
        end = min(total, idx + num_samples_per_chunk)
        chunk = wav_tensor[..., idx:end].cpu()
        # convert to 16-bit PCM
        chunk_int16 = (chunk * 32767.0).clamp(-32768, 32767).to(torch.int16)
        wav_bytes = torchaudio.functional.save_to_buffer(chunk_int16.unsqueeze(0), format="wav", sample_rate=sr)
        yield wav_bytes
        idx = end

async def synthesize_stream(text: str, language: str | None = None, voice: str | None = None):
    # model.generate returns 1D float tensor or numpy array
    kwargs = {}
    if language:
        kwargs["language_id"] = language
    if voice:
        kwargs["voice"] = voice
    # synchronous generation (may be GPU-bound)
    wav = model.generate(text, **kwargs)  # returns FloatTensor [-1..1]
    if isinstance(wav, torch.Tensor):
        wav_t = wav
    else:
        wav_t = torch.from_numpy(wav).float()
    async for chunk in wav_to_chunks_bytes(wav_t, SAMPLE_RATE, CHUNK_MS):
        yield chunk

async def handler(ws):
    """
    Protocol:
      - Client sends JSON messages:
        {"type":"synthesize","text":"Hello","language":"ru","voice":null}
      - Server streams back binary messages containing base64-encoded WAV chunks:
        {"type":"chunk","data":"BASE64_WAV","sr":24000}
      - Finalization message:
        {"type":"end"}
      - Errors:
        {"type":"error", "message":"..."}
    """
    try:
        async for message in ws:
            try:
                msg = json.loads(message)
            except Exception:
                await ws.send(json.dumps({"type":"error", "message":"invalid json"}))
                continue

            if msg.get("type") == "synthesize":
                text = msg.get("text", "")
                language = msg.get("language")
                voice = msg.get("voice")
                # offload generate to threadpool to avoid blocking event loop
                loop = asyncio.get_running_loop()
                try:
                    gen = await loop.run_in_executor(None, lambda: synthesize_stream(text, language, voice))
                    # synthesize_stream itself is async generator; we need to iterate it.
                    async for chunk in gen:
                        # base64 encode bytes
                        b64 = base64.b64encode(chunk).decode("ascii")
                        await ws.send(json.dumps({"type":"chunk", "data": b64, "sr": SAMPLE_RATE}))
                    await ws.send(json.dumps({"type":"end"}))
                except Exception as e:
                    await ws.send(json.dumps({"type":"error", "message": str(e)}))
            else:
                await ws.send(json.dumps({"type":"error", "message":"unknown message type"}))
    except websockets.exceptions.ConnectionClosed:
        return

if __name__ == "__main__":
    print(f"Starting server on ws://{HOST}:{PORT} (device={DEVICE})")

    async def run():
        async with websockets.serve(handler, HOST, PORT, max_size=2**25):
            await asyncio.Future()          # run forever

    asyncio.run(run())
