#!/usr/bin/env python3
"""
Minimal client: streams text → server, accumulates audio, saves .wav
"""

import asyncio, json, wave
import websockets
import numpy as np

URI = "ws://192.168.0.176:8000/ws"
TEXT = (
    "Привет! Это пример реального времени. "
    "Мы говорим с сервером Пайпер и сохраняем ответ в файл."
)

async def record() -> list[bytes]:
    """Return list of raw 16-bit 22 kHz PCM chunks."""
    chunks: list[bytes] = []
    async with websockets.connect(URI) as ws:
        await ws.send(json.dumps({"text": TEXT}))
        async for msg in ws:
            if isinstance(msg, bytes):        # audio chunk
                chunks.append(msg)
            else:                             # future JSON control
                print("ctrl:", msg)
    return chunks

def save_wav(chunks: list[bytes], path: str = "output.wav", sr: int = 22_050):
    """Glue chunks and write 16-bit mono .wav"""
    pcm = b"".join(chunks)
    audio = np.frombuffer(pcm, dtype=np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    print(f"saved {path}  ({len(audio)/sr:.2f} s)")

if __name__ == "__main__":
    chunks = asyncio.run(record())
    save_wav(chunks)