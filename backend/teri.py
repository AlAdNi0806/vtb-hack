#!/usr/bin/env python3
"""
Bidirectional WebSocket -> Piper TTS streaming endpoint
pip install fastapi uvicorn websockets asyncio aiofiles
"""

import asyncio, json, logging, subprocess, shlex, os
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

LOG = logging.getLogger("piper_ws")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# CONFIG  – change paths if you cloned voices elsewhere
# ------------------------------------------------------------------
MODEL_PATH   = "ru_RU-dmitri-medium.onnx"
CONFIG_PATH  = "ru_RU-dmitri-medium.onnx.json"
SAMPLE_RATE  = 22_050
FRAME_SIZE   = SAMPLE_RATE * 2 // 10      # 0.1 s worth of 16-bit PCM
PIPER_CMD    = shlex.split(
    f"piper --model {MODEL_PATH} --config {CONFIG_PATH} --output-raw"
)

# ------------------------------------------------------------------
# Piper process wrapper (asyncio subprocess)
# ------------------------------------------------------------------
class PiperStreamer:
    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self):
        if self._proc is None:
            LOG.info("Starting Piper: %s", " ".join(PIPER_CMD))
            self._proc = await asyncio.create_subprocess_exec(
                *PIPER_CMD,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

    async def synthesize(self, text: str) -> bytes:
        """Send one sentence to Piper and return raw PCM."""
        await self.start()
        assert self._proc and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write((text + "\n").encode())
        await self._proc.stdin.drain()

        pcm = await self._proc.stdout.readuntil(b"\n")  # piper ends with newline
        return pcm[:-1]  # drop newline

    async def stop(self):
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()
            self._proc = None


# ------------------------------------------------------------------
# FastAPI + WebSocket
# ------------------------------------------------------------------
app = FastAPI(title="PiperTTS-WS", version="1.0")

class TextMsg(BaseModel):
    text: str

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    piper = PiperStreamer()
    buffer = ""                       # accumulate until sentence end
    sentence_end = {".", "!", "?", "。"}

    try:
        while True:
            data = await ws.receive_text()
            msg  = TextMsg.parse_raw(data)
            buffer += msg.text

            # flush complete sentences
            while True:
                idx = max((buffer.find(sep) for sep in sentence_end), default=-1)
                if idx == -1:
                    break
                sentence, buffer = buffer[:idx+1], buffer[idx+1:]
                audio = await piper.synthesize(sentence.strip())
                await ws.send_bytes(audio)

    except WebSocketDisconnect:
        LOG.info("Client disconnected")
    except Exception as e:
        LOG.exception(e)
    finally:
        await piper.stop()


# ------------------------------------------------------------------
# Run:  uvicorn app:app --host 0.0.0.0 --port 8000
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)