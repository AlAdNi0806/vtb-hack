#!/usr/bin/env python3
"""
Self-contained WebSocket -> Piper TTS streaming endpoint
- downloads piper binary + ru voice on first run
- caches in ./piper_cache/
- MIT licenced
"""

import os
import asyncio
import logging
import json
import stat
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

LOG = logging.getLogger("piper_ws")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# CONFIG – cachedir inside repo (Modal writable)
# ------------------------------------------------------------------
CACHE_DIR   = Path(__file__).with_name("piper_cache")
PIPER_BIN   = CACHE_DIR / "piper"
VOICE_DIR   = CACHE_DIR / "voices"
MODEL_PATH  = VOICE_DIR / "ru_RU-dmitri-medium.onnx"
CONFIG_PATH = VOICE_DIR / "ru_RU-dmitri-medium.onnx.json"
SAMPLE_RATE = 22_050

# URLs (official releases)
PIPER_URL = (
    "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz"
)
VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/"
)

# ------------------------------------------------------------------
# One-time bootstrap
# ------------------------------------------------------------------
async def download_once(url: str, dest: Path, session: aiohttp.ClientSession) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    LOG.info("Downloading %s -> %s", url, dest)
    async with session.get(url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            async for chunk in resp.content.iter_chunked(8192):
                f.write(chunk)

async def bootstrap() -> None:
    async with aiohttp.ClientSession() as session:
        # 1. download & unpack to cache
        tgz = CACHE_DIR / "piper.tgz"
        await download_once(PIPER_URL, tgz, session)
        if not PIPER_BIN.exists():
            shutil.unpack_archive(str(tgz), extract_dir=CACHE_DIR)
        # 2. copy to /tmp so it can be executed
        exec_bin = Path("/tmp/piper")
        if not exec_bin.exists():
            shutil.copy2(PIPER_BIN, exec_bin)
            exec_bin.chmod(0o755)
        global PIPER_BIN
        PIPER_BIN = exec_bin
        # 3. voices
        for fname in (MODEL_PATH.name, CONFIG_PATH.name):
            await download_once(VOICE_URL + fname, VOICE_DIR / fname, session)

# ------------------------------------------------------------------
# Piper wrapper (unchanged except paths)
# ------------------------------------------------------------------
class PiperStreamer:
    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None

    async def start(self):
        if self._proc is None:
            await bootstrap()   # idempotent
            LOG.info("Starting Piper: %s", PIPER_BIN)
            self._proc = await asyncio.create_subprocess_exec(
                str(PIPER_BIN),
                "--model", str(MODEL_PATH),
                "--config", str(CONFIG_PATH),
                "--output-raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

    async def synthesize(self, text: str) -> bytes:
        await self.start()
        assert self._proc and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write((text + "\n").encode())
        await self._proc.stdin.drain()
        pcm = await self._proc.stdout.readuntil(b"\n")
        return pcm[:-1]

    async def stop(self):
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()
            self._proc = None

# ------------------------------------------------------------------
# FastAPI WebSocket
# ------------------------------------------------------------------
app = FastAPI(title="PiperTTS-WS-Standalone", version="1.0")

class TextMsg(BaseModel):
    text: str

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    piper = PiperStreamer()
    buffer = ""
    sentence_end = {".", "!", "?", "。"}

    try:
        while True:
            data = await ws.receive_text()
            msg = TextMsg.parse_raw(data)
            buffer += msg.text

            while True:
                idx = max((buffer.find(sep) for sep in sentence_end), default=-1)
                if idx == -1:
                    break
                sentence, buffer = buffer[: idx + 1], buffer[idx + 1 :]
                audio = await piper.synthesize(sentence.strip())
                await ws.send_bytes(audio)

    except WebSocketDisconnect:
        LOG.info("Client disconnected")
    except Exception as e:
        LOG.exception(e)
    finally:
        await piper.stop()

# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("teri:app", host="0.0.0.0", port=8000, reload=True)