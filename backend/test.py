# client.py
import asyncio
import base64
import json
import os
import websockets
import sounddevice as sd
import soundfile as sf
import io

WS_URL = os.getenv("WS_URL", "ws://192.168.0.176:8765")

async def run():
    async with websockets.connect(WS_URL, max_size=2**25) as ws:
        req = {"type":"synthesize","text":"Привет, как дела?"}
        await ws.send(json.dumps(req))
        while True:
            msg = await ws.recv()
            obj = json.loads(msg)
            if obj["type"] == "chunk":
                b = base64.b64decode(obj["data"])
                # read wav bytes and play
                data, sr = sf.read(io.BytesIO(b))
                sd.play(data, sr)
                sd.wait()
            elif obj["type"] == "end":
                print("done")
                break
            elif obj["type"] == "error":
                print("error:", obj.get("message"))
                break

asyncio.run(run())
