import asyncio
import websockets
import json
import torch
import nemo.collections.asr as nemo_asr

# Configuration
MODEL_NAME        = "nvidia/parakeet-tdt-0.6b-v3"
SERVER_HOST       = "0.0.0.0"
SERVER_PORT       = 8765

print(f"Loading model {MODEL_NAME}…")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
print("Model loaded.")

async def recognize(websocket):
    print("Client connected.")
    buffer = bytearray()
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                buffer.extend(msg)
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break

            # Every time we receive audio bytes, transcribe what's available
            if buffer:
                audio = bytes(buffer)
                buffer.clear()

                # Partial results (timestamps included)
                output = asr_model.transcribe([audio], timestamps=True)[0]
                text = output.text
                segments = output.timestamp.get('segment', [])

                await websocket.send(json.dumps({
                    "transcript": text,
                    "segments": segments,
                    "is_final": True  # Parakeet handles segmentation itself
                }))
    except websockets.exceptions.ConnectionClosed as e:
        print("Connection closed:", e)
    finally:
        print("Client disconnected.")

async def main():
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"Server running at ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
