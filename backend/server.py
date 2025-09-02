import asyncio
import websockets
import json
import nemo.collections.asr as nemo_asr

# Load pretrained ASR model (example Russian model, change as needed)
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")

async def recognize(websocket, path):
    print("Client connected")
    audio_buffer = b""

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Accumulate audio bytes
                audio_buffer += message
                # Optionally: chunk audio_buffer and run partial decoding for real-time
                # Here, for demo, we run on the entire buffer each time
                transcript = asr_model.transcribe([audio_buffer])
                await websocket.send(json.dumps({"transcript": transcript[0]}))
            else:
                # Handle text messages like commands (optional)
                data = json.loads(message)
                if data.get("action") == "stop":
                    print("Stopping transcription as requested.")
                    break
    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(recognize, "localhost", 8765):
        print("ASR WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
