from fastapi import FastAPI, WebSocket
from piper.voice import PiperVoice
import re  # For sentence splitting

app = FastAPI()

# Load Piper model (do this once at startup)
MODEL_PATH = "./voices/ru/ru_RU-ruslan-medium.onnx"  # Path to your .onnx file
voice = PiperVoice.load(MODEL_PATH)

@app.websocket("/tts")
async def tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive text from client (can be incremental)
            text = await websocket.receive_text()
            if not text:
                continue

            # Split into sentences for progressive streaming (optional but improves real-time feel)
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

            for sentence in sentences:
                if sentence.strip():
                    # Stream audio chunks from Piper
                    for audio_bytes in voice.synthesize_stream_raw(sentence):
                        # Send raw PCM bytes to client
                        await websocket.send_bytes(audio_bytes)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()