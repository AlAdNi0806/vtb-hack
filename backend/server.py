import io
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, WebSocket

app = FastAPI()

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="nvidia/parakeet-rnnt-1.1b",
    map_location="cpu"
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)

            if len(buffer) >= 16000 * 2 * 3:  # 3 seconds
                # Convert raw PCM bytes → float32 numpy
                audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0

                # Pass directly to NeMo
                result = asr_model.transcribe([audio_np])
                if isinstance(result, list) and len(result) > 0:
                    text = result[0]
                    if isinstance(text, list):
                        text = text[0] if text else ""
                else:
                    text = str(result)

                print("Transcription raw result:", result, type(result))
                print("Final text:", text, type(text))
                await websocket.send_text(str(text))
                buffer.clear()

    except Exception as e:
        print("WebSocket closed:", e)

