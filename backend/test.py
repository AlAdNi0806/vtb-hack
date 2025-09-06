import asyncio
import websockets
import soundfile as sf
import numpy as np
import os

async def tts_client():
    uri = "ws://192.168.0.176:8000/tts"
    sample_rate = 22050  # match your model
    channels = 1         # or 2 for stereo
    path = "output_stream.wav"

    # remove existing file so we start fresh
    if os.path.exists(path):
        os.remove(path)

    print("starting TTS client...")
    # Open file for writing in blocks
    with sf.SoundFile(path, mode='w', samplerate=sample_rate, channels=channels, subtype='PCM_16') as f:
        async with websockets.connect(uri) as ws:
            print("connected to server")
            text = "Привет, это тест на русском языке. Это потоковый вывод."
            await ws.send(text)
            while True:
                try:
                    chunk = await ws.recv()
                    if isinstance(chunk, bytes):
                        # Interpret incoming bytes as int16 PCM
                        audio = np.frombuffer(chunk, dtype=np.int16)
                        # Ensure correct shape for channels
                        if channels > 1:
                            audio = audio.reshape(-1, channels)
                        f.write(audio.astype(np.int16))
                    else:
                        # If server sends control messages (e.g., "DONE"), you can break
                        if chunk == "DONE":
                            break
                except websockets.exceptions.ConnectionClosed:
                    break

if __name__ == "__main__":
    asyncio.run(tts_client())
