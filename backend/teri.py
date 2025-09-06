from fastapi import FastAPI, WebSocket
from piper.voice import PiperVoice
import re
import numpy as np

app = FastAPI()

# Load Piper model
MODEL_PATH = "./voices/ru/ru_RU-ruslan-medium.onnx"
voice = PiperVoice.load(MODEL_PATH)

# Buffer for unfinished text
leftover_text = ""

# Regex for SSML-like tags
TAG_PATTERN = re.compile(r"<(pause|rate|emotion)(?:=([^>]+))?>")

def synthesize_with_tags(text: str):
    """Parse SSML-like tags and yield audio chunks."""
    pos = 0
    current_rate = 1.0
    current_noise = voice.config.noise_scale
    current_noise_w = voice.config.noise_w

    for match in TAG_PATTERN.finditer(text):
        # Synthesize the text before the tag
        chunk = text[pos:match.start()].strip()
        if chunk:
            for audio_bytes in voice.synthesize_stream_raw(
                chunk,
                length_scale=current_rate,
                noise_scale=current_noise,
                noise_w=current_noise_w,
            ):
                yield audio_bytes

        tag, value = match.groups()

        if tag == "pause":
            ms = int(value or 500)
            num_silence_samples = int((ms / 1000.0) * voice.config.sample_rate)
            silence_bytes = bytes(num_silence_samples * 2)  # 16-bit PCM
            yield silence_bytes

        elif tag == "rate":
            if value == "slow":
                current_rate = 1.5
            elif value == "fast":
                current_rate = 0.7
            else:
                try:
                    current_rate = float(value)
                except:
                    current_rate = 1.0

        elif tag == "emotion":
            if value == "calm":
                current_noise = 0.2
                current_noise_w = 0.2
            elif value == "excited":
                current_noise = 1.0
                current_noise_w = 1.0
            else:
                current_noise = voice.config.noise_scale
                current_noise_w = voice.config.noise_w

        pos = match.end()

    # Synthesize any trailing text
    tail = text[pos:].strip()
    if tail:
        for audio_bytes in voice.synthesize_stream_raw(
            tail,
            length_scale=current_rate,
            noise_scale=current_noise,
            noise_w=current_noise_w,
        ):
            yield audio_bytes


@app.websocket("/tts")
async def tts_endpoint(websocket: WebSocket):
    global leftover_text
    await websocket.accept()
    try:
        while True:
            new_text = await websocket.receive_text()
            if not new_text:
                continue

            # Combine leftover with new text
            full_text = leftover_text + " " + new_text
            leftover_text = ""  # reset buffer

            print(f"Received text: {full_text}")

            # Stream synthesis with SSML-like tags
            async for chunk in async_generator(synthesize_with_tags(full_text)):
                await websocket.send_bytes(chunk)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


async def async_generator(sync_gen):
    """Helper to convert a sync generator into async for streaming."""
    for item in sync_gen:
        yield item
