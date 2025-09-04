"""
Streaming ASR server – incremental + final results
python server.py
"""

import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures
import webrtcvad
import copy
import threading
import time

# -------------------- CONFIG --------------------
MODEL_NAME          = "nvidia/stt_ru_conformer_transducer_large"
SERVER_HOST         = "0.0.0.0"
SERVER_PORT         = 8765
SAMPLE_RATE         = 16_000
FRAME_DURATION_MS   = 30
FRAME_SIZE          = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_AGGRESSIVENESS  = 2
SILENCE_TIMEOUT     = 1.5
MIN_SPEECH_DURATION = 0.5
PARTIAL_INTERVAL    = 0.40          # seconds between partial emits
# ------------------------------------------------

print("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = asr_model.to(device)
pool      = concurrent.futures.ThreadPoolExecutor(max_workers=4)
vad       = webrtcvad.Vad(VAD_AGGRESSIVENESS)
lock      = threading.Lock()
original_cfg = copy.deepcopy(asr_model.cfg.decoding)
greedy_cfg = copy.deepcopy(original_cfg)
greedy_cfg.strategy = "greedy"
greedy_cfg.compute_langs = False  # Speed up if multilingual, but this is single lang
beam_cfg = copy.deepcopy(original_cfg)
beam_cfg.strategy = "beam"
beam_cfg.beam.beam_size = 16
print("Model loaded and ready.")

# ------------- TRANSCRIPTION HELPERS -----------
def _audio_bytes_to_np(pcm_bytes: bytes) -> np.ndarray:
    pcm_i16   = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm_i16.astype(np.float32) / 32768.0

def transcribe_final(pcm_bytes: bytes) -> str:
    """Full-segment transcription with beam search."""
    if not pcm_bytes:
        return ""
    waveform = _audio_bytes_to_np(pcm_bytes)
    try:
        with torch.no_grad():
            with lock:
                asr_model.change_decoding_strategy(beam_cfg)
                hyps = asr_model.transcribe([waveform], batch_size=1)
        text = hyps[0] if isinstance(hyps[0], str) else hyps[0].text
        return text or ""
    except Exception as e:
        print("Transcription error:", e)
        return ""

def transcribe_partial(pcm_bytes: bytes) -> str:
    """Incremental transcription with greedy decoding – called every PARTIAL_INTERVAL."""
    if not pcm_bytes:
        return ""
    waveform = _audio_bytes_to_np(pcm_bytes)
    try:
        with torch.no_grad():
            with lock:
                asr_model.change_decoding_strategy(greedy_cfg)
                hyps = asr_model.transcribe([waveform], batch_size=1)
        text = hyps[0] if isinstance(hyps[0], str) else hyps[0].text
        return text or ""
    except Exception:
        return ""
# ------------------------------------------------

class SpeechBuffer:
    def __init__(self, sr, frame_ms, silence_timeout, min_speech):
        self.sr            = sr
        self.frame_bytes   = int(sr * frame_ms / 1000) * 2
        self.silence_frames_thres = int(silence_timeout * 1000 / frame_ms)
        self.min_speech_frames    = int(min_speech * 1000 / frame_ms)

        self.buffer               = bytearray()
        self.is_speaking          = False
        self.consecutive_silence  = 0
        self.speech_frames        = 0

    def add_audio(self, data):
        self.buffer.extend(data)

    def process_frames(self):
        """Yields ('frame', data) or ('speech_end',)."""
        events = []
        while len(self.buffer) >= self.frame_bytes:
            frame = self.buffer[:self.frame_bytes]
            self.buffer = self.buffer[self.frame_bytes:]

            is_speech = vad.is_speech(frame, self.sr)

            if is_speech:
                self.consecutive_silence = 0
                self.speech_frames += 1
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_frames = 1
            else:
                if self.is_speaking:
                    self.consecutive_silence += 1
                    if self.consecutive_silence > self.silence_frames_thres:
                        if self.speech_frames >= self.min_speech_frames:
                            events.append(("speech_end", None))
                        self.is_speaking = False
                        self.speech_frames = 0
                        self.consecutive_silence = 0

            if self.is_speaking:
                events.append(("frame", frame))
        return events

    def clear(self):
        self.buffer = bytearray()
        self.is_speaking = False
        self.consecutive_silence = 0
        self.speech_frames = 0

# ------------------------------------------------
async def recognize(websocket):
    print("Client connected.")
    loop = asyncio.get_running_loop()

    speech_buf = SpeechBuffer(SAMPLE_RATE, FRAME_DURATION_MS,
                              SILENCE_TIMEOUT, MIN_SPEECH_DURATION)
    current_speech = bytearray()
    last_partial_t = 0

    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                speech_buf.add_audio(msg)
                for ev, data in speech_buf.process_frames():
                    if ev == "frame":
                        current_speech.extend(data)
                        # --- streaming partial ---
                        if time.time() - last_partial_t > PARTIAL_INTERVAL:
                            txt = await loop.run_in_executor(
                                pool, partial(transcribe_partial, bytes(current_speech)))
                            if txt:
                                await websocket.send(json.dumps(
                                    {"transcript": txt, "is_final": False}))
                            last_partial_t = time.time()

                    elif ev == "speech_end":
                        if current_speech:
                            audio = bytes(current_speech)
                            current_speech = bytearray()  # Clear immediately
                            txt = await loop.run_in_executor(
                                pool, partial(transcribe_final, audio))
                            if txt:
                                await websocket.send(json.dumps(
                                    {"transcript": txt, "is_final": True}))

            elif isinstance(msg, str):
                if json.loads(msg).get("action") == "stop":
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print("Connection closed:", e.code, e.reason)
    finally:
        # Flush any leftovers
        if current_speech:
            txt = await loop.run_in_executor(
                pool, partial(transcribe_final, bytes(current_speech)))
            if txt:
                await websocket.send(json.dumps(
                    {"transcript": txt, "is_final": True, "is_complete": True}))
        print("Client disconnected.")

# ------------------------------------------------
async def main():
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"Streaming ASR server ready at ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()   # run forever

if __name__ == "__main__":
    asyncio.run(main())